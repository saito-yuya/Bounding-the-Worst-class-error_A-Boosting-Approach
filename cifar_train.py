from utils import *
import argparse
import models
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import *
from opts_cifar import parser

best_acc1 = 0
best_worst1 = 0
    
def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_worst1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        print(f"Using {args.gpu} device")
        model = model.to(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_worst1 = checkpoint['best_worst1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.early_stop != None:
        print("Eary_stopping is working")
        earlystopping = EarlyStopping(patience=args.epochs, verbose=True, path = f'{args.root_model}/{args.store_name}/{args.dataset},{args.loss_type},{args.train_rule}.pt')

    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_dataset.targets))), test_size=0.3, stratify=train_dataset.targets)
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=False, download=True, transform=transform_val)

    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_dataset.targets))), test_size=0.3, stratify=train_dataset.targets)
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    _,cls_num_list = make_label_clsnum(train_loader)
    print('cls num list:',cls_num_list)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_validation = open(os.path.join(args.root_log, args.store_name, 'log_validation.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    for epoch in range(args.start_epoch, args.epochs):
        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'fCW':
            train_sampler = None  
            per_cls_weights = torch.FloatTensor(list(map(lambda x: 1/x, cls_num_list)))
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'IBReweight':
            train_sampler = None
            per_cls_weights = 1.0 / np.array(cls_num_list)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        criterion_ib = None
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'WorstLoss':
            criterion = WorstLoss().cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        elif args.loss_type == 'IB':
            criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)
            criterion_ib = IBLoss(weight=per_cls_weights, alpha=1000).cuda(args.gpu)
        elif args.loss_type == 'IBFocal':
            criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)
            criterion_ib = IB_FocalLoss(weight=per_cls_weights, alpha=1000, gamma=1).cuda(args.gpu)
        elif args.loss_type == 'LA':
            criterion = VSLoss(cls_num_list=args.cls_num_list, tau=args.tau, gamma=0,
                               weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'CDT':
            criterion = VSLoss(cls_num_list=args.cls_num_list, tau=0, gamma=args.gamma,
                               weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'VS':
            criterion = VSLoss(cls_num_list=args.cls_num_list, tau=args.tau, gamma=args.gamma,
                               weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, criterion_ib, optimizer, epoch, args,log_training,tf_writer=None)
        
        # evaluate on validation set
        acc1,worst1 = validate(val_loader, model, criterion, criterion_ib, epoch, args,log_validation,tf_writer=None)

        if args.early_stop  != None:
            if args.stop_mode == 'average':
                earlystopping(acc1, model)  
            elif args.stop_mode == 'worst':
                earlystopping(worst1, model) 
            else:
                print("====Choose available stopping mode===")
            if earlystopping.early_stop or epoch == args.epochs - 1: 
                print("==================== Early Stopping ======================")
                print("Epochs :",epoch)
                model.load_state_dict(torch.load(f'{args.root_model}/{args.store_name}/{args.dataset},{args.loss_type},{args.train_rule}.pt'))
                validate(test_loader, model, criterion, criterion_ib, epoch, args, log_testing,tf_writer=None,flag='test')
                print("======================== All process are stopped ============================")
                sys.exit()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # remember best worst@1 and save checkpoint
        is_best_worst = worst1 > best_worst1
        best_worst1 = max(worst1, best_worst1)

        # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        output_worst = 'Worst Prec@1: %.3f\n' % (best_worst1)

        print(output_best)
        log_validation.write(output_best + '\n')

        print(output_worst)
        log_validation.write(output_worst + '\n')
        log_validation.flush()

def train(train_loader, model, criterion, criterion_ib, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    worst = AverageMeter('Worst@1', ':6.2f')
    
    # switch to train mode
    model.train()
    all_preds = []
    all_targets = []

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if 'IB' in args.loss_type and epoch >= args.start_ib_epoch:
            output, features = model(input)
            loss = criterion_ib(output, target, features)
        else:
            output, _ = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        _, pred = torch.max(output, 1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        worst_acc = torch.tensor(min(cls_acc)).to(args.gpu)
        worst.update(worst_acc, input.size(0))

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Worst@1 {worst.val:.3f} ({worst.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, worst=worst, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

def validate(val_loader, model, criterion, criterion_ib, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    worst = AverageMeter('Worst@1', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if 'IB' in args.loss_type and epoch >= args.start_ib_epoch:
                output, features = model(input)
                loss = criterion_ib(output, target, features)
                # Note that while the loss is computed using target, the target is not used for prediction.
            else:
                output, _ = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          'Worst@1 {worst.val:.3f} ({worst.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5 , worst=worst))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        worst_acc = torch.tensor(min(cls_acc)).to(args.gpu)
        worst.update(worst_acc, input.size(0))

        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Worst@1 {worst:.3f} Loss {loss.avg:.5f} '
                .format(flag=flag, top1=top1, top5=top5, worst=min(cls_acc), loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))

        print(output)
        print(out_cls_acc)

        worst3_err = [1.0-n for n in sorted(cls_acc)[0:3]]

        ## Adding
        # print('check_worst', min(out_cls_acc))

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.write('val worst 3 : ' + f'{sorted(cls_acc)[0:3]}' + '\n')
            log.write('val worst 3 err  : ' + f'{worst3_err}' + '\n')
            log.flush()
    return top1.avg,worst.avg


if __name__ == '__main__':
    main()