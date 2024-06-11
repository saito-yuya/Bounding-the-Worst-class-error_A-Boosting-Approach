
from utils import *
import models
from imbalance_cifar import IMBALANCECIFAR10,IMBALANCECIFAR100
from opts_cifar import get_args
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

if __name__ == '__main__':
    args = get_args()
    theta = args.theta
    num_classes = args.num_classes
    gamma =  (math.floor(0.8*num_classes)/num_classes) - (1/2) - args.eps
    torch_fix_seed(args.seed)
    args.store_name = '_'.join([args.dataset, args.arch,args.imb_type,str(args.imb_factor), str(theta), str(gamma)])
    prepare_folders(args)

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    print("theta:" ,args.theta)
    print("device:",args.gpu)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_val = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    if args.dataset == 'cifar10':
        train_set = IMBALANCECIFAR10("./data", imb_factor=args.imb_factor,download=True,train=True,rand_number=args.seed, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_set.targets))), test_size=args.test_size, stratify=train_set.targets)
        val_set = torch.utils.data.Subset(train_set, val_indices)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        test_set = IMBALANCECIFAR10("./data", imb_factor=args.imb_factor,download=True,train=False,rand_number=args.seed, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_set = IMBALANCECIFAR100("./data", imb_factor=args.imb_factor,download=True,train=True,rand_number=args.seed, transform=transform_train)
        train_indices, val_indices = train_test_split(list(range(len(train_set.targets))), test_size=args.test_size, stratify=train_set.targets)
        val_set = torch.utils.data.Subset(train_set, val_indices)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        test_set = IMBALANCECIFAR100("./data", imb_factor=args.imb_factor,download=True,train=False,rand_number=args.seed, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False, num_workers=args.workers)

    labal,cls_num_list = make_label_clsnum(train_loader)
    # print("Train : cls_num_list",cls_num_list)

    # init log for training
    log_training_txt = open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), 'w')
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing_txt = open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))


    model_path = os.path.join(args.root_model, args.store_name)

    # train_err_list = []
    weight,weak_preds = [],[]

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=False)
    torch.save(model.state_dict(), model_path + '/check_point.pt')
    
    if args.gpu is not None:
        # torch.cuda.set_device(args.gpu)
        # model = model.cuda(args.gpu)
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()


    #round Number(The number of weak-learner)
    round_num = math.ceil(2*math.log(num_classes)/(gamma)**2)
    # round_num = 3

    weight_tmp = torch.tensor([1/num_classes]*num_classes)
    weight.append(weight_tmp.to('cpu').detach().numpy().copy().tolist())

    criterion = nn.CrossEntropyLoss(weight=weight_tmp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    OP_init(model,train_loader,optimizer,weight_tmp,args.loss_type,device)

    for t in range(round_num):
        model = Network_init(model,model_path + '/check_point.pt',device)
        train_acc,weak_model,y_preds,rt = train(model,train_loader,num_classes,weight_tmp,optimizer,args.loss_type,args.max_epoch,theta,gamma,log_training,device)
        
        torch.save(weak_model.state_dict(), model_path + f'/weak_model({t}).pt')

        weight_tmp = Hedge(weight_tmp,rt,num_classes,round_num,device)
        weight.append(weight_tmp.to('cpu').detach().numpy().copy().tolist())

    print("############################ Finish Main roop ##############################")

    # print(weight)

    # weight_show(weight,classes,f'{args.root_log}/{args.store_name}/weight.pdf')

    ## validation 
    val_label,cls_num_list = make_label_clsnum(val_loader)

    for t in tqdm(range(round_num),leave=False):
        tmp_h = []
        model.load_state_dict(torch.load(model_path + f'/weak_model({t}).pt'))
        model  = model.to(device)
        check_accuracy,tmp_h,_,_  = class_wise_acc(model,val_loader,device)
        weak_preds.append(tmp_h)
    res = best_N(weak_preds,val_label)
    ave_accuracy = ave(weak_preds,val_label)
    worst,n,idx = worst_val_idx(res)


    print("val_accuracy :",ave_accuracy[n])
    print("val_worst :",worst)
    print("number of models :",n)
    print("worst_idx :",idx)

    with open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), "w") as train_f:
        print(f"dataset : {args.dataset}",file=train_f)
        print(f"Imb_factor : {args.imb_factor}",file=train_f)
        print(f"theta,gamma = {theta,gamma}",file=train_f)
        print("===================Validation===================",file=train_f)
    
    with open(os.path.join(args.root_log, args.store_name, 'log_train.txt'),"a") as train_f:
        print("best number of models :",n+1,file=train_f)
        print("val_accuracy :",res[n],file=train_f)
        print("val_Average_accuracy :",ave_accuracy[n],file=train_f)
        print("val_worst :",worst,file=train_f)
        print("Allmodel_worst_class_idx :",idx,file=train_f)
        print("worst_class_idx :",idx[n],file=train_f)
        print("Weight : ",weight,file=train_f)

    
#################
    ## Test
    test_label,cls_num_list = make_label_clsnum(test_loader)

    weak_preds = []

    for t in tqdm(range(n+1)):
        tmp_h = []
        model.load_state_dict(torch.load(model_path + f'/weak_model({t}).pt'))
        check_accuracy,tmp_h,_,_  = class_wise_acc(model,test_loader,device)
        weak_preds.append(tmp_h)

    ## 全モデルに対するaccuracy_list
    # res = best_N(weak_preds,test_label)
    test_h = voting(transposition(weak_preds))

    _,_,_,test_correct  = class_wise_acc(model,test_loader,device)

    ans = class_wise_acc_h(test_h,test_label)

    # print('ans : ',ans)
    print('classification report', classification_report(test_label,test_h))

    test_out = []
    for i in range(num_classes):
        class_acc = (ans[i] / test_correct[i])
        test_out.append(class_acc)
    print(test_out)
    print("test_Average_accuracy",sum(ans.values())/sum(test_correct))
    print("Worst_class_accuracy :" , min(test_out))

# 複数の最大値のインデックスを全て取得
    idx = [i for i, v in enumerate(test_out) if v == min(test_out)]
    print("test_worst_idx :",idx)

    with open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), "w") as test_f:
        print(f"dataset : {args.dataset}",file=test_f)
        print(f"Imb_factor : {args.imb_factor}",file=test_f)
        print(f"theta,gamma = {theta,gamma}",file=test_f)
        print("===================Test===================",file=test_f)
    
    with open(os.path.join(args.root_log, args.store_name, 'log_test.txt'), "a") as test_f:
        print("Test_accuracy :",test_out,file=test_f)
        print('classification report', classification_report(test_label,test_h),file=test_f)
        print("Test_Average_accuracy",sum(ans.values())/sum(test_correct),file=test_f)
        print("Worst_3class_accuracy :" , sorted(test_out)[0:3],file=test_f)
        print("Worst_class_accuracy :" , min(test_out),file=test_f)
        print("test_worst_idx :",idx,file=test_f)

    print("--------------------------------------Finish---------------------------------------------------")
