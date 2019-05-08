import numpy as np 
import datautils
import logging
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import argparse
import os
import time
import datetime
import sys

print("Cuda available?", torch.cuda.is_available())
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MNIST_DATA_PATH = os.path.join(CUR_DIR, "temp/")
sys.path.append(os.path.join(CUR_DIR, "../CGNet"))
from SphericalCNN import SphericalCNN as SphericalCNN


parser = argparse.ArgumentParser(description='train the network')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers if not ResNet; number of layer for each lmax in ResNet')
parser.add_argument('--tau_type', type=int, default=1, help='how to set tau for each layer. Choose from 1: all tau==tau_man; 2: ceil(tau_man/(2l+1)); 3: ceil(tau_man/sqrt(2l+1))')
parser.add_argument('--tau_man', type=int, default=2, help='see description of tau_type')
parser.add_argument('--logPath', default=None, help='log path')
parser.add_argument('--cuda', action="store_true", default=False, help='use gpu WARNING in this version of code turning this on will cause segfault!')
parser.add_argument('--norm', type=int, default=0, 
                        help='normalization strategy: \n' + \
                        '0: no normlization \n' + \
                        '1: batch-normalization in each layer by scaling each fragment down by a moving average of standard deviation\n')
parser.add_argument('--resume', default=None,
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=20, 
                    help='Input batch size for training (default: 20)')
parser.add_argument('--num-epoch', type=int, default=20, 
                    help='Number of epochs (default: 20)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='Initial learning rate (default: 5e-4)')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help='weight_decay rate (default: 1e-5)')
parser.add_argument('--lmax', type=int, default=9, 
                    help='lmax')
parser.add_argument('--skip', type=int, default=0, 
                    help='0: no skipping; 1: concat output of all layers (l=0 for all layers); 2: ResNet')
parser.add_argument('--quick', type=float, default=1., help='use only *quick* percent data')
parser.add_argument('--dropout', type=str, default="None", help='If set, dropout in the fully connected layers with this probability. ')
parser.add_argument('--nfc', type=int, default=1, help='number of fully connected layers')


parser.add_argument('--unrot-test', action="store_true", default=False, help='if True, measure on unrotated test set')

parser.add_argument('--rotate-train', action="store_true", default=False, help='train on rotated training set')

#python main.py --nlayers 5 --tau_type 2 --batch-size 5 --num-epoch 2 --lmax 3 --batch-norm



class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()


        if args.tau_type == 1:
            taus = [args.tau_man for l in range(args.lmax+1)]
        elif args.tau_type == 2:
            taus = [int(np.ceil(args.tau_man/(2*l+1.))) for l in range(args.lmax+1)]
        elif args.tau_type == 3:
            taus = [int(np.ceil(args.tau_man/np.sqrt(2*l+1.))) for l in range(args.lmax+1)]
        if args.skip != 2:
            self.SphericalCNN = SphericalCNN(args.lmax, taus, args.nlayers,
                                                cudaFlag=args.cuda, 
                                                normalization=args.norm, 
                                                skip_type=args.skip,
                                                #use_relu=args.relu,
                                                num_channels_input=1)
        else:
            lmax_step=2
            layer_step=args.nlayers
            self.SphericalCNN = SphericalResCNN(args.lmax, taus, 
                                                    lmax_step=lmax_step, 
                                                    layer_step=layer_step,
                                                    num_channels_input=1)
        """

        if args.tau_type == 1:
            taus = args.tau_man
        elif args.tau_type == 2:
            taus = [[int(np.ceil(args.tau_man/(2*l+1.))) for l in range(args.lmax+1)] for layer in range(args.nlayers)]
        elif args.tau_type == 3:
            taus = [[int(np.ceil(args.tau_man/np.sqrt(2*l+1.))) for l in range(args.lmax+1)] for layer in range(args.nlayers)]
            
        if args.skip != 2:
            print(taus)
            self.SphericalCNN = SphericalCNN(args.lmax, taus, args.nlayers,
                                                cudaFlag=args.cuda, 
                                                normalization=args.norm, 
                                                skip_type=args.skip,
                                                use_relu=args.relu)
        else :
            lmax_step=2
            layer_step=args.nlayers
            taus_pre = taus[0]
            taus = []
            for L in range(args.lmax, lmax_step, -lmax_step):
                taus = [[[x for x in taus_pre[0:(L+1)]]] * layer_step] + taus
            print(taus)
            self.SphericalCNN = SphericalResCNN(args.lmax,lmax_step,layer_step,taus,
                                                cudaFlag=args.cuda,
                                                normalization=args.norm,
                                                use_relu=args.relu)
        """

        if args.norm != 0:
            self.bm1 = nn.BatchNorm1d(self.SphericalCNN.output_length)
        else:
            self.bm1 = None

        self.fcs = nn.ModuleList([])
        self.dropout_layers = None if args.dropout is None else nn.ModuleList([])
        for layer in range(args.nfc):
            self.fcs.append(nn.Linear(self.SphericalCNN.output_length if layer == 0 else 256,256))
            if args.dropout is not None:
                self.dropout_layers.append(nn.Dropout(p=args.dropout))
        print("Fully connected layser\n", self.fcs)
        print("Paired with dropout layers? \n", self.dropout_layers)
        self.final = nn.Linear(256, 10)
        if args.cuda:
            self.cuda()
        
    def forward(self, x):
        x = self.SphericalCNN(x)
        if self.bm1 is not None:
            x = self.bm1(x)
        for layer in range(len(self.fcs)):
            x = Func.relu(self.fcs[layer](x))
            if self.dropout_layers is not None:
                x = self.dropout_layers[layer](x)
        x = nn.LogSoftmax(dim=1)(self.final(x))
        return x
    
def prepare_datasets(args, N_TRAIN=60000,N_TEST=10000,train_ratios=[11.,1.,0.]):
    global MNIST_DATA_PATH
    all_data = datautils.precomputing_coefs(MNIST_DATA_PATH, args.lmax, N_TRAIN, coeftype="train_rotate" if args.rotate_train else "train")
    #print(all_imag.shape)
    data_test = datautils.precomputing_coefs(MNIST_DATA_PATH, args.lmax, N_TEST,
                                                        data_file_name="t10k-images.idx3-ubyte",
                                                        coeftype="test" if args.unrot_test else "test_rotate")
    all_labels = datautils.extract_labels(MNIST_DATA_PATH+ "train-labels.idx1-ubyte", N_TRAIN)
    label_test = datautils.extract_labels(MNIST_DATA_PATH+"t10k-labels.idx1-ubyte", N_TEST)

    np.random.seed(0)
    idx = np.random.permutation(N_TRAIN)

    data_train, data_valid, _ = datautils.split_data(all_data[idx], ratios=train_ratios)
    label_train, label_valid, _ = datautils.split_data(all_labels[idx], ratios=train_ratios)

    label_train = torch.tensor(label_train, requires_grad=False).long()
    label_valid = torch.tensor(label_valid, requires_grad=False).long()
    label_test = torch.tensor(label_test, requires_grad=False).long()
    if args.cuda:
        label_train = label_train.cuda()
        label_valid = label_valid.cuda()
        label_test  = label_test.cuda()
    #print(data_train.dtype, data_valid.shape, data_test.shape)
    return data_train, label_train, data_valid, label_valid, data_test, label_test



def eval_data_and_log(net, data, label, criterion, eval_err=None, logger_name="log_train", extra_info=""):
    def eval_pred_error(output, target):
        n = target.size(0)
        predict = torch.max(output.data, 1)[1].squeeze()
        t = torch.eq(predict, target.data).float()
        return 1. - t.sum()/float(n)
    if eval_err is None:
        eval_err = eval_pred_error
    logger = logging.getLogger(logger_name)
    output = net(data)
    loss = criterion(output, label)
    #print(loss)
    err  = eval_err(output, label)
    logger.info(" loss={}, error={}. time={}. ".format(loss.item(), err, datetime.datetime.now()) + extra_info)
    return loss, err

#small helper
def _update_hist(hist, cur):
    _get_float = lambda x: x if isinstance(x, float) else x.item()
    hist[0].append(_get_float(cur[0]))
    hist[1].append(_get_float(cur[1]))

def eval_data_in_batch(net, data, label, criterion, eval_err=None, extra_info="", logger_name="log_train", batch_size=10, CAP=None):
    logger = logging.getLogger(logger_name)
    logger.info("Training? {}".format(net.training))
    #print batch_size, "AAAA"
    history = [[],[]]
    N = data.shape[0] if CAP is None else CAP
    for st in range(0, N, batch_size):
        ed = min(N, st+batch_size)
        if ed <= st:
            break
        results = eval_data_and_log(net, data[st:ed], label[st:ed], criterion, logger_name=logger_name, extra_info="{}/{}".format(st, N))
        _update_hist(history, results)
    overall_loss = np.mean(np.asarray(history[0]))
    overall_err = np.mean(np.asarray(history[1]))
    logger.info("average loss={}, error={}. ".format(overall_loss, overall_err) + extra_info)
    return overall_loss, overall_err


#from MPNN 
def save_net(state, is_best, directory):
    import shutil
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'last_check_point.pth')
    best_net_file = os.path.join(directory, 'net_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_net_file)


def main(args, N_TRAIN=60000, N_TEST=10000, save_period=4000):
    #log_name = args.logPath.split("/")[-1].split(".")[0]
    logging.basicConfig(filename=args.logPath, level=logging.INFO)
    data_train,label_train,data_valid,label_valid,data_test,label_test=prepare_datasets(args,N_TRAIN=N_TRAIN,N_TEST=N_TEST)
    
    expand_channel_dim = lambda x: np.expand_dims(x,1) if len(x.shape)==2 else x
    data_train, data_valid, data_test = expand_channel_dim(data_train), expand_channel_dim(data_valid), expand_channel_dim(data_test)
    N_TRAIN = data_train.shape[0]
    N_VALID = data_valid.shape[0]

    logger = logging.getLogger("log_main")
    
    torch.manual_seed(1)
    net = Net(args)
    logger.debug('Optimizer')
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=args.lr, weight_decay=args.weight_decay)#added weigth decay
    shapes = [(lambda y: y.cpu() if y.is_cuda else y)(x).detach().numpy().shape for x in list(filter(lambda p: p.requires_grad, net.parameters()))]
    def get_volume(shape):
        v = 1
        for i in range(len(shape)):
            v *= shape[i]
        return v
    print("Shapes of parameters", shapes, np.sum(np.asarray([get_volume(s) for s in shapes])))
    best_err = 1.0
    cur_err = 1.0
    #return None
    #from MPNN  
    # get the best checkpoint if available without training
    args.start_epoch = 0
    args.st = 0
    if args.resume:
        checkpoint_dir = args.resume
        last_net_file = os.path.join(checkpoint_dir, 'last_check_point.pth')
        if os.path.isfile(last_net_file):
            logger.info("=> loading last net '{}'".format(last_net_file))
            checkpoint = torch.load(last_net_file)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_err']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            args.st = checkpoint['st']
            logger.info("=> loaded last net '{}' (epoch {}, st={})".format(last_net_file, checkpoint['epoch'], args.st))
        else:
            logger.info("=> no stored net found at '{}'".format(last_net_file))

    #for param in net.parameters():
    #    print(type(param.data), param.size())
    criterion = nn.CrossEntropyLoss()

        
    train_history = [[], []]#[0] is history of losses, [1] is errors
    valid_history = [[], []]
    #net.eval()
    #results = eval_data_in_batch(net, data_valid, label_valid,criterion,logger_name="log_valid",batch_size=args.batch_size)
    #_update_hist(valid_history, results)
    for epoch in range(args.start_epoch, args.num_epoch):
        last_save_pt = 0
        net.train()
        for st in range(args.st, N_TRAIN, args.batch_size):
            ed = min(N_TRAIN, st+args.batch_size)
            if ed <= st:
                break
            optimizer.zero_grad()
            results = eval_data_and_log(net, data_train[st:ed], label_train[st:ed], criterion, logger_name="log_train", extra_info="{}/{}".format(st, N_TRAIN))
            _update_hist(train_history, results)
            results[0].backward()
            #print(net.SphericalCNN.us[0].ws[0].grad[0,0:10,:])
            #print(1/0)
            optimizer.step()

            """
            if (not args.turn_off_rotate) and ((ed - last_save_pt >= save_period) and st > 0):
                net.eval()
                last_save_pt = ed
                eval_data_in_batch(net, data_test, label_test,criterion,logger_name="log_rotate",batch_size=args.batch_size, CAP=200)[1]
                #saving periodically
                save_net({'epoch': epoch, 'state_dict': net.state_dict(), 'best_err': min(cur_err, best_err),
                                'optimizer': optimizer.state_dict(), 'st':last_save_pt}, False, args.resume)
                net.train()
            """
            if (ed - last_save_pt >= save_period) and st > 0:
                logging.info("Saving at {}: best_err={}".format(datetime.datetime.now(), best_err))
                last_save_pt = ed
                save_net({'epoch': epoch, 'state_dict': net.state_dict(), 'best_err': min(cur_err, best_err),
                                'optimizer': optimizer.state_dict(), 'st':last_save_pt}, False, args.resume)

        #eval on validation set and save
        net.eval()
        results = eval_data_in_batch(net, data_valid, label_valid,criterion,logger_name="log_valid",batch_size=args.batch_size)
        _update_hist(valid_history, results)
        cur_err = valid_history[1][-1]
        save_net({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_err': min(cur_err, best_err),
                            'optimizer': optimizer.state_dict(), 'st':0}, cur_err < best_err, args.resume)
        if cur_err < best_err:
            best_err = cur_err
        elif cur_err > best_err * 2:
            logging.info("epoch {}: validation error {} > best error {} a lot, quitting as it might be overfitting".format(epoch, cur_err, best_err))
            break
        logging.info("epoch {} done, Time{}".format(epoch, datetime.datetime.now()))
        args.st = 0


    if args.resume:
        checkpoint_dir = args.resume
        best_net_file = os.path.join(checkpoint_dir, 'net_best.pth')
        if os.path.isfile(best_net_file):
            logger.info("=> loading best net '{}'".format(best_net_file))
            checkpoint = torch.load(best_net_file)
            best_err = checkpoint['best_err']
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded best net with valid_err={}".format(best_err))
        else:
            logger.info("=> no best net found at '{}'".format(best_net_file))

    net.eval()
    print(data_test.shape)
    test_error = eval_data_in_batch(net, data_test, label_test,criterion,logger_name="log_test",batch_size=args.batch_size)[1]
    logger.info("FINAL test error = {}".format(test_error))
    return train_history, valid_history
#newest


if __name__ == "__main__":
    args = parser.parse_args()

    args.dropout = None if args.dropout == "None" else float(args.dropout)


    if args.skip == 2:
        train_name = "ResNet_epoch{}_lr{}_wd{}_layerstep{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}".format(args.num_epoch, args.lr,args.weight_decay, args.nlayers, args.tau_type, args.tau_man, args.lmax, args.batch_size, args.norm, args.nfc)
    else:
        train_name = "epoch{}_lr{}_wd{}_layer{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}".format(args.num_epoch, args.lr,args.weight_decay, args.nlayers, args.tau_type, args.tau_man, args.lmax, args.batch_size, args.norm, args.nfc)
        train_name += "" if args.skip == 0 else "_connect-to-output"
    train_name += "_new"
    if args.rotate_train and (not args.unrot_test):
        train_name += "R-R"
    elif (not args.rotate_train) and (not args.unrot_test):
        train_name += "NR-R"
    else:
        assert((not args.rotate_train) and args.unrot_test)
        train_name += "NR-NR"
    train_name += "" if args.dropout is None else "_dropout{}".format(args.dropout) 

    if args.logPath is None:
        args.logPath = CUR_DIR + "/temp/logs/{}.log".format(train_name) 
    if args.resume is None:
        args.resume = CUR_DIR + "/temp/checkpoint/{}/".format(train_name)
    log_dir = os.path.dirname(args.logPath)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    ckpt_dir = os.path.dirname(args.resume)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    main(args, N_TRAIN=int(60000 * args.quick), N_TEST=int(10000 * args.quick))
    #test_exploding_weights(args, N_TRAIN=int(600), N_TEST=int(100))
   # main(epoch=1)
