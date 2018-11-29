import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../CGNet"))

import numpy as np 
import logging
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import argparse
import time
import datetime
import csv
from SphericalCNN_fast import SphericalCNN_fast as SphericalCNN
from SphericalCNN_fast import SphericalResCNN_fast as SphericalResCNN
from SphericalCNN_fast import SphericalResCNN_fast2 as SphericalResCNN2

MAX_TRAIN=35764
MAX_VALID=5133
MAX_TEST=10265

parser = argparse.ArgumentParser()
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--tau_type', type=int, default=1, help='how to set tau for each layer. Choose from 1: all tau==tau_man; 2: ceil(tau_man/(2l+1)); 3: ceil(tau_man/sqrt(2l+1))')
parser.add_argument('--tau_man', type=int, default=2, help='see description of tau_type')
parser.add_argument('--logPath', default=None, help='log path')
parser.add_argument('--cuda', action="store_true", default=False, 
    help='use gpu WARNING in this version of code turning this on will cause segfault!')
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
                    help='0: no skipping; 1: concat output of all layers (l=0 for all layers); 2: ResNet 3: ResNet2')
parser.add_argument('--quick', type=float, default=1., help='use only *quick* percent data')
parser.add_argument('--dropout', type=str, default="None", help='If set, dropout in the fully connected layers with this probability. ')
parser.add_argument('--nfc', type=int, default=1, help='number of fully connected layers')

#below is specific to Shrec17 experiment
parser.add_argument("--lazy-read", action="store_true", default=False, 
    help="If set true, only reads ~3k training data in memory. Reads validation only at the end of each epoch")
parser.add_argument("--predict", action="store_true", default=False, 
    help="directly predict the results, do not train anymore")
parser.add_argument("--merged_coefs_path", type=str, default=os.path.join(CUR_DIR,"precomputed_coefs"))
parser.add_argument("--data_path", type=str, default="/media/zhen/New Volume/data")
parser.add_argument("--unmerged_coefs_path", type=str, default=os.path.join(CUR_DIR,"unmerged_coefs"))
args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        if args.tau_type == 1:
            taus = [args.tau_man for l in range(args.lmax+1)]
        elif args.tau_type == 2:
            taus = [int(np.ceil(args.tau_man/(2*l+1.))) for l in range(args.lmax+1)]
        elif args.tau_type == 3:
            taus = [int(np.ceil(args.tau_man/np.sqrt(2*l+1.))) for l in range(args.lmax+1)]

        assert(args.norm==1)
        assert(args.cuda)
        if args.skip < 2:
            self.SphericalCNN = SphericalCNN(args.lmax, taus, args.nlayers,
                skip_type=args.skip, num_channels_input=6)
        else:
            lmax_step=2
            layer_step=args.nlayers
            if args.skip == 2:
                self.SphericalCNN = SphericalResCNN(args.lmax, taus, 
                                                    lmax_step=lmax_step, 
                                                    layer_step=layer_step,
                                                    num_channels_input=6)
            else:
                self.SphericalCNN = SphericalResCNN2(args.lmax, taus, 
                                                    lmax_step=lmax_step, 
                                                    layer_step=layer_step,
                                                    num_channels_input=6)



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
        self.final = nn.Linear(256, 55)

        if args.cuda:
            self.cuda()
        
    def forward(self, x):
        x = self.SphericalCNN(x)
        #print(x.shape)
        if self.bm1 is not None:
            x = self.bm1(x)
        for layer in range(len(self.fcs)):
            x = Func.relu(self.fcs[layer](x))
            if self.dropout_layers is not None:
                x = self.dropout_layers[layer](x)
        x = nn.LogSoftmax(dim=1)(self.final(x))
        return x

def prepare_datasets(args, 
                    dataset,
                    N_TRAIN=MAX_TRAIN, 
                    N_VALID=MAX_VALID, 
                    N_TEST=MAX_TEST,
                    datasetid=None):
    #The data read are complex128 np.ndarray's
    ed = lambda x: N_TEST if "test" in x else (N_VALID if "val" in x else N_TRAIN)
    put_cuda = lambda x: x.cuda() if args.cuda else x
    if datasetid is None:
        data = np.load(os.path.join(args.merged_coefs_path, dataset + ".npy"))[:ed(dataset), :, 0:(args.lmax+1)**2]
        if dataset != "test":
            label = np.load(os.path.join(args.merged_coefs_path, dataset + "_label.npy"))[:ed(dataset)]
    else:
        coefs_path = args.unmerged_coefs_path
        f_names = [f_name for f_name in os.listdir(coefs_path) if (dataset in f_name and f_name.endswith(".npy"))]
        labels_names = [f for f in f_names if "label" in f]
        data_names = [f for f in f_names if not "label" in f]
        data_names = {int(f.split(".")[0].split("_")[3][2:]): f for f in data_names}
        k = sorted(data_names.keys())[datasetid]

        data = np.load(os.path.join(coefs_path, data_names[k]))[:, :, 0:(args.lmax+1)**2]
        label = None if dataset == "test" else np.load(os.path.join(coefs_path, data_names[k].split(".")[0] + "_label.npy"))
    if dataset != "test":
        label = put_cuda(torch.tensor(label, requires_grad=False).long())
    return data if dataset == "test" else (data, label)

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
    err  = eval_err(output, label)
    logger.info(" loss={}, error={}. time={}. ".format(loss.item(), err, datetime.datetime.now()) + extra_info)
    return loss, err

#small helper
def _update_hist(hist, cur):
    if isinstance(cur[0], float):
        hist[0].append(cur[0])
    else:
        hist[0].append(cur[0].item())
    hist[1].append(cur[1])

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
        if label is None:
            history[0].append((lambda y:y.cpu() if y.is_cuda else y)(net(data[st:ed])).detach().numpy())
        else:
            results = eval_data_and_log(net, data[st:ed], label[st:ed], criterion, logger_name=logger_name, extra_info="{}/{}".format(st, N))
            _update_hist(history, results)
    if label is None:
        logger.info("No label provided. Concatenating output...")
        return np.concatenate(history[0], axis=0)
    overall_loss = np.mean(np.asarray(history[0]))
    overall_err = np.mean(np.asarray(history[1]))
    logger.info("average loss={}, error={}. ".format(overall_loss, overall_err) + extra_info)
    return overall_loss, overall_err


#code is from MPNN 
def save_net(state, is_best, directory):
    import shutil
    if not os.path.isdir(directory): os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'last_check_point.pth')
    best_net_file = os.path.join(directory, 'net_best.pth')
    torch.save(state, checkpoint_file)
    if is_best: shutil.copyfile(checkpoint_file, best_net_file)

def main(args=args, N_TRAIN=MAX_TRAIN, N_VALID=MAX_VALID, N_TEST=MAX_TEST, save_period=1000 if args.batch_size < 60 else 3000):
    #log_name = args.logPath.split("/")[-1].split(".")[0]
    logging.basicConfig(filename=args.logPath, level=logging.INFO)

    logger = logging.getLogger("log_main")
    
    net = Net(args)
    logger.debug('Optimizer')
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=args.lr, weight_decay=args.weight_decay)#added weigth decay

    best_err = 1.0
    cur_err = 1.0
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

    criterion = nn.CrossEntropyLoss()

        
    train_history = [[], []]#[0] is history of losses, [1] is errors
    valid_history = [[], []]
    
    if not args.predict:
        logging.info("Starting to train at {}".format(datetime.datetime.now()))
        for epoch in range(args.start_epoch, args.num_epoch):
            #if not args.lazy_read:
            data_train,label_train=prepare_datasets(args,"train",N_TRAIN=N_TRAIN, datasetid=0 if args.lazy_read else None)
            #N_TRAIN = data_train.shape[0]
            last_save_pt = 0
            net.train()
            """for st in range(args.st, N_TRAIN, args.batch_size):"""
            st = args.st
            while (st < N_TRAIN - 1):
                ed = min(N_TRAIN, st+args.batch_size)

                if args.lazy_read:
                    offset_st = st % 3000
                    offset_ed = ed % 3000
                    if offset_ed < offset_st:
                        #means we are crossing batches
                        offset_ed += 3000
                        st += -args.batch_size + 3000 - offset_st
                    if offset_st < args.batch_size:
                        del data_train, label_train
                        data_train,label_train=prepare_datasets(args,"train",N_TRAIN=N_TRAIN, datasetid=st//3000)

                else: #easy in this case
                    offset_st = st
                    offset_ed = ed


                optimizer.zero_grad()
                results = eval_data_and_log(net, data_train[offset_st:offset_ed], label_train[offset_st:offset_ed], criterion, logger_name="log_train", extra_info="{}/{}".format(st, N_TRAIN))
                _update_hist(train_history, results)
                results[0].backward()
                optimizer.step()

                st += args.batch_size
                if (ed - last_save_pt >= save_period) and st > 0:
                    logging.info("Saving at {}: best_err={}".format(datetime.datetime.now(), best_err))
                    #last_save_pt = ed
                    last_save_pt = st
                    save_net({'epoch': epoch, 'state_dict': net.state_dict(), 'best_err': min(cur_err, best_err),
                                    'optimizer': optimizer.state_dict(), 'st':last_save_pt}, False, args.resume)
            del data_train, label_train

            #eval on validation set and save
            net.eval()
            data_valid, label_valid = prepare_datasets(args,"val", N_VALID=N_VALID)
            N_VALID = data_valid.shape[0]
            results = eval_data_in_batch(net, data_valid, label_valid,criterion,logger_name="log_valid",batch_size=args.batch_size)
            _update_hist(valid_history, results)
            del data_valid, label_valid

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
    
    
    if N_TEST >= 500:
        net.eval()
        data_test = prepare_datasets(args,"test")
        test_pred = eval_data_in_batch(net, data_test, None, None, logger_name="log_test", batch_size=args.batch_size)
        np.save(os.path.join(args.resume, "prediction.npy"), test_pred)
    
    if args.predict:
        import precomputation
        import shutil
        temp_eval_dir = os.path.abspath("./eval_temps/{}/".format(os.path.splitext(os.path.basename(args.logPath))[0])) 
        print(temp_eval_dir)
        precomputation.evaluate(probvector_path=os.path.join(args.resume, "prediction.npy"),
                                source_data_path=args.data_path,
                                eval_dir=temp_eval_dir)

    return train_history, valid_history
#newest

if __name__ == "__main__":
    assert(args.norm >= 0 and args.norm <= 2)
    assert(args.tau_type >= 0 and args.tau_type <= 3)

    args.dropout = None if args.dropout == "None" else float(args.dropout)

    if args.skip >= 2:
        train_name = "ResNet{}_epoch{}_lr{}_wd{}_layerstep{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}".format("2" if args.skip == 3 else "", 
                            args.num_epoch, args.lr,args.weight_decay, args.nlayers, args.tau_type, args.tau_man, args.lmax, args.batch_size, args.norm, args.nfc)
    else:
        train_name = "epoch{}_lr{}_wd{}_layer{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}".format(args.num_epoch, args.lr,args.weight_decay, args.nlayers, args.tau_type, args.tau_man, args.lmax, args.batch_size, args.norm, args.nfc)
        train_name += "" if args.skip == 0 else "_connect-to-output"
    train_name += "_new"
    train_name += "" if args.dropout is None else "_dropout{}".format(args.dropout) 
    
    save_path = CUR_DIR
    if args.logPath is None:
        args.logPath = os.path.join(save_path, "temp_new/logs/{}.log".format(train_name) )
    if args.resume is None:
        args.resume = os.path.join(save_path, "temp_new/checkpoint/{}/".format(train_name))

    log_dir = os.path.dirname(args.logPath)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    ckpt_dir = os.path.dirname(args.resume)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    main(args, N_TRAIN=int(MAX_TRAIN * args.quick), N_VALID=max(1000,int(MAX_VALID*args.quick)), N_TEST=int(MAX_TEST * args.quick))