# pylint: disable=E1101,R,C,W1202
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import numpy as np
from lie_learn.spaces import S2

import os
import shutil
import time
import logging
import copy
import types
import argparse
import random 
import datetime
import requests
from subprocess import check_output
import zipfile

import sys

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
#from dataset_slurm import Shrec17, CacheNPY, ToMesh, ProjectOnSphere, make_sgrid
from dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere, make_sgrid
sys.path.append(os.path.join(CUR_DIR, "../CGNet"))
import geometries
import pickle
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--augmentation", type=int, default=1,
                    help="Generate multiple image with random rotations and translations")
parser.add_argument("--dataset", choices={"test", "val", "train"}, default="train")

parser.add_argument("--bw", type=int, default=128)
parser.add_argument("--lmax", type=int, default=40)
parser.add_argument("--start_position", type=int, default=0)
parser.add_argument("--end_position", type=int, default=205)
parser.add_argument("--coefs_path", type=str, default=os.path.join(CUR_DIR,"unmerged_coefs"))
parser.add_argument("--data_path", type=str, default="/media/zhen/New Volume/data")
parser.add_argument("--merge", action="store_true", default=False)
parser.add_argument("--merged_coefs_path", type=str, default=os.path.join(CUR_DIR,"precomputed_coefs"))
parser.add_argument("--slurm", action="store_true", default=False)

parser.add_argument("--predict", action="store_true", default=False)
parser.add_argument("--predict_file", type=str, default="./prediction.npy")
parser.add_argument("--eval_dir", type=str, default="./eval/")

def merge(args):
    f_names = [f_name for f_name in os.listdir(args.coefs_path) if (args.dataset in f_name and f_name.endswith(".npy"))]
    labels_names = [f for f in f_names if "label" in f]
    f_names = [f for f in f_names if not "label" in f]
    print(labels_names, f_names)
    f_names = {int(f.split(".")[0].split("_")[3][2:]): f for f in f_names}
    print("Start ", datetime.datetime.now())

    data_to_cat = []
    labels_to_cat = []
    for k in sorted(f_names.keys()):
        data_file_path = os.path.join(args.coefs_path, f_names[k])
        data_to_cat.append(np.load(data_file_path))
        if args.dataset != "test":
            label_file_name = f_names[k].split(".")[0] + "_label.npy"
            assert(label_file_name in labels_names)
            label_file_path = os.path.join(args.coefs_path, label_file_name)
            labels_to_cat.append(np.load(label_file_path))
    safe_cat = lambda x: np.concatenate(x, axis=0) if len(x) > 1 else x
    if not os.path.isdir(args.merged_coefs_path):
        os.makedirs(args.merged_coefs_path)
    data_catted = safe_cat(data_to_cat)
    np.save(os.path.join(args.merged_coefs_path, "{}.npy".format(args.dataset)), data_catted)
    if args.dataset != "test":
        labels_catted = safe_cat(labels_to_cat)
        np.save(os.path.join(args.merged_coefs_path, "{}_label.npy".format(args.dataset)), labels_catted)
    print("Done ", datetime.datetime.now())
    #return None

def cache_train_data(args, random_rotate=False):

    logger = logging.getLogger("datautils_logger")


    ## Load the dataset
    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(prefix="b{}_".format(args.bw), repeat=args.augmentation, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=random_rotate, random_translation=0.1),
            ProjectOnSphere(bandwidth=args.bw)
        ]
    ))
    #I think pick_randomly does not make any difference in the case of augment==1

    def target_transform(x):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        return classes.index(x[0])
    #transform = None
    data_to_cache = Shrec17(args.data_path, args.dataset, perturbed=True, download=True, transform=transform, target_transform=target_transform)
    logger.info("There are {} data".format(len(data_to_cache)))
    
    if random_rotate:
        logger.info("random rotate..")
    num_imgs = len(data_to_cache)
    num_channels = data_to_cache[0].shape[0] if args.dataset=="test" else data_to_cache[0][0].shape[0]

    save_path = os.path.join(args.coefs_path,"{}.npy".format(args.task_name))
    save_label_path = os.path.join(args.coefs_path,"{}_label.npy".format(args.task_name))
    if False and os.path.exists(save_path):
        coefs = np.load(save_path)
        labels = np.load(save_label_path)
        start = coefs.shape[0] + args.start_position
    else:
        if not os.path.exists(args.coefs_path):
            os.makedirs(args.coefs_path)
        coefs = None
        labels = None
        start = args.start_position
    logger.info("Read {} computed data".format(0 if coefs is None else coefs.shape[0]))
    logger.info("Starting from {}, to {} inclusive".format(start,args.end_position))

    sphs = [None for c in range(num_channels)]
    grid = S2.change_coordinates(make_sgrid(args.bw, 0, 0, 0), 'C', 'S')
    fs = []
    ls = []
    for i in range(start, args.end_position+1):
        f = np.zeros((num_channels, (1+args.lmax)**2), dtype=complex)
        if args.dataset == "test":
            img = data_to_cache[i]
        else:
            img, label = data_to_cache[i]
            ls.append(label)
        img = img.reshape(img.shape[0], 4 * args.bw ** 2)
        for channel in range(num_channels):
            """f[channel, :], sphs[channel] = geometries.get_coef_C(img[channel, :], grid[:,0], grid[:,1], lmax=args.lmax, sph=None if random_rotate else sphs[channel])"""
            f[channel, :], sphs[channel] = geometries.get_coef_C(img[channel, :], grid[:,0], grid[:,1], lmax=args.lmax, sph=sphs[channel])
        fs.append(f)
        logger.info("Working on {}/{} at time {}".format(i, args.end_position, datetime.datetime.now()))
        if ((i+1) % 100 == 0 or i == args.end_position) and i > 0:
            fs = np.stack(fs, axis=0)
            coefs = fs if coefs is None else np.concatenate([coefs, fs], axis=0)
            np.save(save_path, coefs)
            fs = []
            if args.dataset != "test":
                ls = np.stack(ls, axis=0)
                #print(ls)
                labels = ls if labels is None else np.concatenate([labels, ls], axis=0)
                np.save(save_label_path, labels)
                print(save_label_path)
                ls = []
            logger.info("save at time{}".format(datetime.datetime.now()))

def evaluate(probvector_path,source_data_path,eval_dir):

    prob = np.load(probvector_path)
    data_to_cache = Shrec17(source_data_path, 'test', perturbed=True, download=True, transform=None)
    print(prob.shape)
    get_file_name = lambda x: x.split("/")[-1].split(".")[0]
    classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
    
    basepath = os.path.abspath(eval_dir)
    basename = os.path.basename(os.path.abspath(eval_dir))
    eval_dir = os.path.join(eval_dir, "test_perturbed")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    
    pred = np.argmax(prob,axis=1)
    ids = [get_file_name(data_to_cache[i]) for i in range(len(pred))]
    #i = 0
    for i in range(len(ids)):
        idfile = os.path.join(eval_dir, ids[i])

        retrieved = [(prob[j, pred[j]], ids[j]) for j in range(len(ids)) if pred[j] == pred[i]]
        retrieved = sorted(retrieved, reverse=True)
        retrieved = [s for _, s in retrieved]
        #print(retrieved[:100])
        with open(idfile, 'w') as f:
            f.write("\n".join(retrieved))
        if i % 100 == 99:
            print("{}/{}".format(i, len(ids)))

    
    url = "https://shapenet.cs.stanford.edu/shrec17/code/evaluator.zip"
    file_path = "evaluator.zip"

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(".")
    zip_ref.close()

    #log_dir = os.path.abspath(os.path.join(eval_dir,os.path.pardir))
    print(basepath)
    print(check_output(["nodejs", "evaluate.js", basepath + "/"], cwd='evaluator').decode("utf-8"))
    #basename = os.path.basename(os.path.abspath(eval_dir))
    print(os.path.join("evaluator", basename + ".summary.csv"))
    print(os.path.join(basepath, "summary.csv"))
    shutil.copy2(os.path.join("evaluator", basename + ".summary.csv"), os.path.join(basepath, "summary.csv"))

    return None

if __name__ == "__main__":
    args = parser.parse_args()

    if args.slurm:
        args.coefs_path = "/share/data/vision-greg/shubhendu/Shrec17/coefs"
        args.merged_coefs_path = "/share/data/vision-greg/shubhendu/Shrec17/precomputed_coefs"
        args.data_path = "/scratch/shubhendu/data_fast"
        args.eval_dir = "/scratch/shubhendu/Shrec17/eval/"
        from dataset_slurm import Shrec17, CacheNPY, ToMesh, ProjectOnSphere, make_sgrid
    else:
        from dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere, make_sgrid


    if args.merge:
        merge(args)
    elif args.predict:
        evaluate(probvector_path=args.predict_file,source_data_path=args.data_path,eval_dir=args.eval_dir)
    else:
        args.task_name = "{}_lmax{}_bw{}_st{}_ed{}".format(args.dataset, args.lmax, args.bw, args.start_position, args.end_position)
        log_name = "Shrec17_{}.log".format(args.task_name)
        log_dir = os.path.join(CUR_DIR, "logging")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, log_name), level=logging.INFO)

        if args.slurm:
            import time
            t_sleep = np.random.randint(100)
            print("Sleep {} seconds".format(t_sleep))
            time.sleep(t_sleep)

        random.seed(args.start_position)

        cache_train_data(args, args.dataset == "test")