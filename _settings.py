import os
import sys
DATA_PATH = "Z:/Data" #Local

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__CUR_FILE_PATH)))

MNIST_NAME = 'MNIST'
MNIST_PATH = os.path.join(DATA_PATH, MNIST_NAME)

#==============================Data Related
WORKSPACE = os.path.join(__CUR_FILE_PATH, "Temp")
_PERSIST_PATH = os.path.join(WORKSPACE, 'cache')
LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7

NCOLS = 80

import torch
import numpy as np
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
