import datetime
import _settings

LOG_FOLDER = _settings.LOG_OUTPUT_DIR
TODAY_STR = datetime.datetime.today().strftime('%Y%m%d')

def set_all_seeds(random_seed=_settings.RANDOM_SEED):
    import torch
    import numpy as np
    # torch.set_deterministic(True)#This is only available in 1.7
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print("Setting seeds to %d" % random_seed)
    #os.environ["DEBUSSY"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
