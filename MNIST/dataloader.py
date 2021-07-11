import numpy as np
import _settings as _settings
import torch
TRAIN = 'train'
VALID = 'val'
TEST = 'test'


from torch.utils.data import Dataset

class MNISTData(Dataset):
    DATASET = _settings.MNIST_NAME
    LABEL_MAP = {str(i): i for i in range(10)}
    @classmethod
    def split_data(cls, seed=_settings.RANDOM_SEED, split_ratio=[85, 15], n=60000):
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(np.arange(n), test_size = split_ratio[1] / float(sum(split_ratio)), random_state=seed)
        return {TRAIN: sorted(train), VALID: sorted(val)}

    def __init__(self, mode=TRAIN,
                 lmax=10, rotate=False, dilate=False, a=1.0,
                 to_tensor=True, seed=_settings.RANDOM_SEED,
                 raw=False):
        super(MNISTData, self).__init__()
        self.mode = mode
        self._seed = seed
        import MNIST.MNIST_data_process

        split = TEST if mode == TEST else TRAIN
        kwargs = {'a': 0.2, 'amax': 1.2} if dilate else {'a': a}
        self.coefs = MNIST.MNIST_data_process.precomputing_coefs_new(lmax=lmax, rotate=rotate, split=split, **kwargs)
        self.labels = MNIST.MNIST_data_process.get_labels(split=split)
        if mode == TEST:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = sorted(self.split_data(seed=self._seed)[self.mode])
        self.n = len(self.indices)
        self.to_tensor = to_tensor
        self.lmax = lmax
        self.raw = raw

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        #_preprocessing is done here
        if torch.is_tensor(idx): idx = idx.tolist()
        if idx >= self.n: raise IndexError("%d is out of range (%d elements)"%(idx, self.n))
        x = self.coefs[idx]
        y = self.labels[idx]
        #if self.to_tensor and y is not None: y = torch.tensor(y, dtype=torch.long)
        #return torch.tensor(x, dtype=torch.float), y, idx

        if self.raw: return x, y, idx
        #==================
        x = np.stack([x.real, x.imag], 1)
        x = np.expand_dims(x, 0) #Add channel layer

        #This step make the middle dimension l-tau-m if necessary
        x = np.concatenate([x[:, l**2:(l+1)**2,:].reshape(-1, 2) for l in range(self.lmax)]).astype(np.float32)
        return x, y, idx
