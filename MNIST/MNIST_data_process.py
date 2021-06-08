from importlib import reload
import numpy as np
import os
import sys
import utils.geometries as geometries; reload(geometries)
import gzip
import logging
import _settings
import utils.s2_sph as s2_sph
import tqdm, requests

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
REAL_PART = 0
IMAG_PART = 1

_CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def download_file(url, dst, overwrite=False):
    # NOTE the stream=True parameter below
    if not overwrite: assert not os.path.isfile(dst), "{} already exists.".format(dst)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f:
            #for chunk in r.iter_content(chunk_size=8192):
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192), total=int(r.headers.get('content-length', 0)) / 8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return dst



def download(data_dir = None):
    if data_dir is None: data_dir = _settings.MNIST_PATH
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    filenames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz"]

    def _download_help(filename, folder, SOURCE_URL='http://yann.lecun.com/exdb/mnist/'):
        import gzip
        file_path = os.path.join(folder, filename)
        new_filename = filename.replace(".gz", "").replace("-idx", ".idx")
        if not os.path.isfile(file_path): download_file(SOURCE_URL + filename, file_path)
        if not os.path.isfile(os.path.join(folder, new_filename)):
            with gzip.open(file_path, 'rb') as infile:
                with open(os.path.join(folder, new_filename), 'wb') as outfile:
                    for line in infile:
                        outfile.write(line)
        return new_filename

    for i, filename in enumerate(filenames):
        filenames[i] = _download_help(filename, data_dir)
    return data_dir, filenames


def extract_data(filename, num_images):
    print('Extracting data', filename)
    open_func = gzip.open if filename[-3:-1] == ".gz" else (lambda x: open(x, "rb"))
    # with gzip.open(filename) as bytestream:
    # with open(filename, 'rb') as bytestream:
    with open_func(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        return data / (PIXEL_DEPTH + 0.1)


def extract_labels(filename, num_images):
    print('Extracting label', filename)
    with open(filename, 'rb') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def _get_input_from2D(img2D, lmax=14, rotate_angles=None, separateParts=False, rot_func=False, a=1.0):
    coords, hs = geometries.get_coef_grid(img2D, a=a)
    coefs = s2_sph.get_sph_coefs(hs, coords, lmax, angle=rotate_angles, rot_func=rot_func)
    if separateParts:
        reals = coefs.real
        imags = coefs.imag
        f_0 = (reals, imags)
        return f_0, hs
    else:
        return coefs, hs

def np_save_safe(file_path, data):
    folder_path = os.path.dirname(file_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    print("Saving to {}".format(file_path))
    np.save(file_path, data)

def get_inputs_from2D(lmax, img2Ds, random_rotate=False, rot_func=False, a=1.0, amax=None):
    logger = logging.getLogger("_datautils_logger")
    if random_rotate or amax is not None:
        np.random.seed(len(img2Ds))
        logger.info("random rotate..")
    assert len(img2Ds.shape) == 3, "(index, x, y)"
    batch_size = len(img2Ds)
    fs = []
    if amax is not None: amin = a
    for i in tqdm.tqdm(range(batch_size), ncols=60):
        angles = 2 * np.pi * np.random.uniform(size=3) if random_rotate else None
        if i % 100 == 0 and logger is not None:
            logger.info("working on {}".format(i))
        if amax is not None: a = np.random.uniform(amin, amax)
        coefs, _ = _get_input_from2D(img2Ds[i], lmax=lmax, rotate_angles=angles, rot_func=rot_func, a=a)
        fs.append(coefs)
    f_0 = np.stack(fs, 0)
    return f_0

def __find_old_coefs(lmax, fdir):
    fs = os.listdir(fdir)
    if len(fs) == 0: return None
    old_lmaxs = [int(f.split("_")[1].split(".")[0]) for f in fs]
    if max(old_lmaxs) >= lmax:
        return os.path.join(fdir, 'L_%d.npy'%max(old_lmaxs))
    return None

def precomputing_coefs_new(lmax, split='test', rotate=False, data_dir=None, recompute=False, rot_func=False, a=1.0, amax=None):
    data_dir, filenames = download(data_dir)
    coefs_folder = os.path.join(data_dir, 'coefs')
    coeftype = f"{split}{'_rotate' if rotate else ''}"
    if rotate and rot_func: coeftype += "_rotfunc"
    if a != 1.0:
        a = np.round(a, 2)
        coeftype += f"_a{a:.2f}"
        if amax is not None:
            coeftype += f"-{amax:.2f}"
    file_path = os.path.join(coefs_folder, coeftype, f"L_{lmax}.npy")
    if not recompute:
        if not os.path.isdir(os.path.dirname(file_path)): os.makedirs(os.path.dirname(file_path))
        old_fpath = __find_old_coefs(lmax, os.path.dirname(file_path))
        if old_fpath is not None:
            print('Loading lmax=%d from %s'%(lmax, old_fpath))
            return np.load(old_fpath)[:,:lmax**2]
    data_file_name = os.path.join(data_dir, filenames[{'train': 0, 'test': 2}[split]])
    data = extract_data(data_file_name, 60000 if split == 'train' else 10000)
    coefs = get_inputs_from2D(lmax, data, random_rotate=rotate, rot_func=rot_func, a=a, amax=amax)
    np_save_safe(file_path, coefs)
    return coefs, data

def get_labels(split='test', data_dir=None):
    data_dir, filenames = download(data_dir)
    data_file_name = os.path.join(data_dir, filenames[{'train': 1, 'test': 3}[split]])
    cache_file_path = os.path.join(data_dir, "label_%s.npy"%split)
    if os.path.isfile(cache_file_path): return np.load(cache_file_path)
    labels = extract_labels(data_file_name, 60000 if split == 'train' else 10000)
    np_save_safe(cache_file_path, labels)
    return labels

if __name__ == "__main__":
    for lmax in [12]: #scale
        for split in ['train', 'test']:
            precomputing_coefs_new(lmax, split, rotate=False, a=1.0)
            precomputing_coefs_new(lmax, split, rotate=True, a=1.0)