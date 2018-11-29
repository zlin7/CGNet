import numpy as np
import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../CGNet"))
import geometries
import gzip
import logging
import sys
import random
import datetime

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
REAL_PART=0
IMAG_PART=1
 
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def extract_data(filename, num_images):
    print('Extracting data', filename)
    open_func = gzip.open if filename[-3:-1] == ".gz" else (lambda x:open(x,"rb"))
    #with gzip.open(filename) as bytestream:
    #with open(filename, 'rb') as bytestream:
    with open_func(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        return data / (PIXEL_DEPTH + 0.1)

def extract_labels(filename, num_images):
    print('Extracting label', filename)
    with open(filename, 'rb') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

def _get_input_from2D(img2D, idx=None, lmax=14, rotate_angles=None, 
                        complexFlag=True,
                        separateParts=False,
                        sphs=None):
    coords, hs = geometries.get_coef_grid(img2D)
    if rotate_angles is not None:
        beta, alpha = geometries.rotate_coords(coords[:,0], coords[:,1], rotate_angles)
    else:
        beta = coords[:,0]
        alpha = coords[:,1]
    coefs, sphs = geometries.get_coef_C(hs, beta, alpha, lmax=lmax, chop_coeffs=False, complexFlag=complexFlag, sph=sphs)
    if separateParts:
        reals = coefs.real
        imags = coefs.imag
        f_0 = (reals, imags)
        return f_0, sphs, hs, beta, alpha
    else:
        return coefs, sphs, hs, beta, alpha

def get_inputs_from2D(lmax, img2Ds, random_rotate=False):
    logger = logging.getLogger("_datautils_logger")
    if random_rotate:
        logger.info("random rotate..")
    d = len(img2Ds.shape)
    if (d > 2):
        sphs = None
        batch_size = len(img2Ds)
        fs = []
        for i in range(batch_size):
            angles = 2*np.pi*np.random.uniform(size=3) if random_rotate else None
            if i % 10 == 0 and logger is not None:
                #print(i)
                logger.info("working on {}".format(i))
            coefs, sphs,_,_,_ = _get_input_from2D(img2Ds[i], idx=i, lmax=lmax, rotate_angles=angles, sphs=sphs if not random_rotate else None)
            fs.append(coefs)
        f_0 = np.stack(fs, 0)
    else:
        angles = 2*np.pi*np.random.uniform(size=3) if random_rotate else None
        #angles = tuple([random.random()*2.*np.pi for i in range(3)]) if random_rotate else None
        batch_size = 0
        f_0, _, _, _, _ = _get_input_from2D(0,img2Ds, lmax=self.lmax, rotate_angles=angles)
    return f_0


def np_save_safe(file_path, data):
    folder_path = os.path.dirname(file_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    print("Saving to {}".format(file_path))
    np.save(file_path, data)


def precomputing_coefs(data_folder, lmax, 
                        NUM_IMAGES_TO_USE=1000, 
                        data_file_name="train-images.idx3-ubyte",
                        st=None,
                        ed=None,
                        coeftype="train"):
    rotate = "rotate" in coeftype
    coefs_folder = os.path.join(CUR_DIR, "precomputed_coefs/")
    print("in datautils.precomputing_coefs: doing %s"%coeftype)
    if st is None or ed is None:
        file_path, num_got = _find_old_file(coefs_folder,lmax,NUM_IMAGES_TO_USE,coeftype=coeftype)
        if file_path is not None:
            data = np.load(file_path)[0:NUM_IMAGES_TO_USE,0:((lmax+1)**2)]
        if (num_got < NUM_IMAGES_TO_USE):
            print("WARNING - did not read enough data! computing the rest...")
            file_path = os.path.join(os.path.join(coefs_folder,coeftype),"L_{}_N_{}.npy".format(lmax, NUM_IMAGES_TO_USE))
            all_data = extract_data(os.path.join(data_folder, data_file_name), NUM_IMAGES_TO_USE)
            newly_read = get_inputs_from2D(lmax, all_data[num_got:NUM_IMAGES_TO_USE], random_rotate=rotate)
            data = newly_read if num_got == 0 else np.concatenate([data, newly_read],0)
            np_save_safe(file_path, data)
    else:
        file_path = os.path.join(coefs_folder, "separate/{}/L_{}_st_{}_ed_{}.npy".format(coeftype, lmax, st, ed))
        all_data = extract_data(os.path.join(data_folder, data_file_name), ed)[st:ed]

        data = get_inputs_from2D(lmax, all_data, random_rotate=rotate)
        np_save_safe(file_path, data)
    return data

#def split_data(data_real, data_imag, ratios=[0.9,0.05,0.05]):
def split_data(data_to_split, ratios=[0.9,0.05,0.05]):
    s = float(ratios[0] + ratios[1] + ratios[2])
    r0 = ratios[0]/s
    r1 = r0 + ratios[1] / s
    n = len(data_to_split)
    data_train = data_to_split[0:int(r0*n)]
    data_valid = data_to_split[int(r0*n):int(r1*n)]
    data_test  = data_to_split[int(r1*n):n]
    return data_train, data_valid, data_test

def _find_old_file(coefs_folder, lmax, NUM_IMAGES_TO_USE, coeftype="train"):
    data_folder = os.path.join(coefs_folder, coeftype)
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    files = [x for x in os.listdir(data_folder) if x.endswith('.npy')]
    d = {}
    max_N = 0
    for f in files:
        _, L, _, N = f.split('.')[0].split("_")
        L = int(L)
        N = int(N)
        if L >= lmax:
            if N >= NUM_IMAGES_TO_USE:
                return os.path.join(data_folder, f), N
            elif N > max_N:
                max_N = N
                d[N] = os.path.join(data_folder, f)
    return d.get(max_N), max_N

#TODO: fix unzipping issue (now need to manually unzip)
def download_MNIST_and_precompute_all(n=60000, work_directory=os.path.join(CUR_DIR, "temp"), st=None, ed=None, istrain=True, rotate=False):
    import gzip
    import os
    import numpy
    import urllib
    if not os.path.isdir(work_directory):
        os.makedirs(work_directory)
    log_name = "precompute.log" if st is None or ed is None else "pre_{}_{}_{}.log".format(st,ed, "train" if istrain else "test")
    logging.basicConfig(filename=os.path.join(os.path.join(CUR_DIR,"temp"),log_name), level=logging.INFO)
    filenames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    def _download(filename, folder,SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'):
        file_path = os.path.join(folder, filename)
        new_filename = filename.replace(".gz", "").replace("-idx", ".idx")
        if not os.path.exists(file_path):
            file_path, _ = urllib.urlretrieve(SOURCE_URL + filename, file_path)
        if not os.path.exists(os.path.join(folder, new_filename)):
            import gzip
            with gzip.open(file_path, 'rb') as infile:
                with open(os.path.join(folder, new_filename), 'wb') as outfile:
                    for line in infile:
                        outfile.write(line)
        return new_filename
    for i in range(len(filenames)):
        filenames[i] = _download(filenames[i], work_directory)
        
    #filenames = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"]

    coeftype = ("train" if istrain else "test") + ("_rotate" if rotate else "")
    if st is None or ed is None:
        precomputing_coefs(work_directory, 12, n, data_file_name=filenames[0 if istrain else 2], coeftype=coeftype)
    else:
        precomputing_coefs(work_directory, 12, st=st, ed=ed, data_file_name=filenames[0 if istrain else 2], coeftype=coeftype)
    return None

if __name__ == "__main__":
    np.random.seed(1)
    print(CUR_DIR)
    if len(sys.argv) == 2:
        if sys.argv[1] == "all":
            download_MNIST_and_precompute_all(n=60000, istrain=True, rotate=False)
            download_MNIST_and_precompute_all(n=10000, istrain=False, rotate=False)
            download_MNIST_and_precompute_all(n=60000, istrain=True, rotate=True)
            download_MNIST_and_precompute_all(n=10000, istrain=False, rotate=True)
    elif len(sys.argv) == 4:
        st = int(sys.argv[1])
        ed = int(sys.argv[2])
        #0: train unrotate
        #1: test unrotate
        #2: train rotate
        #3: test rotate
        istrain = int(sys.argv[3])%2 == 0
        rotate = int(sys.argv[3])//2 == 1
        starttime = datetime.datetime.now()
        download_MNIST_and_precompute_all(st=st,ed=ed, istrain=istrain, rotate=rotate)
        print("start {}, now {}".format(starttime,datetime.datetime.now()))