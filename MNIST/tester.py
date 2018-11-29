import numpy as np
import datautils
reload(datautils)
#import SphericalCNN_complex as Sc
import SphericalCNN_Fast as Sc
reload(Sc)
import geometries
reload(geometries)
from lie_learn.representations.SO3 import wigner_d as wd
from lie_learn.representations.SO3 import spherical_harmonics as sh
import WignerD
import scipy
MNIST_DATA_PATH = "../../hw2/MNIST/"
LMAX=3
NUM_IMAGES_TO_USE=1000


"""Below are testing..."""


def _checker(f1_all, f2_all, lmax, angles, complexFlag=True):
    #f1 is not rotated, f2 is rotated by angles
    for l in range(lmax+1):
        f1 = f1_all[l].squeeze(0)
        f2 = f2_all[l].squeeze(0)
        #print(f1.shape, f2.shape)
        if complexFlag:
            import WignerD
            Dl = WignerD.Dm(angles, l)
        else:
            Dl = wd.wigner_D_matrix(l, angles[0], angles[1], angles[2])

        for m in range(-l,l+1):
            for channel in range(f2.shape[1]):
                f2_ = np.sum(Dl[m+l,:]*f1[:,channel])
                if (np.absolute(f2[m+l,channel] - f2_) > 1e-4 * max(1,min(abs(f2_), abs(f2[m+l,channel] )))):
                    print("\ndifference at l={0:.0f}, m={1:.0f}, values {2:.3f} and {3:.3f}, (unrotated f1={4:.3f})".format( l, m, f2[m+l,channel], f2_, f1[m+l,channel]))
                else:
                    print("\nGood: {0:.3f} vs {1:.3f} ".format(f2[m+l,channel], f2_))
    return None

def test_covariance(complexFlag=True, raw_coefs = False, skip=0, norm=0):
    #Cannot test batch input yet
    global MNIST_DATA_PATH, LMAX, NUM_IMAGES_TO_USE
    data_folder = MNIST_DATA_PATH
    train_data = datautils.extract_data(data_folder + "train-images.idx3-ubyte", 1)
    img2D = train_data[0]

    angles = (np.pi/6*1, np.pi/9*1, np.pi/7*1)
    if skip == 2:
        s = Sc.SphericalResCNN_new(LMAX,2,2,3)
    else:
        s = Sc.SphericalCNN_new(LMAX, 2, 1, skip_type=skip, normalization=norm)
        if skip != 3 or norm == 1:
            s.eval()

    #print(img2D)
    f1 = datautils._get_input_from2D(img2D, lmax=LMAX, rotate_angles=None, complexFlag=complexFlag)[0]
    print("unrotated:")
    f1 = s.forward_test_angle(f1, 1 if raw_coefs else 2)
    print("rotated:")
    f2 = datautils._get_input_from2D(img2D, lmax=LMAX, rotate_angles=angles, complexFlag=complexFlag)[0]
    #print(f2)
    f2 = s.forward_test_angle(f2, 1 if raw_coefs else 2)

    _checker(f1, f2, s.lmax, angles, complexFlag)
    return None


def test_invariance(complexFlag=True, skip=0, norm=0):
    #Cannot test batch input yet
    global MNIST_DATA_PATH, LMAX, NUM_IMAGES_TO_USE
    data_folder = MNIST_DATA_PATH
    train_data = datautils.extract_data(data_folder + "train-images.idx3-ubyte", 1)
    img2D = train_data[0]

    angles = (np.pi/6*1, np.pi/9*1, np.pi/7*3)
    if skip == 2:
        s = Sc.SphericalResCNN_new(LMAX,2,2,3)
        #s.eval()
    else:
        s = Sc.SphericalCNN_new(LMAX, 2, 3, skip_type=skip, normalization=norm)
        if skip != 3 or norm == 1:
            s.eval()
    #print(img2D)
    f1 = datautils._get_input_from2D(img2D, lmax=LMAX, rotate_angles=None, complexFlag=complexFlag)[0]
    #f1 = np.stack([f1[0], f1[1]], 1)
    f1 = s.forward_test_angle(f1, 0)
    f2 = datautils._get_input_from2D(img2D, lmax=LMAX, rotate_angles=angles, complexFlag=complexFlag)[0]
    #f2 = np.stack([f2[0],  f2[1]], 1)
    f2 = s.forward_test_angle(f2, 0)
    print(f1, f2)
    #_checker(f1, f2, s.lmax, angles, complexFlag)
    return None

def test_precompute_coefs():
    global MNIST_DATA_PATH, LMAX, NUM_IMAGES_TO_USE
    data_train, label_train, data_valid, label_valid = datautils.precomputing_coefs(MNIST_DATA_PATH, LMAX-2, 500)
    print(data_train[0].shape,data_train[1].shape, data_valid[0].shape, data_valid[1].shape)

def testing_only_rotation(angles=(np.pi/2*0, np.pi/2*0, np.pi/2*0),a=1.0):
    global MNIST_DATA_PATH, LMAX
    img2D = datautils.extract_data(MNIST_DATA_PATH + "train-images.idx3-ubyte", 1)[0]
    #angles = (np.pi/6*0, np.pi/9*0, np.pi/6*0)

    coords, hs = geometries.get_coef_grid(img2D, a)
    #hs[0:len(hs)] = 1.0
    #hs[50:52] = 0.0
    beta, alpha = geometries.rotate_coords(coords[:,0], coords[:,1], angles)
    geometries.plot_sphere_func(hs, beta,alpha)
    return None


def test_WD_w_Y(rhat=[1,2,3],complexFlag=False):
    from lie_learn.spaces import S2
    global LMAX
    angles = (np.pi/3*1, np.pi/7*1, np.pi/13*1)
    lmax=LMAX

    NORMALIZATION="quantum"
    CSPhase=True
    R = geometries._rotation_matrix(angles, False)

    rhat = np.asmatrix(rhat)/np.sqrt(rhat[0]**2+rhat[1]**2+rhat[2]**2)
    #print(rhat)
    rhat_rot = np.asarray((R * np.asmatrix(rhat).T).T)
    #print(rhat, rhat_rot)
    rhat = S2.change_coordinates(rhat,"C","S").squeeze()
    rhat_rot = S2.change_coordinates(rhat_rot,"C","S").squeeze()
    #print(rhat, rhat_rot)
    sph = np.zeros((lmax+1)**2,dtype=complex)
    sph_rot = np.zeros((lmax+1)**2,dtype=complex)
    for l in range(lmax+1):
        for m in range(-l,l+1):
            sph[l**2+(m+l)] = sh.csh(l,m,rhat[0], rhat[1],NORMALIZATION,CSPhase)
            sph_rot[l**2+(m+l)] = sh.csh(l,m,rhat_rot[0],rhat_rot[1],NORMALIZATION,CSPhase)
    for l in range(lmax+1):
        Y = sph[(l**2):((l+1)**2)]
        Y_rot = sph_rot[(l**2):((l+1)**2)]
        Dl = WignerD.Dm(angles,l)
        #print Dl
        for m in range(-l,l+1):
            Y_rot_ = np.sum(Dl[m+l,:]*Y)
            if (np.absolute(Y_rot[m+l] - Y_rot_) > 1e-5):
                print("\ndifference at l={0:.0f}, m={1:.0f}, values {2:.3f} and {3:.3f}, (unrotated Y1={4:.3f})".format( l, m, Y_rot[m+l], Y_rot_, Y[m+l]))
            else:
                print("\nGood: {0:.3f} vs {1:.3f} ".format(Y_rot[m+l], Y_rot_))

def sample_img(idx=0, random_rotate=False,get_grid=False):

    global MNIST_DATA_PATH, LMAX
    img2D = datautils.extract_data(MNIST_DATA_PATH + "train-images.idx3-ubyte", 1)[0]
    #angles = (np.pi/6*0, np.pi/9*0, np.pi/6*0)

    coords, hs = geometries.get_coef_grid(img2D, 1.0)
    return hs if not get_grid else (hs, coords)

def get_sph(hs):
    global MNIST_DATA_PATH, LMAX
    img2D = datautils.extract_data(MNIST_DATA_PATH + "train-images.idx3-ubyte", 1)[0]
    #angles = (np.pi/6*0, np.pi/9*0, np.pi/6*0)
    coords, _ = geometries.get_coef_grid(img2D, 1.0)
    coefs, _= geometries.get_coef_C(hs, coords[:,0], coords[:,1], lmax=LMAX, chop_coeffs=False, complexFlag=True)
    return coefs

def reconstruct(f):
    global MNIST_DATA_PATH, LMAX
    img2D = datautils.extract_data(MNIST_DATA_PATH + "train-images.idx3-ubyte", 1)[0]
    #angles = (np.pi/6*0, np.pi/9*0, np.pi/6*0)
    grid, _ = geometries.get_coef_grid(img2D, 1.0)

    return geometries.reconstruct(f, grid[:,0], grid[:,1], complexFlag=True)


def plot_coef(f, img2D=False):
    global MNIST_DATA_PATH, LMAX
    img2D = datautils.extract_data(MNIST_DATA_PATH + "train-images.idx3-ubyte", 1)[0]
    #angles = (np.pi/6*0, np.pi/9*0, np.pi/6*0)
    grid, _ = geometries.get_coef_grid(img2D, 1.0)

    geometries.plot_sphere_func(f, grid[:,0], grid[:,1])


if __name__ == "__main__":
    #test_precompute_coefs()
    test_covariance(True, False, skip=1)
    #test_invariance(True, skip=1)
    #test_WD_w_Y(complexFlag=True)
    #testing_only_rotation()
    #test_covariance(True, False)
