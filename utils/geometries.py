import numpy as np
import lie_learn
from lie_learn.representations.SO3 import spherical_harmonics as sh
from lie_learn.spaces import S2, spherical_quadrature as sq
from lie_learn.representations.SO3 import wigner_d as wd
import ipdb
from functools import lru_cache

#good for complex
def change_coordinates_R2(coords, p_from = "C"):
    #p_from can be "C" (Cartesian) or anything else (polar coordinates)
    #coords should be an ndarray of two columns (x,y) or (r, ro)
    #routine used to transform 2d picture to coords that represent points on the sphere
    #coords[:,0] is beta and coords[:,1] is alpha
    new_coords = coords.copy()
    if p_from == "C":
        new_coords[:, 0] = np.sum(coords ** 2, 1)
        new_coords[:, 1] = np.arctan2(coords[:, 1], coords[:, 0])
        new_coords[:, 0] = np.pi * new_coords[:, 0]
        #ipdb.set_trace()
    if p_from == 'P':
        r = np.sqrt(coords[:,0]/np.pi)
        new_coords[:, 1] = np.sin(coords[:, 1]) * r
        new_coords[:, 0] = np.cos(coords[:, 1]) * r
    return new_coords

def resampling(activations, old_coords, new_coords, d=0.1):
    import scipy.interpolate as interpolate
    #coords should be polar
    #extend the alpha's [-pi, pi] range to [-pi - d, pi+d]
    idx = (np.abs(old_coords[:, 1]) > np.pi - d)
    more_activations = np.concatenate([activations, activations[idx]])
    more_coords = np.concatenate([old_coords, old_coords[idx]])
    new_activations = interpolate.griddata(more_coords, more_activations, new_coords, fill_value=0.)
    return new_activations

def get_coef_grid2(b=3, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    ipdb.set_trace()
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_

#good for complex
def get_coef_grid(f_grid, a = 1.0):
    #The smaller a is, the smaller the picture is on the graph
    #Generate the coordinates and activation values according to Risi's paper
    n = f_grid.shape[0]
    x, y = np.where(f_grid == f_grid)
    new_f = f_grid[x, y]
    x = (x.astype(float) - (n - 1) / 2) / (n/2) * a
    y = (y.astype(float) - (n - 1) / 2) / (n/2) * a
    r = x**2 + y ** 2
    idx = r < 1
    coords = np.stack((x[idx], y[idx]), axis=-1)
    new_f = new_f[idx]
    new_coords = change_coordinates_R2(coords, p_from='C')
    return new_coords, new_f


def _rotation_matrix(angles, zyz=False):
    s1 = np.sin(angles[0])
    s2 = np.sin(angles[1])
    s3 = np.sin(angles[2])
    c1 = np.cos(angles[0])
    c2 = np.cos(angles[1])
    c3 = np.cos(angles[2])
    if zyz: #This is the convention used by s2cnn, Z1-Y2-Z3 convention
        #https://en.wikipedia.org/wiki/Euler_angles
        R_Z1 = np.asmatrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        R_Y2 = np.asmatrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        R_Z3 = np.asmatrix([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
        R = R_Z1 @ R_Y2 @ R_Z3
    else:
        Rz = np.asmatrix([[c1,-s1,0], [s1,c1,0], [0,0,1]])
        Ry = np.asmatrix([[c2, 0, -s2], [0,1,0], [s2,0, c2]])
        Rz2 = np.asmatrix([[c3,-s3,0], [s3,c3,0], [0,0,1]])
        R = (Rz2 * Ry * Rz).T
    return R


def test_angles():
    angles = np.asarray([1./3, 0.3, 0.6]) * np.pi
    R = _rotation_matrix(angles, zyz=True)
    angles_inv = (-angles[2], -angles[1], -angles[0])
    R_inv = _rotation_matrix(angles_inv, zyz=True)
    print(R)
    print(R_inv)
    print(R @ R_inv, R_inv @ R)

#good for complex
def rotate_coords(beta, alpha, angles, direction=1, zyz=False):
    #Rotate the spherical coordinates by angles (a tuple of three Euler angles)
    if direction == -1:
        angles = (-angles[2], -angles[1], -angles[0])
    scoords = np.stack((beta, alpha), -1)
    ccoords = S2.change_coordinates(scoords, "S", "C")

    R = _rotation_matrix(angles, zyz=zyz)
    ccoords_rotated = np.asarray((R * ccoords.T).T)

    ccoords_rotated = np.clip(ccoords_rotated, -1, 1) #some numerical error might cause issues

    scoords_rotated = S2.change_coordinates(ccoords_rotated, 'C', 'S')
    return scoords_rotated[:,0], scoords_rotated[:,1]

def chop_coefs_func(coefs, dim=1):
    if dim not in {0, 1}: raise NotImplementedError()
    n = coefs.shape[dim]
    lmax = int(n ** 0.5)
    assert np.power(lmax, 2) == n
    if dim == 0:
        coefs = [coefs[l**2:(l+1)**2] for l in range(lmax)]
    if dim == 1:
        coefs = [coefs[:, l**2:(l+1)**2] for l in range(lmax)]
    return coefs

#def get_coef_C_new(f, )

#good for complex (caveat: only taking inner product)
def get_coef_C(f, beta, alpha, 
                lmax=14, 
                chop_coeffs=False, 
                complexFlag=True,
                sph=None):
    #Each row of f, beta, alpha form a point on the sphere, where f is the activation value
    #Compute the coefficients, spherical harmonics values, and return them and f,beta,alpha back
    # Here, beta is the the colatitude / polar angle, and alpha is the longitude / azimuthal angle, ranging from 0 to 2 pi.
    if sph is None:
        #If sph is given, reuse it. This might help improve performance
        sph = np.zeros((len(f),(lmax+1)**2), dtype=complex)
        for l in range(lmax + 1):
            for m in range(-l, l+1):
                if complexFlag:
                    sph[:, l ** 2 + (m + l)] = sh.csh(l, m, beta, alpha, 'quantum', True) #/ (2 * l + 1)
                else:
                    sph[:, l ** 2 + (m + l)] = sh.rsh(l, m, beta, alpha, 'quantum', True) #/ (2 * l + 1)
    coefs = (np.expand_dims(f, 1) * sph).mean(0) * 4 * np.pi #normalize
    if chop_coeffs: coefs = chop_coefs_func(coefs)
    return coefs, sph


def reconstruct(f, beta=None, alpha=None, complexFlag=False):
    if beta is None or alpha is None:
        coords = get_coef_grid(np.zeros((28,28)))[0]
        beta = coords[:,0]
        alpha = coords[:,1]
    vs = np.zeros(beta.shape, dtype=complex if complexFlag else float)
    sh_func = sh.csh if complexFlag else sh.rsh
    if isinstance(f, list):
        lmax = len(f) - 1
        for l in range(lmax+1):
            for m in range(-l,l+1):
                vs[:] += f[l][m + l] * sh_func(l, m, beta, alpha, "quantum", True)
    else:
        lmax = int(np.sqrt(f.shape[0])) - 1
        for l in range(lmax+1):
            for m in range(-l,l+1):
                vs[:] += f[l ** 2 + m + l] * sh_func(l, m, beta, alpha, "quantum", True)
    return vs


#NOT good for complex
def plot_sphere_func(f, beta=None, alpha=None, normalize=True, title=''):
    if beta is None or alpha is None:
        coords = get_coef_grid(np.zeros((28,28)))[0]
        beta = coords[:,0]
        alpha = coords[:,1]
    # TODO: update this  function now that we have changed the order of axes in f
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.special import sph_harm

    if normalize:
        #f = (f - np.min(f)) / (np.max(f) - np.min(f))
        print("Normalizing pixels")
        f = (f - np.mean(f)) / np.std(f)

    x = np.sin(beta) * np.cos(alpha)
    y = np.sin(beta) * np.sin(alpha)
    z = np.cos(beta)
    if f.ndim == 2:
        f = cm.gray(f)
        print('2')

    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=f ) # cm.gray(f))
    #ax.plot_trisurf(x,y,z,color=f)
    ax.scatter(xs=x,ys=y,zs=z,c=f)

    # Turn off the axis planes
    ax.set_axis_off()
    ax.set_title(title)
    plt.show()

def rotate_Y(l, Y, alpha, beta, gamma):
    #Rotating the Y (spherical harmonic values) with three Euler angles
    #each row of Y is the spherical harmonic values at one point (or different bases)
    Y2 = Y.copy()
    D = wd.wigner_D_matrix(l, alpha,beta,gamma)
    for pix in range(Y.shape[0]):
        for m in range(-l, l+1, 1):    
            Y2[pix,m] = np.sum(D[m,:] * Y[pix,:])
    return Y2

if __name__ == '__main__':
    pass