import numpy as np
import lie_learn
from lie_learn.representations.SO3 import spherical_harmonics as sh
from lie_learn.spaces import S2, spherical_quadrature as sq
from lie_learn.representations.SO3 import wigner_d as wd

#good for complex
def change_coordinates_R2(coords, p_from = "C"):
    #p_from can be "C" (Cartesian) or anything else (polar coordinates)
    #coords should be an ndarray of two columns (x,y) or (r, ro)
    #routine used to transform 2d picture to coords that represent points on the sphere
    #coords[:,0] is beta and coords[:,1] is alpha
    if p_from == "C":
        cartesian_coords = coords.copy()
        coords[:, 0] = np.sum(cartesian_coords**2,1)
        coords[:, 1] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:,0])
    coords[:, 0] = np.pi * coords[:, 0]
    return coords


#good for complex
def get_coef_grid(f_grid, a = 1.0):
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
    return change_coordinates_R2(coords), new_f

def _rotation_matrix(angles, wikipedia=False):
    s1 = np.sin(angles[0])
    s2 = np.sin(angles[1])
    s3 = np.sin(angles[2])
    c1 = np.cos(angles[0])
    c2 = np.cos(angles[1])
    c3 = np.cos(angles[2])
    if wikipedia:
        R = np.asmatrix([[c1*c2*c3 - s1*s3, -c3*s1 - c1*c2*s3,  c1*s2],
                         [c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3,   s1*s2],
                         [-c3*s2,           s2*s3,              c2]])
    else:
        Rz = np.asmatrix([[c1,-s1,0], [s1,c1,0], [0,0,1]])
        Ry = np.asmatrix([[c2, 0, -s2], [0,1,0], [s2,0, c2]])
        Rz2 = np.asmatrix([[c3,-s3,0], [s3,c3,0], [0,0,1]])
        R = (Rz2 * Ry * Rz).T
    return R


#good for complex
def rotate_coords(beta, alpha, angles, direction=1):
    #Rotate the spherical coordinates by angles (a tuple of three Euler angles)
    if direction == -1:
        angles = (-angles[0], -angles[1], -angles[2])
    scoords = np.stack((beta, alpha), -1)
    ccoords = S2.change_coordinates(scoords, "S", "C")

    R = _rotation_matrix(angles, wikipedia=False)
    ccoords_rotated = np.asarray((R * ccoords.T).T)

    scoords_rotated = S2.change_coordinates(ccoords_rotated, 'C', 'S')
    return scoords_rotated[:,0], scoords_rotated[:,1]



#good for complex (caveat: only taking inner product)
def get_coef_C(f, beta, alpha, 
                lmax=14, 
                chop_coeffs=False, 
                complexFlag=True,
                sph=None):
    #Each row of f, beta, alpha form a point on the sphere, where f is the activation value
    #Compute the coefficients, spherical harmonics values, and return them and f,beta,alpha back

    if sph is None:
        #If sph is given, reuse it. This might help improve performance
        sph = np.zeros((len(f),(lmax+1)**2), dtype=complex)
        for l in range(lmax + 1):
            for m in range(-l, l+1):
                if complexFlag:
                    sph[:, l**2+(m+l)] = sh.csh(l,m,beta,alpha,'quantum',True)
                else:
                    sph[:, l**2+(m+l)] = sh.rsh(l,m,beta,alpha,'quantum',True)
    coefs = (np.expand_dims(f, 1)*sph).sum(0)
    
    if chop_coeffs:
        n = len(coefs)
        st = 0
        d = 1
        coefs_old = coefs
        coefs = []
        while(st < n):
            coefs.append(coefs_old[st:(st+d)].copy())
            st += d
            d += 2
        return coefs, sph
    else:
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
                vs[:] += f[l][m+l] * sh_func(l,m,beta, alpha,"quantum",False)
    else:
        lmax = int(np.sqrt(f.shape[0])) - 1
        for l in range(lmax+1):
            for m in range(-l,l+1):
                vs[:] +=  f[l**2+m+l] * sh_func(l,m,beta, alpha,"quantum",False)
    return vs

#NOT good for complex
def plot_sphere_func(f, beta=None, alpha=None, normalize=True):
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
    plt.show()

#NOT good for complex
"""Checking covariance"""
def rotate_Y(l, Y, alpha, beta, gamma):
    #Rotating the Y (spherical harmonic values) with three Euler angles
    #each row of Y is the spherical harmonic values at one point (or different bases)
    Y2 = Y.copy()
    D = wd.wigner_D_matrix(l, alpha,beta,gamma)
    for pix in range(Y.shape[0]):
        for m in range(-l, l+1, 1):    
            Y2[pix,m] = np.sum(D[m,:] * Y[pix,:])
    return Y2
