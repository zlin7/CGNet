import numpy as np
from importlib import reload
import utils.geometries as geometries; reload(geometries)
import utils.WignerD as WignerD
from lie_learn.representations.SO3 import wigner_d as wd
from lie_learn.representations.SO3 import spherical_harmonics as sh
from lie_learn.spaces import S2
import ipdb
import os
from functools import lru_cache

DEFAULT_GRID = 'Gauss-Legendre'

#@lru_cache(maxsize=100)
#def get_quad_weights(b, grid_type=DEFAULT_GRID):
#    ws = S2.quadrature_weights(b=b, grid_type=grid_type)
#    return ws.flatten()

@lru_cache(maxsize=50)
def get_grid_and_weights(b, grid_type=DEFAULT_GRID):
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    ws = S2.quadrature_weights(b=b, grid_type=grid_type)
    theta, phi = theta.flatten(), phi.flatten() - np.pi
    ws = ws.flatten()
    return theta, phi, ws

#def get_rotated_grid_and_weights(b, angle=None, grid_type=DEFAULT_GRID):
#    theta, phi, ws = get_grid_and_weights(b, grid_type)
#    if angle is None: return theta, phi, ws

@lru_cache(maxsize=100)
def get_default_sphs(lmax, complexFlag=True, grid_type=DEFAULT_GRID):
    theta, phi, _ = get_grid_and_weights(lmax, grid_type)
    sph_func = sh.csh if complexFlag else sh.rsh
    sph = np.zeros((len(theta), lmax ** 2), dtype=complex)
    for l in range(lmax):
        for m in range(-l, l + 1):
            sph[:, l ** 2 + (m + l)] = sph_func(l, m, theta, phi, 'quantum', True)
    return sph

def get_sphs(lmax, complexFlag=True, grid_type=DEFAULT_GRID, angles=None):
    if angles is None: return get_default_sphs(lmax, complexFlag, grid_type)
    theta, phi, _ = get_grid_and_weights(lmax, grid_type)
    theta, phi = geometries.rotate_coords(theta, phi, angles, direction=1, zyz=True)
    sph_func = sh.csh if complexFlag else sh.rsh
    sph = np.zeros((len(theta), lmax ** 2), dtype=complex)
    for l in range(lmax):
        for m in range(-l, l + 1):
            sph[:, l ** 2 + (m + l)] = sph_func(l, m, theta, phi, 'quantum', True)
    return sph

def sample_grid_value(hs, coords, lmax, grid_type=DEFAULT_GRID):
    theta, phi, _ = get_grid_and_weights(lmax, grid_type)
    new_coords = np.stack([theta, phi], axis=1)
    new_hs = geometries.resampling(hs, coords, new_coords)
    return new_hs


#=================main functions
def get_sampled_fs_with_rotation(fs, coords, lmax, angle=None, rot_func=True, grid_type=DEFAULT_GRID):
    theta, phi = coords[:, 0], coords[:, 1]
    if angle is not None and rot_func:
        theta, phi = geometries.rotate_coords(theta, phi, angle, direction=1, zyz=True)
    coords = np.stack([theta, phi], 1)
    fs = sample_grid_value(fs, coords, lmax, grid_type)
    return fs, coords

def get_sampled_fs_from_func_with_rotation(func, lmax, angle=None, rot_func=True, grid_type=DEFAULT_GRID):
    #fft
    theta, phi, ws = get_grid_and_weights(lmax, grid_type)
    theta_eval, phi_eval = theta, phi
    if angle is not None and rot_func:
        theta_eval, phi_eval = geometries.rotate_coords(theta, phi, angle, direction=-1, zyz=True)
    fs = func(theta_eval, phi_eval)
    coords = np.stack([theta, phi], 1)
    return fs, coords


def get_sph_coefs(fs, coords, lmax, angle=None, rot_func=True, grid_type=DEFAULT_GRID):
    #fft
    fs, coords = get_sampled_fs_with_rotation(fs, coords, lmax, angle, rot_func, grid_type)
    _, _, ws = get_grid_and_weights(lmax, grid_type)
    sphs = get_sphs(lmax, grid_type=grid_type, angles=None if rot_func else angle)
    coefs = (np.expand_dims(fs * ws, 1) * np.conj(sphs)).mean(0) #conjugate
    coefs *= 4 * np.pi
    return coefs

def get_sph_coefs_from_func(func, lmax, angle=None, rot_func=True, grid_type=DEFAULT_GRID):
    #fft
    fs, coords = get_sampled_fs_from_func_with_rotation(func, lmax, angle, rot_func, grid_type)
    _, _, ws = get_grid_and_weights(lmax, grid_type)
    sphs = get_sphs(lmax, grid_type=grid_type, angles=None if rot_func else angle)
    coefs = (np.expand_dims(fs * ws, 1) * np.conj(sphs)).mean(0)
    coefs *= 4 * np.pi
    return coefs

def reconstruct(coefs, grid_type=DEFAULT_GRID):
    #ifft
    lmax = int(len(coefs) ** 0.5)
    theta, phi, _ = get_grid_and_weights(lmax, grid_type=grid_type)
    sphs = get_default_sphs(lmax, grid_type=grid_type)
    res = (sphs * np.expand_dims(coefs, 0)).sum(1)
    return res, theta, phi


def rotate_coefs(coefs, angles, complexFlag=True):
    if angles is None: return coefs
    lmax = int(np.sqrt(len(coefs)))
    new_coef = np.empty(coefs.shape, dtype=np.complex)
    for l in range(lmax):
        old_coef = coefs[l**2:(l+1)**2]
        if complexFlag:
            Dl = WignerD.Dm(angles, l)
        else:
            Dl = wd.wigner_D_matrix(l, angles[0], angles[1], angles[2])
        new_coef[l**2:(l+1)**2] = Dl @ old_coef
    return new_coef