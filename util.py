#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy.linalg import norm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.markers import MarkerStyle


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def rand_rotation_matrix(deflection=1.0, seed=None):
    """Creates a random rotation matrix.

    Args:
        deflection: the magnitude of the rotation. For 0, no rotation; for 1,
                    completely random rotation. Small deflection => small
                    perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """
    if seed is not None:
        np.random.seed(seed)

    theta, phi, z = np.random.uniform(size=(3,))

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r,
         np.cos(phi) * r,
         np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def add_rotation_to_pcloud(pcloud):
    r_rotation = rand_rotation_matrix()

    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch = batch.dot(r_rotation)
    return batch


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside
    the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def get_pc_plot( pc, elev=0, azim=0, roll=90, color='red'):

    fig = plt.figure(num=1, figsize=(15,15))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[...,0], pc[...,1], pc[..., 2], marker=MarkerStyle('o', fillstyle='full'), s=150, depthshade=0, c=color)
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim, roll=roll)
    set_axes_equal(ax)

    #fig.savefig(fname, dpi=500)
    return fig


def animated_PC_compare_multi(x_gt, y_gt, z_gt, x_rec_1, y_rec_1, z_rec_1, x_rec_2, y_rec_2, z_rec_2, title=''):

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title)

    def animate(angle):
        angle_norm = (angle + 180) % 360 - 180

        ax1.view_init(elev=angle_norm, azim=angle_norm, roll=angle_norm)
        ax2.view_init(elev=angle_norm, azim=angle_norm, roll=angle_norm)
        ax3.view_init(elev=angle_norm, azim=angle_norm, roll=angle_norm)
        return ax1, ax2, ax3


    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x_gt, y_gt, z_gt, marker='o', c='green', s=5)

    ax1.set_title(f'Ground Truth')
    ax1.set_axis_off()
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x_rec_1, y_rec_1, z_rec_1, marker=MarkerStyle('o', fillstyle='full'), c='red', s=5)
    ax2.set_title(f'Reconstructed Conv')
    ax2.set_axis_off()
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x_rec_2, y_rec_2, z_rec_2, marker=MarkerStyle('o', fillstyle='full'), c='blue', s=5)
    ax3.set_title('Reconstructed SH 15')
    ax3.set_axis_off()

    set_axes_equal(ax1)
    set_axes_equal(ax2)
    set_axes_equal(ax3)

    frames = np.linspace(0, 361, num=10)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=2500, blit=False)

    return anim



def animated_PC_compare(x_gt, y_gt, z_gt, x_rec, y_rec, z_rec, title=''):

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f'{title}')

    def animate(angle):
        angle_norm = (angle + 180) % 360 - 180

        ax1.view_init(elev=angle_norm, azim=angle_norm, roll=angle_norm)
        ax2.view_init(elev=angle_norm, azim=angle_norm, roll=angle_norm)

        return ax1, ax2


    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_gt, y_gt, z_gt, marker=',', c='green', s=1)

    ax1.set_title(f'Ground Truth')
    ax1.set_axis_off()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_rec, y_rec, z_rec, marker=',', c='red', s=1)
    ax2.set_title(f'Reconstructed')
    ax2.set_axis_off()

    set_axes_equal(ax1)
    set_axes_equal(ax2)

    frames = np.linspace(0, 361, num=10)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1200, blit=False)

    return anim



def plot_PC_compare(x_gt, y_gt, z_gt, x_rec, y_rec, z_rec, epoch=0):

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_gt, y_gt, z_gt, marker=',', c='green', s=0.2)
    ax1.view_init(elev=90, azim=0)

    ax1.set_title(f'Ground Truth')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_rec, y_rec, z_rec, marker=',', c='red', s=0.2)
    ax2.view_init(elev=90, azim=0)

    ax2.set_title(f'Reconstructed')
    set_axes_equal(ax1)
    set_axes_equal(ax2)

    plt.title(f'Epoch {epoch}')

    return fig

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
    plt.switch_backend('agg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def transform_point_clouds(X, only_z_rotation=False, deflection=1.0):
    r_rotation = rand_rotation_matrix(deflection)
    if only_z_rotation:
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
    X = X.dot(r_rotation).astype(np.float32)
    return X
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
