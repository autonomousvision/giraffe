import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_middle_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    u_m, u_v, r_v = sum(range_u) * 0.5, sum(range_v) * \
        0.5, sum(range_radius) * 0.5
    loc = sample_on_sphere((u_m, u_m), (u_v, u_v), size=(batch_size))
    radius = torch.ones(batch_size) * r_v
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_camera_pose(range_u, range_v, range_r, val_u=0.5, val_v=0.5, val_r=0.5,
                    batch_size=32, invert=False):
    u0, ur = range_u[0], range_u[1] - range_u[0]
    v0, vr = range_v[0], range_v[1] - range_v[0]
    r0, rr = range_r[0], range_r[1] - range_r[0]
    u = u0 + val_u * ur
    v = v0 + val_v * vr
    r = r0 + val_r * rr

    loc = sample_on_sphere((u, u), (v, v), size=(batch_size))
    radius = torch.ones(batch_size) * r
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,),
                     to_pytorch=True):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    sample = to_sphere(u, v)
    if to_pytorch:
        sample = torch.tensor(sample).float()

    return sample


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5,
            to_pytorch=True):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)

    if to_pytorch:
        r_mat = torch.tensor(r_mat).float()

    return r_mat


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_dcm()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r
