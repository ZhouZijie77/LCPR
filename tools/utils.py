import math
import numpy as np
import os
import yaml


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        return cfg


def eulerAngles2rotationMat(angles, mode='degree', dim=3):
    if dim == 3:
        if len(angles) != 3:
            raise Exception('dimensionality error!!')
        if mode == 'degree':
            angles = [i * math.pi / 180.0 for i in angles]

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(angles[0]), -math.sin(angles[0])],
                        [0, math.sin(angles[0]), math.cos(angles[0])]
                        ])

        R_y = np.array([[math.cos(angles[1]), 0, math.sin(angles[1])],
                        [0, 1, 0],
                        [-math.sin(angles[1]), 0, math.cos(angles[1])]
                        ])

        R_z = np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                        [math.sin(angles[2]), math.cos(angles[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    else:
        if mode == 'degree':
            angles = angles / 180 * math.pi

        R = np.array([[math.cos(angles), -math.sin(angles)],
                      [math.sin(angles), math.cos(angles)]])
        return R


def range_projection(current_vertex, fov_up, fov_down, proj_H, proj_W, max_range=80, cut_z=False,
                     low=0.1, high=6):
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    if cut_z:
        z = current_vertex[:, 2]
        kept = (depth > 0) & (depth < max_range) & (z < high) & (z > low)
    else:
        kept = (depth > 0) & (depth < max_range)
    current_vertex = current_vertex[kept]  # get rid of [0, 0, 0] points
    depth = depth[kept]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx


def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def check_path(*paths):
    for p in paths:
        assert os.path.exists(p)
