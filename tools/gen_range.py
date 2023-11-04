import sys
sys.path.append('../')
import os
import pickle
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from tools.utils import range_projection, eulerAngles2rotationMat, check_dir


def gen_range_data(dataroot, infos, target_folder, fov_up, fov_down, proj_H, proj_W):
    assert os.path.exists(target_folder)
    for i in range(len(infos)):
        fname = infos[i]['lidar_infos']['LIDAR_TOP']['filename']
        pcl = LidarPointCloud.from_file(os.path.join(dataroot, fname))
        pcl.remove_close(radius=2)
        points = pcl.points[:3, :]  # 3 x N
        rot_mat = eulerAngles2rotationMat(angles=[0, 0, -90])
        points = rot_mat.dot(points)
        proj_range, proj_vertex, _ = range_projection(points.T, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H,
                                                      proj_W=proj_W,
                                                      max_range=80,
                                                      cut_z=False)

        dst_path = os.path.join(target_folder, os.path.split(fname.split('.')[0])[-1]) + '.npy'
        np.save(dst_path, proj_range)
        print('finished generating depth data at: ', dst_path)


def main():
    fov_up = 10
    fov_down = -30
    proj_H = 32
    proj_W = 1056
    dataroot = '/media/zzj/DATA/DataSets/nuScenes'

    with open(os.path.join(dataroot, 'nuscenes_infos-bs.pkl'), 'rb') as f:
        infos = pickle.load(f)
    target_folder = os.path.join(dataroot, 'samples/RANGE_DATA')
    check_dir(target_folder)
    gen_range_data(dataroot, infos, target_folder, fov_up, fov_down, proj_H, proj_W)

    with open(os.path.join(dataroot, 'nuscenes_infos-son.pkl'), 'rb') as f:
        infos = pickle.load(f)
    target_folder = os.path.join(dataroot, 'samples/RANGE_DATA')
    check_dir(target_folder)
    gen_range_data(dataroot, infos, target_folder, fov_up, fov_down, proj_H, proj_W)

    with open(os.path.join(dataroot, 'nuscenes_infos-shv.pkl'), 'rb') as f:
        infos = pickle.load(f)
    target_folder = os.path.join(dataroot, 'samples/RANGE_DATA')
    check_dir(target_folder)
    gen_range_data(dataroot, infos, target_folder, fov_up, fov_down, proj_H, proj_W)

    with open(os.path.join(dataroot, 'nuscenes_infos-sq.pkl'), 'rb') as f:
        infos = pickle.load(f)
    target_folder = os.path.join(dataroot, 'samples/RANGE_DATA')
    check_dir(target_folder)
    gen_range_data(dataroot, infos, target_folder, fov_up, fov_down, proj_H, proj_W)


if __name__ == '__main__':
    main()
