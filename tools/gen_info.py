import os
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


def gen_info(nusc, sample_tokens):
    infos = list()
    for sample_token in tqdm(sample_tokens):
        sample = nusc.get('sample', sample_token)
        info = dict()
        cam_datas = list()
        lidar_datas = list()
        info['sample_token'] = sample_token
        info['timestamp'] = sample['timestamp']
        info['scene_token'] = sample['scene_token']
        cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]
        lidar_names = ['LIDAR_TOP']
        cam_infos = dict()
        lidar_infos = dict()
        for cam_name in cam_names:
            cam_data = nusc.get('sample_data', sample['data'][cam_name])
            cam_datas.append(cam_data)
            cam_info = dict()
            cam_info['sample_token'] = cam_data['sample_token']
            cam_info['ego_pose'] = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_info['timestample'] = cam_data['timestamp']
            cam_info['filename'] = cam_data['filename']
            cam_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_infos[cam_name] = cam_info
        for lidar_name in lidar_names:
            lidar_data = nusc.get('sample_data',
                                  sample['data'][lidar_name])
            lidar_datas.append(lidar_data)
            lidar_info = dict()
            lidar_info['sample_token'] = lidar_data['sample_token']
            lidar_info['ego_pose'] = nusc.get(
                'ego_pose', lidar_data['ego_pose_token'])
            lidar_info['timestamp'] = lidar_data['timestamp']
            lidar_info['filename'] = lidar_data['filename']
            lidar_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_infos[lidar_name] = lidar_info
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        infos.append(info)

    return infos


def get_location_sample_tokens(nusc, location):
    # Get the sample tokens of a specific location

    location_indices = get_location_indices(nusc, location)

    sample_token_list = []

    for scene_index in location_indices:
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']

        while not sample_token == '':
            sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            sample_token = sample['next']

    return sample_token_list


def get_location_indices(nusc, location):
    location_indices = []
    for scene_index in range(len(nusc.scene)):
        scene = nusc.scene[scene_index]
        if nusc.get('log', scene['log_token'])['location'] != location:
            continue
        location_indices.append(scene_index)
    return np.array(location_indices)


def main():
    dataroot = '/media/zzj/DATA/DataSets/nuScenes'
    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot=dataroot, verbose=True)

    # ====================generate infos====================
    dataroot = '/media/zzj/DATA/DataSets/nuScenes'
    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='boston-seaport')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='boston-seaport')
    infos = gen_info(nusc_trainval, sample_tokens_trainval)
    infos.extend(gen_info(nusc_test, sample_tokens_test))
    with open(os.path.join(dataroot, 'nuscenes_infos-bs.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-onenorth')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='singapore-onenorth')
    infos = gen_info(nusc_trainval, sample_tokens_trainval)
    infos.extend(gen_info(nusc_test, sample_tokens_test))
    with open(os.path.join(dataroot, 'nuscenes_infos-son.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-hollandvillage')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='singapore-hollandvillage')
    infos = gen_info(nusc_trainval, sample_tokens_trainval)
    infos.extend(gen_info(nusc_test, sample_tokens_test))
    with open(os.path.join(dataroot, 'nuscenes_infos-shv.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-queenstown')
    sample_tokens_test = get_location_sample_tokens(nusc_test, location='singapore-queenstown')
    infos = gen_info(nusc_trainval, sample_tokens_trainval)
    infos.extend(gen_info(nusc_test, sample_tokens_test))
    with open(os.path.join(dataroot, 'nuscenes_infos-sq.pkl'), 'wb') as f:
        pickle.dump(infos, f)


if __name__ == '__main__':
    main()
