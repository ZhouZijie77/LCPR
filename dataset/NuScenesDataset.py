import os
import h5py
import torch
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, data_root_dir, info_path):
        super().__init__()
        self.infos = self.read_infos(info_path)
        self.dataroot = data_root_dir
        self.img_transforms = None

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_infos(self, info_path):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos

    def load_data(self, index):

        channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

        camera_images = self.load_camera_data(index, channels)

        lidar_image = self.load_lidar_data(index)

        return camera_images, lidar_image

    def load_lidar_data(self, index):
        lidar_data = self.infos[index]['lidar_infos']['LIDAR_TOP']
        fname = os.path.split(lidar_data['filename'].split('.')[0])[-1] + '.npy'
        depth_data_path = os.path.join(self.dataroot, 'samples', 'RANGE_DATA', fname)
        if not os.path.exists(depth_data_path):
            raise Exception(f'FileNotFound! {depth_data_path}')
        depth_data = np.load(depth_data_path, allow_pickle=True)
        depth_data_tensor = torch.from_numpy(depth_data).float().unsqueeze(0)

        return depth_data_tensor

    def load_camera_data(self, index, channels):
        imgs = []
        for channel in channels:
            cam_data = self.infos[index]['cam_infos'][channel]
            filename = cam_data['filename']
            img_path = os.path.join(self.dataroot, filename)
            if not os.path.exists(img_path):
                raise Exception(f'FileNotFound! {img_path}')
            img = Image.open(img_path)
            img_tensor = self.img_transforms(img)
            imgs.append(img_tensor)

        imgs = torch.stack(imgs)
        return imgs


class TripletDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, cache_dir, img_transforms, nNeg, nNegSample,
                 nonTrivPosDistThres, posDistThr, margin):
        super().__init__(data_root_dir, info_path)
        self.data_base = np.load(database_path)
        self.nNeg = nNeg  # number of negatives used for training
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nonTrivPosDistThres = nonTrivPosDistThres
        self.posDistThr = posDistThr
        self.margin = margin
        self.queries = np.load(query_path)
        self.img_transforms = img_transforms

        knn = NearestNeighbors()
        knn.fit(self.data_base[:, 1:])
        self.nontrivial_positives = list(knn.radius_neighbors(self.queries[:, 1:],
                                                              radius=self.nonTrivPosDistThres,
                                                              return_distance=False))
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)

        potential_positives = list(knn.radius_neighbors(self.queries[:, 1:], radius=self.posDistThr,
                                                        return_distance=False))
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.data_base.shape[0]), pos,
                                                         assume_unique=True))

        # filepath of HDF5 containing feature vectors for this epoch
        self.cache = os.path.join(cache_dir, 'feat_cache.hdf5')
        self.negCache = [np.empty((0,)) for _ in range(len(self.queries))]

    def __getitem__(self, index):
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get('features')
            qOffset = len(self.data_base)
            qFeat = h5feat[index + qOffset]
            qFeat = torch.tensor(qFeat)
            posFeat = h5feat[self.nontrivial_positives[index]]
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1)
            result = dist.topk(1, largest=False)  # find the closest positive
            posdist, posidx = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posidx].item()
            if self.negCache[index].ndim == 0:
                self.negCache[index] = self.negCache[index].reshape(1)

            # randomly select from potential negatives
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]).astype(int))
            negFeat = h5feat[negSample]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1)
            result = dist.topk(self.nNeg * 10, largest=False)
            negdist, negidx = result.values, result.indices

            # try to find negatives that are within margin, if there aren't any return none
            vilatingNeg = negdist.numpy() < posdist.numpy() + self.margin
            if np.sum(vilatingNeg) < 1:
                return None

            negidx = negidx[vilatingNeg][:self.nNeg]
            nNeg = len(negidx)
            negIndex = negSample[negidx].astype(int)
            self.negCache[index] = negIndex

        query_idx = int(self.queries[index][0])
        pos_idx = int(self.data_base[posIndex][0])

        q_camera_images, q_lidar_image = \
            self.load_data(query_idx)
        pos_camera_images, pos_lidar_image = \
            self.load_data(pos_idx)

        if self.data_base[negIndex].ndim == 1:
            negIndex = np.array([negIndex])
        neg_idx = self.data_base[negIndex][:, 0].astype(int)
        camera_images, lidar_images = [], []
        for i in range(len(neg_idx)):
            neg_camera_images, neg_lidar_image = self.load_data(neg_idx[i])
            camera_images.append(neg_camera_images)
            lidar_images.append(neg_lidar_image)

        camera_images.extend([pos_camera_images, q_camera_images])

        lidar_images.extend([pos_lidar_image, q_lidar_image])

        res_dict = dict({'camera_feature': torch.stack(camera_images),
                         'lidar_feature': torch.stack(lidar_images)})
        return res_dict, nNeg

    def __len__(self):
        return len(self.queries)

    def read_infos(self, infos_path):
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)
        return infos


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    input_dict, nNeg = zip(*batch)

    return default_collate(input_dict), list(nNeg)


class DatabaseQueryDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, transforms, nonTrivPosDistThres):
        super().__init__(data_root_dir, info_path)
        data_base = np.load(database_path)
        query = np.load(query_path)

        self.dataset = np.concatenate((data_base, query), axis=0)
        self.num_db = len(data_base)
        self.num_query = len(query)
        self.positives = None
        self.distances = None
        self.cache = None
        self.nonTrivPosDistThres = nonTrivPosDistThres
        self.img_transforms = transforms

    def __getitem__(self, item):
        index = int(self.dataset[item][0])
        camera_images, lidar_image = self.load_data(index)
        res = dict({'camera_feature': camera_images,
                    'lidar_feature': lidar_image})
        return res

    def __len__(self):
        return len(self.dataset)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            dataset = np.ascontiguousarray(self.dataset[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset[self.num_db:, 1:], radius=self.nonTrivPosDistThres,
                                                       return_distance=False))
        return self.positives
