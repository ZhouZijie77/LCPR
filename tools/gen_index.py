import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle


def main():
    random.seed(1)
    with open('/media/zzj/DATA/DataSets/nuScenes/nuscenes_infos-boston.pkl', 'rb') as f:
        infos_boston = pickle.load(f)

    with open('/media/zzj/DATA/DataSets/nuScenes/nuscenes_infos-son.pkl', 'rb') as f:
        infos_son = pickle.load(f)

    with open('/media/zzj/DATA/DataSets/nuScenes/nuscenes_infos-shv.pkl', 'rb') as f:
        infos_shv = pickle.load(f)

    with open('/media/zzj/DATA/DataSets/nuScenes/nuscenes_infos-sq.pkl', 'rb') as f:
        infos_sq = pickle.load(f)

    dataroot = '/media/zzj/DATA/DataSets/nuScenes'
    pos_whole_bs = []
    pos_whole_son = []
    pos_whole_shv = []
    pos_whole_sq = []
    timestamps_bs = []

    for i, info in enumerate(infos_boston):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_bs.append(pos[:2])
        timestamp = info['timestamp']
        timestamps_bs.append(timestamp)

    for i, info in enumerate(infos_son):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_son.append(pos[:2])

    for i, info in enumerate(infos_shv):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_shv.append(pos[:2])

    for i, info in enumerate(infos_sq):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_sq.append(pos[:2])

    pos_whole_bs = np.array(pos_whole_bs, dtype=np.float32)
    pos_whole_son = np.array(pos_whole_son, dtype=np.float32)
    pos_whole_shv = np.array(pos_whole_shv, dtype=np.float32)
    pos_whole_sq = np.array(pos_whole_sq, dtype=np.float32)
    timestamps_bs = np.array(timestamps_bs, dtype=np.float32).reshape(-1, 1)

    print('total frames for boston seaport: ', pos_whole_bs.shape[0])
    print('total frames for singapore one north: ', pos_whole_son.shape[0])
    print('total frames for singapore holland village: ', pos_whole_shv.shape[0])
    print('total frames for singapore queenstown: ', pos_whole_sq.shape[0])

    # ==================================================================
    #                    generate database indices
    # ==================================================================
    print('==> generating database...')
    DIS_TH = 1  # Map Point Distance (m)

    pos_whole_bs = np.concatenate(
        (np.arange(len(pos_whole_bs), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_bs)),
        axis=1).astype(np.float32)

    pos_whole_son = np.concatenate(
        (np.arange(len(pos_whole_son), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_son)),
        axis=1).astype(np.float32)

    pos_whole_shv = np.concatenate(
        (np.arange(len(pos_whole_shv), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_shv)),
        axis=1).astype(np.float32)

    pos_bs_db = pos_whole_bs[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_bs.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_bs_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_bs[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH:
            pos_bs_db = np.concatenate((pos_bs_db, pos_whole_bs[i, :].reshape(1, -1)), axis=0)

    pos_son_db = pos_whole_son[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_son.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_son_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_son[i, 1:4].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH:
            pos_son_db = np.concatenate((pos_son_db, pos_whole_son[i, :].reshape(1, -1)), axis=0)

    pos_shv_db = pos_whole_shv[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_shv.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_shv_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_shv[i, 1:4].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH:
            pos_shv_db = np.concatenate((pos_shv_db, pos_whole_shv[i, :].reshape(1, -1)), axis=0)

    pos_sq_db = pos_whole_sq[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_sq.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_sq_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_sq[i, 1:4].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH:
            pos_sq_db = np.concatenate((pos_sq_db, pos_whole_sq[i, :].reshape(1, -1)), axis=0)

    print('boston seaport database frames: ', pos_bs_db.shape[0])
    print('singapore one north database frames: ', pos_son_db.shape[0])
    print('singapore holland village database frames: ', pos_shv_db.shape[0])
    print('singapore queenstown database frames: ', pos_sq_db.shape[0])

    # ==================================================================
    # generate train_query, val_query, test_query indices
    # ==================================================================
    print('==> generating query indices...')
    timestamps = np.array(timestamps_bs - min(timestamps_bs)) / (3600 * 24 * 1e6)
    SEPERATE_TH = 105
    fi_bs_train, _ = np.where(timestamps < SEPERATE_TH)
    fi_bs_testval, _ = np.where(timestamps >= SEPERATE_TH)

    fi_bs_db = pos_bs_db[:, 0].astype(int)
    fi_bs_train_query = list(set(fi_bs_train) - set(fi_bs_db))
    fi_bs_testval_query = list(set(fi_bs_testval) - set(fi_bs_db))
    fi_bs_val_query = random.sample(fi_bs_testval_query, int(len(fi_bs_testval_query) * 0.25))
    fi_bs_test_query = list(set(fi_bs_testval_query) - set(fi_bs_val_query))
    pos_bs_train_query = pos_whole_bs[fi_bs_train_query]
    pos_bs_test_query = pos_whole_bs[fi_bs_test_query]
    pos_bs_val_query = pos_whole_bs[fi_bs_val_query]

    fi_son_db = pos_son_db[:, 0].astype(int)
    fi_son_whole = pos_whole_son[:, 0].astype(int)
    fi_son_query = list(set(fi_son_whole) - set(fi_son_db))
    fi_son_query = np.array(list(fi_son_query))
    pos_son_query = pos_whole_son[fi_son_query]

    fi_shv_db = pos_shv_db[:, 0].astype(int)
    fi_shv_whole = pos_whole_shv[:, 0].astype(int)
    fi_shv_query = list(set(fi_shv_whole) - set(fi_shv_db))
    fi_shv_query = np.array(list(fi_shv_query))
    pos_shv_query = pos_whole_shv[fi_shv_query]

    fi_sq_db = pos_sq_db[:, 0].astype(int)
    fi_sq_whole = pos_whole_sq[:, 0].astype(int)
    fi_sq_query = list(set(fi_sq_whole) - set(fi_sq_db))
    fi_sq_train_query = random.sample(fi_sq_query, int(len(fi_sq_query) * 0.10))
    fi_sq_test_query = list(set(fi_sq_query) - set(fi_sq_train_query))
    pos_sq_train_query = pos_whole_sq[fi_sq_train_query]
    pos_sq_test_query = pos_whole_sq[fi_sq_test_query]

    # ==================================================================
    # delete train/val/test query who has no GT positive in the database
    # ==================================================================
    DIS_TH = 9
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_bs_db[:, 1:3])

    pos_bs_train_query_new = list()
    for i in range(len(pos_bs_train_query)):
        dis, index = knn.kneighbors(pos_bs_train_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_train_query_new.append(pos_bs_train_query[i, :])
    pos_bs_train_query = np.array(pos_bs_train_query_new)

    pos_bs_test_query_new = list()
    for i in range(len(pos_bs_test_query)):
        dis, index = knn.kneighbors(pos_bs_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_test_query_new.append(pos_bs_test_query[i, :])
    pos_bs_test_query = np.array(pos_bs_test_query_new)

    pos_bs_val_query_new = list()
    for i in range(len(pos_bs_val_query)):
        dis, index = knn.kneighbors(pos_bs_val_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_val_query_new.append(pos_bs_val_query[i, :])
    pos_bs_val_query = np.array(pos_bs_val_query_new)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_son_db[:, 1:3])

    pos_son_query_new = list()
    for i in range(len(pos_son_query)):
        dis, index = knn.kneighbors(pos_son_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_son_query_new.append(pos_son_query[i, :])
    pos_son_query = np.array(pos_son_query_new)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_shv_db[:, 1:3])

    pos_shv_query_new = list()
    for i in range(len(pos_shv_query)):
        dis, index = knn.kneighbors(pos_shv_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_shv_query_new.append(pos_shv_query[i, :])
    pos_shv_query = np.array(pos_shv_query_new)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_sq_db[:, 1:3])

    pos_sq_train_query_new = list()
    for i in range(len(pos_sq_train_query)):
        dis, index = knn.kneighbors(pos_sq_train_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_sq_train_query_new.append(pos_sq_train_query[i, :])
    pos_sq_train_query = np.array(pos_sq_train_query_new)

    pos_sq_test_query_new = list()
    for i in range(len(pos_sq_test_query)):
        dis, index = knn.kneighbors(pos_sq_test_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_sq_test_query_new.append(pos_sq_test_query[i, :])
    pos_sq_test_query = np.array(pos_sq_test_query_new)

    print('boston seaport train query: ', pos_bs_train_query.shape[0])
    print('boston seaport test query: ', pos_bs_test_query.shape[0])
    print('boston seaport val query: ', pos_bs_val_query.shape[0])
    print('singapore one north query: ', pos_son_query.shape[0])
    print('singapore holland village query: ', pos_shv_query.shape[0])
    print('singapore queenstown train query: ', pos_sq_train_query.shape[0])
    print('singapore queenstown test query: ', pos_sq_test_query.shape[0])

    # ============================================================
    # save database, train queries, test queries, validate queries
    # ============================================================

    print('===> saving database and queries..')
    np.save(os.path.join(dataroot, 'bs_db.npy'), pos_bs_db)
    np.save(os.path.join(dataroot, 'bs_train_query.npy'), pos_bs_train_query)
    np.save(os.path.join(dataroot, 'bs_val_query.npy'), pos_bs_val_query)
    np.save(os.path.join(dataroot, 'bs_test_query.npy'), pos_bs_test_query)
    np.save(os.path.join(dataroot, 'son_db.npy'), pos_son_db)
    np.save(os.path.join(dataroot, 'son_query.npy'), pos_son_query)
    np.save(os.path.join(dataroot, 'shv_db.npy'), pos_shv_db)
    np.save(os.path.join(dataroot, 'shv_query.npy'), pos_shv_query)
    np.save(os.path.join(dataroot, 'sq_train_query.npy'), pos_sq_train_query)
    np.save(os.path.join(dataroot, 'sq_test_query.npy'), pos_sq_test_query)
    np.save(os.path.join(dataroot, 'sq_db.npy'), pos_sq_db)


if __name__ == '__main__':
    main()
