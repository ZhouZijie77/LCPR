import h5py
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
import faiss
import gc


class Trainer:
    def __init__(self, model, train_loader, whole_train_loader, whole_val_set, whole_val_loader,
                 device, num_epochs, resume_path, log, log_dir, ckpt_dir, cache_dir,
                 resume_scheduler, lr, step_size, gamma, margin):
        self.train_loader = train_loader
        self.whole_train_loader = whole_train_loader
        self.whole_val_set = whole_val_set
        self.whole_val_loader = whole_val_loader
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.device = device

        self.num_epochs = num_epochs
        self.resume_path = resume_path
        self.log = log
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.cache_dir = cache_dir
        self.output_dim = 256
        self.resume_scheduler = resume_scheduler
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.margin = margin

    def train(self):
        print("Start training ...")
        if self.resume_path is not None:
            print("Resuming from ", self.resume_path)
            checkpoint = torch.load(self.resume_path)
            starting_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['net'])
            del checkpoint
        else:
            print("Training from scratch ...")
            starting_epoch = 0

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)

        if self.resume_path and self.resume_scheduler:
            print("Resuming scheduler")
            checkpoint = torch.load(self.resume_path)
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint
        writer = None

        if self.log:
            time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
            writer = SummaryWriter(log_dir=os.path.join(self.log_dir, time_stamp))

        self.model = self.model.to(self.device)

        for epoch in range(starting_epoch + 1, self.num_epochs):
            print("============================================\n")
            print('epoch: ', epoch)
            print('number of queries: ', len(self.train_loader))
            print('learning rate: ', self.optimizer.state_dict()['param_groups'][0]['lr'])
            if self.log:
                writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            print("\n============================================\n")

            # -------------------------build cache---------------------------------
            print('building cache..')

            self.model.eval()
            self.build_cache()

            # -------------------------training---------------------------------
            self.model.train()
            loss_each_epoch = 0
            used_num = 0
            for i, (input_dict, nNeg) in enumerate(self.train_loader):
                if input_dict is None:
                    continue
                used_num += 1
                camera_feature = input_dict['camera_feature'].squeeze(0).to(self.device)

                lidar_feature = input_dict['lidar_feature'].squeeze(0).to(self.device)
                nNeg = nNeg[0]

                self.optimizer.zero_grad()

                global_des = self.model(camera_feature, lidar_feature)
                del camera_feature, lidar_feature

                neg_des, pos_des, query_des = torch.split(
                    global_des, [nNeg, 1, 1], dim=0)

                query_des = query_des.expand(nNeg, -1)
                pos_des = pos_des.expand(nNeg, -1)

                loss = nn.functional.triplet_margin_loss(query_des, pos_des, neg_des, margin=self.margin)

                if torch.isnan(loss):
                    print('something wrong!!!')
                    continue

                del query_des, pos_des, neg_des
                loss.backward()
                self.optimizer.step()
                print(used_num, 'loss: ', loss.item())

                loss_each_epoch = loss_each_epoch + loss.item()

            self.scheduler.step()
            print("epoch {} loss {}".format(epoch, loss_each_epoch / used_num))
            print("saving weights ...")
            ckpt_path = os.path.join(self.ckpt_dir, 'LCPR_epoch_' + str(epoch) + '.pth.tar')

            checkpoint = {'epoch': epoch,
                          'net': self.model.state_dict(),
                          'scheduler': self.scheduler.state_dict()
                          }
            torch.save(checkpoint, ckpt_path)
            print("Model Saved As " + 'LCPR_epoch_' + str(epoch) + '.pth.tar')
            if self.log:
                writer.add_scalar("train/loss", loss_each_epoch / used_num, global_step=epoch)

            recalls = self.val()

            if self.log:
                for n, recall in recalls.items():
                    writer.add_scalar('val/recall@{}'.format(n), recall, epoch)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def val(self):
        self.model.eval()
        with torch.no_grad():
            print('getting recall..')
            cache = os.path.join(self.cache_dir, 'feat_cache_val.hdf5')
            with h5py.File(cache, mode='w') as h5:
                dbFeat = h5.create_dataset('features', [len(self.whole_val_set), self.output_dim], dtype=np.float32)
                for i, input_dict in enumerate(tqdm(self.whole_val_loader)):
                    camera_feature = input_dict['camera_feature'].to(self.device)
                    lidar_feature = input_dict['lidar_feature'].to(self.device)
                    descriptor = self.model(camera_feature, lidar_feature)
                    dbFeat[i, :] = descriptor.cpu().numpy()

        with h5py.File(cache, mode='r') as h5:
            dbFeat = h5.get('features')

            n_values = [1, 5, 10, 20]
            qFeat = dbFeat[self.whole_val_set.num_db:].astype('float32')
            dbFeat = dbFeat[:self.whole_val_set.num_db].astype('float32')
            faiss_index = faiss.IndexFlatL2(self.output_dim)
            faiss_index.add(dbFeat)
            dists, predictions = faiss_index.search(qFeat, len(dbFeat))  # the results are sorted

            # for each query get those within threshold distance
            gt = self.whole_val_set.getPositives()
            correct_at_n = np.zeros(len(n_values))
            for qIx, pred in enumerate(predictions):
                for i, n in enumerate(n_values):
                    if np.any(np.in1d(pred[:n], gt[qIx])):
                        correct_at_n[i:] += 1
                        break
            recall_at_n = correct_at_n / self.whole_val_set.num_query * 100.0

            recalls = {}  # make dict for output
            for i, n in enumerate(n_values):
                recalls[n] = recall_at_n[i]

            print('[validate]')
            print('recall@1: {:.2f}\t'.format(recalls[1]), end='')
            print('recall@5: {:.2f}\t'.format(recalls[5]), end='')
            print('recall@10: {:.2f}\t'.format(recalls[10]), end='')
            print('recall@20: {:.2f}\t'.format(recalls[20]))

            return recalls

    def build_cache(self):
        with h5py.File(os.path.join(self.cache_dir, "feat_cache.hdf5"), mode='w') as h5:
            h5feat = h5.create_dataset('features', [len(self.whole_train_loader), self.output_dim], dtype=np.float32)
            with torch.no_grad():
                for i, input_dict in enumerate(tqdm(self.whole_train_loader)):
                    camera_feature = input_dict['camera_feature'].to(self.device)
                    lidar_feature = input_dict['lidar_feature'].to(self.device)
                    descriptor = self.model(camera_feature, lidar_feature)
                    h5feat[i, :] = descriptor.cpu().numpy()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Evaluator:
    def __init__(self, model, whole_test_set, whole_test_loader, result_dir, device):
        self.whole_test_set = whole_test_set
        self.whole_test_loader = whole_test_loader
        self.model = model
        self.output_dim = 256
        self.result_dir = result_dir
        self.device = device

    @torch.no_grad()
    def get_feature(self, ckpt_path, feature_path):
        print('=================evaluating=================')
        assert ckpt_path is not None
        print('load weights from: ', ckpt_path)
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['net'])
        self.model = self.model.to(self.device)
        print('predicting.....')
        print('database: ', self.whole_test_set.num_db, 'test query: ', self.whole_test_set.num_query)
        self.model.eval()

        with torch.no_grad():
            gt = self.whole_test_set.getPositives()
            Feat = np.empty((len(self.whole_test_set), self.output_dim))
            for i, input_dict in enumerate(tqdm(self.whole_test_loader)):
                camera_feature = input_dict['camera_feature'].to(self.device)
                lidar_feature = input_dict['lidar_feature'].to(self.device)
                descriptor = self.model(camera_feature, lidar_feature)
                Feat[i, :] = descriptor.detach().cpu().numpy()
                del input_dict, descriptor

            qFeat = Feat[self.whole_test_set.num_db:].astype('float32')
            dbFeat = Feat[:self.whole_test_set.num_db].astype('float32')

            with open(feature_path, 'wb') as f:
                feature = {'qFeat': qFeat, 'dbFeat': dbFeat, 'gt': gt}
                pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('saved at: ', feature_path)

    def get_recall_at_n(self, feature_path):
        assert feature_path is not None
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f)
            qFeat = feature['qFeat']
            dbFeat = feature['dbFeat']
            gt = feature['gt']

        n_values = [1, 5, 10]
        # ----------------------------------------------------- faiss --------------------------------------------
        faiss_index = faiss.IndexFlatL2(self.output_dim)
        faiss_index.add(dbFeat)
        # dists, predictions = faiss_index.search(qFeat, max(n_values))   # the results is sorted
        dists, preds = faiss_index.search(qFeat, len(dbFeat))  # the results are sorted
        # ------------------------------------------------------- - -----------------------------------------------

        correct_at_n = np.zeros(len(n_values))
        print('getting recall...')
        for qIx, pred in enumerate(preds):
            for i, n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n = correct_at_n / self.whole_test_set.num_query * 100.0

        recalls = {}  # make dict for output
        for i, n in enumerate(n_values):
            recalls[n] = recall_at_n[i]

        print('[test]')
        for i, n in enumerate(n_values):
            print('recall@', n, ': {:.2f}\t'.format(recalls[n]), end='')
        print()
        return recalls

    def get_pr(self, feature_path, vis=True):
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f)
            qFeat = feature['qFeat']
            dbFeat = feature['dbFeat']
            gt = feature['gt']

        faiss_index = faiss.IndexFlatL2(self.output_dim)
        faiss_index.add(dbFeat)
        dists, preds = faiss_index.search(qFeat, len(dbFeat))  # the results are sorted
        dists_max = dists[:, 0].max()
        dists_min = dists[:, 0].min()
        if dists_min - 0.1 > 0:
            dists_min -= 0.1
        dists_u = np.linspace(dists_min, dists_max + 0.1, 1000)

        recalls = []
        precisions = []
        print('getting pr...')
        for th in tqdm(dists_u, ncols=40):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            for index_q in range(dists.shape[0]):
                # Positive
                if dists[index_q, 0] < th:
                    # True
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        TPCount += 1
                    else:
                        FPCount += 1
                else:
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        FNCount += 1
                    else:
                        TNCount += 1
            assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            recalls.append(recall)
            precisions.append(precision)
        return recalls, precisions

    def get_f1score(self, recalls, precisions):
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ind = np.argsort(recalls)
        recalls = recalls[ind]
        precisions = precisions[ind]
        f1s = []
        for index_j in range(len(recalls)):
            f1 = 2 * precisions[index_j] * recalls[index_j] / (precisions[index_j] + recalls[index_j])
            f1s.append(f1)

        print('f1 score: ', max(f1s))
        return f1s
