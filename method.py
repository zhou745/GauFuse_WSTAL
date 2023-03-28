from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


import utils.utils
from evaluation.eval import ft_eval, ss_eval
from model.memory import Memory
from model.main_branch import WSTAL, random_walk, predictor_svd
from model.losses import NormalizedCrossEntropy, AttLoss, CategoryCrossEntropy
from utils.video_dataloader import VideoDataset
# from tensorboard_logger import Logger
from torch.utils.tensorboard import SummaryWriter

class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # create logger
        log_dir = './logs/' + self.args.dataset_name + '/' + str(self.args.model_id)
        # self.logger = Logger(log_dir)
        self.logger = SummaryWriter(log_dir)
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')

        #

        # dataloader
        if self.args.dataset_name in ['Thumos14', 'Thumos14reduced']:
            if self.args.run_type == 0:
                self.train_dataset = VideoDataset(self.args, 'train')
                self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                                     batch_size=1,
                                                                     shuffle=True,
                                                                     num_workers=0,
                                                                     drop_last=False)
                self.train_data_loader_tmp = torch.utils.data.DataLoader(self.train_dataset, batch_size=1,
                                                                         shuffle=False, drop_last=False,num_workers=0)
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'test'), batch_size=1,
                                                                    shuffle=False, drop_last=False,num_workers=0)
            elif self.args.run_type == 1:
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'test'), batch_size=1,
                                                                    shuffle=False, drop_last=False,num_workers=0)
        else:
            raise ValueError('Do Not Exist This Dataset')

        # Loss Function Setting
        self.loss_att = AttLoss(8.0)
        self.loss_nce = NormalizedCrossEntropy()
        self.loss_spl = CategoryCrossEntropy(self.args.T)

        # Model Setting
        self.model = WSTAL(self.args).to(self.device)
        self.memory = Memory(self.args).to(self.device)
        #distilation model setting by zhoujq
        self.dist_model = WSTAL(self.args).to(self.device)

        # self.predictor = predictor_svd(sharp_factor=2.0, base_value = 1.0).to(self.device)

        # Model Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model_module = self.model.module

            self.dist_model =  nn.DataParallel(self.dist_model, device_ids=self.args.gpu_ids)
            self.dist_model_module = self.dist_model.module

            # self.predictor = nn.DataParallel(self.predictor, device_ids=self.args.gpu_ids)
            # self.predictor_module = self.predictor.module
        else:
            self.model_module = self.model
            self.dist_model_module = self.dist_model
            # self.predictor_module = self.predictor


        # Loading Pretrained Model
        if self.args.run_type ==1:
            model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                self.args.load_epoch) + '.pkl'
            if os.path.isfile(model_dir):
                self.model_module.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')

        if self.args.pretrained and self.args.pretrained_epoch is not None and self.args.run_type !=1:
            model_dir = self.args.load_ckpt_path+"/"+str(self.args.pretrained_epoch)+".pkl"
            if os.path.isfile(model_dir):
                self.model_module.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')


        #ensure weights are the same in the beginning by zhoujq
        self.update_model(polyak_factor=1.0)
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, betas=[0.9, 0.99],
                                            weight_decay=self.args.weight_decay)

        # Optimizer Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.args.gpu_ids)
            self.optimizer_module = self.optimizer.module
        else:
            self.optimizer_module = self.optimizer

    def processing(self):
        if self.args.run_type == 0:
            self.train()
        elif self.args.run_type == 1:
            self.val(self.args.load_epoch)
        else:
            raise ValueError('Do not Exist This Processing')

    def update_model(self, polyak_factor=1.0):
        for target_param, param in zip(self.dist_model.parameters(),self.model.parameters()):
            target_param.data.copy_(polyak_factor * param.data + target_param.data * (1.0 - polyak_factor))

    def train(self):
        print('Start training!')
        self.model_module.train(mode=True)

        if self.args.pretrained and self.args.pretrained_epoch is not None:
            epoch_range = range(self.args.pretrained_epoch, self.args.max_epoch)
        else:
            epoch_range = range(self.args.max_epoch)

        iter = 0
        step = 0
        current_lr = self.args.lr

        #by zhoujq
        loss_recorder = {
            'cls_fore': 0,
            'cls_back': 0,
            'att': 0,
            'spl': 0,
            'KD': 0,
        }
        feature_list = []
        #precompute pretrained feature for dist
        for idx in self.args.load_dist_ckpt:
            model_dir = self.args.load_ckpt_path + '/' + str(idx) + '.pkl'
            if os.path.isfile(model_dir):
                self.dist_model_module.load_state_dict(torch.load(model_dir))
            feature_list.append([None for idx in range(len(self.train_data_loader))])
            # precompute all the features and record it
            for num, sample in tqdm(enumerate(self.train_data_loader)):
                np_features = sample['data'].numpy()
                idx = sample['idx'].numpy()
                np_labels = sample['labels'].numpy()

                features = torch.from_numpy(np_features).float().to(self.device)
                labels = torch.from_numpy(np_labels).float().to(self.device)
                f_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)
                # compute the prediction of the distilation model by zhoujq
                with torch.no_grad():
                    d_o_out, d_m_out, d_em_out = self.dist_model(features)
                # compute the self distlation psuedo label
                pred_label = utils.utils.fuse_current_pred(d_o_out, d_m_out, d_em_out, f_labels, self.args)
                feature_list[-1][idx[0]] = pred_label

        print("compute dist feature finished")

        for epoch in epoch_range:

            if epoch in self.args.dist_epoch:
                self.memory._emprty_queue()
                self.update_model(polyak_factor=1.)

                idx = self.args.dist_epoch.index(epoch)
                self.args.dist_temperature = self.args.dist_temperature_decay[idx]*self.args.dist_temperature_scale
                if len(feature_list)<3:
                    feature_list.append([None for idx in range(len(self.train_data_loader))])
                else:
                    feature_list = feature_list[-2:]+[[None for idx in range(len(self.train_data_loader))]]
                #precompute all the features and record it
                for num, sample in tqdm(enumerate(self.train_data_loader)):
                    np_features = sample['data'].numpy()
                    idx = sample['idx'].numpy()
                    np_labels = sample['labels'].numpy()

                    features = torch.from_numpy(np_features).float().to(self.device)
                    labels = torch.from_numpy(np_labels).float().to(self.device)
                    f_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)
                    #compute the prediction of the distilation model by zhoujq
                    with torch.no_grad():
                        d_o_out, d_m_out, d_em_out = self.dist_model(features)
                    #compute the self distlation psuedo label
                    pred_label = utils.utils.fuse_current_pred(d_o_out, d_m_out, d_em_out, f_labels, self.args)
                    feature_list[-1][idx[0]] = pred_label


            for num, sample in enumerate(self.train_data_loader):
                if self.args.decay_type == 0:
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 1:
                    if num == 0:
                        current_lr = self.Step_decay_lr(epoch)
                        for param_group in self.optimizer_module.param_groups:
                            param_group['lr'] = current_lr
                elif self.args.decay_type == 2:
                    current_lr = self.Cosine_decay_lr(epoch, num)
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 3:
                    if num == 0:
                        current_lr = self.Step_change_lr(epoch)
                        for param_group in self.optimizer_module.param_groups:
                            param_group['lr'] = current_lr

                iter = iter + 1
                np_features = sample['data'].numpy()
                np_labels = sample['labels'].numpy()

                labels = torch.from_numpy(np_labels).float().to(self.device)
                features = torch.from_numpy(np_features).float().to(self.device)

                f_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)
                b_labels = torch.cat([labels, torch.ones(labels.size(0), 1).to(self.device)], -1)

                idx = sample['idx'].numpy()
                #compute the distilation loss by zhoujq
                if epoch<=100:
                    _,L,_ = features.shape
                    pred_label = torch.zeros((1,L,21),dtype=torch.float32).to(self.device)
                    self.args.spl_loss_weight = 1.0
                    pseudolabel_rate = 0.1
                    attn_label = None
                    noise_rate = None
                elif epoch<=200:
                    _,L,_ = features.shape
                    pred_label = torch.zeros((1,L,21),dtype=torch.float32).to(self.device)
                    self.args.spl_loss_weight = 0.5
                    pseudolabel_rate = 0.1
                    attn_label = None
                    noise_rate = None
                elif epoch <= 250:
                    pred_label = feature_list[-1][idx[0]].to(self.device)
                    self.args.spl_loss_weight = 1.0
                    pseudolabel_rate = 0.1
                    attn_label = None
                    noise_rate = None
                #start using delta pseudo label
                elif epoch <=300:
                    pred_label = feature_list[-1][idx[0]].to(self.device)*2-feature_list[-2][idx[0]].to(self.device)*2
                    self.args.spl_loss_weight = 1.0
                    pseudolabel_rate = 0.1
                    attn_label = None
                    noise_rate = None
                elif epoch<=350:
                    pred_label = feature_list[-1][idx[0]].to(self.device)*2.15-feature_list[-2][idx[0]].to(self.device)*2.15
                    self.args.spl_loss_weight = 1.0
                    pseudolabel_rate = 0.1
                    attn_label = None
                    noise_rate = None

                o_out, m_out, em_out = self.model(features,
                                                  pseudolabel = attn_label,
                                                  pseudolabel_rate = pseudolabel_rate,
                                                  noise_rate = noise_rate)


                loss_dist = utils.utils.compute_distloss(o_out, m_out, em_out, pred_label,self.args)
                loss_dist_weight = min(self.args.dist_loss*(self.args.dist_loss_increase**(epoch-self.args.pretrained_epoch)),self.args.max_dist_weight)

                if self.args.use_foreloss==1:
                    vid_fore_loss = self.loss_nce(o_out[0], f_labels) + self.loss_nce(m_out[0], f_labels)
                else:
                    vid_fore_loss = torch.tensor(0).to(self.device)

                if self.args.use_backloss==1:
                    vid_back_loss = self.loss_nce(o_out[1], b_labels) + self.loss_nce(m_out[1], b_labels)
                else:
                    vid_back_loss = torch.tensor(0).to(self.device)

                if self.args.use_attloss == 1:
                    vid_att_loss = self.loss_att(o_out[2])
                else:
                    vid_att_loss = torch.tensor(0).to(self.device)

                if epoch > self.args.warmup_epoch and self.args.use_mem==1:
                    idxs = np.where(np_labels==1)[0].tolist()
                    cls_mu = self.memory._return_queue(idxs).detach()
                    reallocated_x = random_walk(em_out[0], cls_mu, self.args.w)
                    r_vid_ca_pred, r_vid_cw_pred, r_frm_fore_att, r_frm_pred = self.model.PredictionModule(reallocated_x)

                    vid_fore_loss += 0.5 * self.loss_nce(r_vid_ca_pred, f_labels)
                    vid_back_loss += 0.5 * self.loss_nce(r_vid_cw_pred, b_labels)
                    vid_spl_loss = self.loss_spl(o_out[3], r_frm_pred * 0.2 + m_out[3] * 0.8)

                    self.memory._update_queue(em_out[1].squeeze(0), em_out[2].squeeze(0), idxs)
                else:
                    vid_spl_loss = self.loss_spl(o_out[3], m_out[3])

                #compute the constrained cluster loss
                total_loss = vid_fore_loss*self.args.fore_loss_weight + vid_back_loss * self.args.back_loss_weight \
                + vid_att_loss * self.args.att_loss_weight + vid_spl_loss * self.args.spl_loss_weight \
                + loss_dist_weight*loss_dist

                loss_recorder['cls_fore'] += vid_fore_loss.item()
                loss_recorder['cls_back'] += vid_back_loss.item()
                loss_recorder['att'] += vid_att_loss.item()
                loss_recorder['spl'] += vid_spl_loss.item()
                loss_recorder['KD'] += loss_dist.item()
                # loss_recorder['SVD'] += loss_predictor.item()
                total_loss.backward()

                if iter % self.args.batch_size == 0:
                    step += 1
                    print('Epoch: {}/{}, Iter: {:02d}, Lr: {:.6f}'.format(
                        epoch + 1,
                        self.args.max_epoch,
                        step,
                        current_lr), end=' ')
                    for k, v in loss_recorder.items():
                        print('Loss_{}: {:.4f}'.format(k, v / self.args.batch_size), end=' ')
                        loss_recorder[k] = 0

                    print()
                    self.optimizer_module.step()
                    self.optimizer_module.zero_grad()

            if epoch == self.args.warmup_epoch:
                self.model_module.eval()
                mu_queue, sc_queue, lbl_queue = ft_eval(self.train_data_loader_tmp, self.model_module, self.args, self.device)
                self.memory._init_queue(mu_queue, sc_queue, lbl_queue)
                self.model_module.train()


            if (epoch + 1) % self.args.save_interval == 0:
                out_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                    epoch + 1) + '.pkl'
                torch.save(self.model_module.state_dict(), out_dir)
                self.model_module.eval()
                ss_eval(epoch + 1, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
                self.model_module.train()




    def val(self, epoch):
        print('Start testing!')
        self.model_module.eval()
        ss_eval(epoch, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
        print('Finish testing!')

    def Step_change_lr(self, epoch):
        current_epoch = epoch + 1

        lr_range = self.args.changeLR_list.copy()
        lr_range.insert(0, 0)
        lr_range.append(self.args.max_epoch + 1)

        if len(self.args.changeLR_list) != 0:
            for i in range(0, len(lr_range) - 1):
                if lr_range[i + 1] >= current_epoch > lr_range[i]:
                    lr_step = i
                    break

        current_lr = self.args.changeLR_value[lr_step]
        return current_lr

    def Step_decay_lr(self, epoch):
        lr_list = []
        current_epoch = epoch + 1
        for i in range(0, len(self.args.changeLR_list) + 1):
            lr_list.append(self.args.lr * ((self.args.lr_decay) ** i))

        lr_range = self.args.changeLR_list.copy()
        lr_range.insert(0, 0)
        lr_range.append(self.args.max_epoch + 1)

        if len(self.args.changeLR_list) != 0:
            for i in range(0, len(lr_range) - 1):
                if lr_range[i + 1] >= current_epoch > lr_range[i]:
                    lr_step = i
                    break

        current_lr = lr_list[lr_step]
        return current_lr

    def Cosine_decay_lr(self, epoch, batch):
        if self.args.warmup:
            max_epoch = self.args.max_epoch - self.args.warmup_epoch
            current_epoch = epoch + 1 - self.args.warmup_epoch
        else:
            max_epoch = self.args.max_epoch
            current_epoch = epoch + 1

        current_lr = 1 / 2.0 * (1.0 + np.cos(
            (current_epoch * self.args.batch_num + batch) / (max_epoch * self.args.batch_num) * np.pi)) * self.args.lr

        return current_lr
