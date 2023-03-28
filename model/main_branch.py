import random

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f

def random_walk(x, y, w):
    x_norm = calculate_l1_norm(x)
    y_norm = calculate_l1_norm(y)
    eye_x = torch.eye(x.size(1)).float().to(x.device)

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])
    # mat_inv_x = torch.linalg.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
    mat_inv_x = torch.inverse(eye_x - (w ** 2) * affinity_mat)
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd', [latent_z, y]) + x
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])

    return refined_x

class predictor_yololike(nn.Module):
    def __init__(self,sharp_factor = 1.0,base_value=1.0):
        super().__init__()
        self.Linear =nn.Linear(2048,4,bias=False)

    def forward(self, x):
        output = self.Linear(x)
        return(output)

class predictor_svd(nn.Module):
    def __init__(self,sharp_factor = 1.0,base_value=1.0):
        super().__init__()
        self.U = None
        self.s = None
        self.V = None
        self.sharp_factor = sharp_factor
        self.base_value = base_value
        self.scale = nn.Parameter(torch.ones((21,),dtype=torch.float32))

    def forward(self, x,T=0.01):
        s_rescale = self.s *self.scale
        classifier = self.U @ (s_rescale[:, None] * self.V.t()[:21, :])
        output = F.softmax(x@classifier.t()/T,dim=-1)
        return(output)

    def set_SVD(self,centers):
        U_c, S_c, V_c = torch.svd(centers, some=False, compute_uv=True)
        self.U = U_c
        self.s = S_c
        self.V = V_c
        #resclae s
        self.s = self.base_value*torch.abs(S_c/self.base_value)**self.sharp_factor
        self.s = torch.sign(S_c)*self.s

class WSTAL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature embedding
        self.w = args.w
        self.n_in = args.inp_feat_num
        self.n_out = args.out_feat_num

        self.n_mu = args.mu_num
        self.em_iter = args.em_iter
        self.n_class = args.class_num
        self.scale_factor = args.scale_factor
        self.dropout = args.dropout

        self.mu = nn.Parameter(torch.randn(self.n_mu, self.n_out))
        torch_init.xavier_uniform_(self.mu)

        self.ac_center = nn.Parameter(torch.randn(self.n_class + 1, self.n_out))
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

        self.feature_embedding = nn.Sequential(
                                    nn.Linear(self.n_in, self.n_out),
                                    nn.ReLU(inplace=True)
                                    )
        self.droplayer = nn.Dropout(self.dropout)

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init_random)

        self.use_noise = args.use_noise
        self.noise_rate = args.noise_rate
        self.noise_centers = torch.rand((2,2048))

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x)
        for _ in range(self.em_iter):
        # for _ in range(100):
            norm_mu = calculate_l1_norm(mu)
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True)+1e-9)
            new_mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
            error = (torch.abs(mu - new_mu)) / (torch.abs(mu) + 1e-9)
            mu = new_mu
        return mu,error

    def PredictionModule_old(self, x):
        # normalization
        norms_x = calculate_l1_norm(x)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * self.scale_factor

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
        class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

        ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
        cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

        # normalization
        norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
        norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

        # classification
        frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
        ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * self.scale_factor
        cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * self.scale_factor

        # prediction
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

    def PredictionModule(self, x,pertube_factor=-0.1,pertube_offset=1.0,
                         pseudolabel=None,pseudolabel_rate=0.1):


        # normalization
        norms_x = calculate_l1_norm(x)
        #pertube the predictor
        if pertube_factor>0.:
            noise = (torch.rand(self.ac_center.data.shape)*2-pertube_offset)*pertube_factor
            noise = noise.to(norms_x.device)
            ac_center_noise = self.ac_center + noise * self.ac_center.std()
            norms_ac = calculate_l1_norm(ac_center_noise)
        else:
            norms_ac = calculate_l1_norm(self.ac_center)


        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * self.scale_factor

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)

        weight_scr_base = torch.sigmoid(frm_scr)
        weight_fg_scr_base = torch.sigmoid(frm_fb_scrs)


        #2nd order approximation

        weight_scr = weight_scr_base
        weight_fg_scr = weight_fg_scr_base
        if pseudolabel is not None:
            weight_scr = weight_scr*(1-pseudolabel_rate)+pseudolabel_rate*pseudolabel.to(torch.float32)[None,:,:]
        weight_scr = weight_scr/(torch.sum(weight_scr,dim=1,keepdim=True)+1e-5)
        weight_fg_scr = weight_fg_scr / (torch.sum(weight_fg_scr, dim=1, keepdim=True) + 1e-5)

        ca_vid_feat = torch.einsum('ntd,nt->nd', [x, weight_fg_scr])
        cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, weight_scr])
        norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)


        norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)
        ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac])* self.scale_factor
        cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac])* self.scale_factor


        # prediction
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

    def forward(self, x,pseudolabel=None,use_drop = True,noise_rate=None,
                pertube_factor = -0.1,pertube_offset=1.0,pseudolabel_rate=0.2):
        n, t, _ = x.size()

        # feature embedding
        x = self.feature_embedding(x)


        if use_drop:
            x = self.droplayer(x)
        if self.use_noise==2:
            #infer some noise
            _,N,d = x.shape
            noise_centers = self.noise_centers.to(x.device)
            noise_centers = x@noise_centers.t()
            noise_centers = noise_centers.repeat(1,1,d//2)
            noise = self.droplayer(noise_centers)
            if noise_rate is not None:
                x = x+ noise*noise_rate
            else:
                x = x + noise * self.noise_rate

        #centering the fearure
        # Expectation Maximization of class agnostic tokens
        mu = self.mu[None, ...].repeat(n, 1, 1)
        mu, error = self.EM(mu, x)
        # feature reallocate
        reallocated_x = random_walk(x, mu, self.w)
        reallocated_x_drop = self.droplayer(reallocated_x)

        # original feature branch
        o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred = self.PredictionModule(x,pertube_factor=pertube_factor,
                                                                                pertube_offset=pertube_offset,
                                                                                pseudolabel=pseudolabel,
                                                                                pseudolabel_rate = pseudolabel_rate)

        # reallocated feature branch
        m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred = self.PredictionModule(reallocated_x,pertube_factor=pertube_factor,
                                                                                pertube_offset=pertube_offset,
                                                                                pseudolabel=pseudolabel,
                                                                                pseudolabel_rate = pseudolabel_rate)

        # mu classification scores
        norms_mu = calculate_l1_norm(mu)
        norms_ac = calculate_l1_norm(self.ac_center)
        mu_scr = torch.einsum('nkd,cd->nkc', [norms_mu, norms_ac]) * self.scale_factor
        mu_pred = F.softmax(mu_scr, -1)

        return [o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred],\
               [m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred],\
               [x, mu, mu_pred, reallocated_x,reallocated_x_drop]