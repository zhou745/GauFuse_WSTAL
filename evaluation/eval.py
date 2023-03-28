import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.utils as utils
from .classificationMAP import getClassificationMAP as cmAP
from .detectionMAP import getDetectionMAP as dtmAP


def ft_eval(dataloader, model, args, device):
    mu_num = args.mu_num
    class_num = args.class_num
    out_feat_num = args.out_feat_num

    mu_ft_lst = []
    mu_sc_lst = []
    lbl_lst = []

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Testing test data point %d of %d' % (num + 1, len(dataloader)))
        if isinstance(sample, list):
            label = sample[1].numpy()
            features = sample[0].numpy()
        else:
            features = sample['data'].numpy()
            label = sample['labels'].numpy()

        features = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            _, _, em_out = model(Variable(features))
            mu = em_out[1]
            mu_sc = em_out[2]

        mu_ft_lst.append(mu.squeeze(0))
        mu_sc_lst.append(mu_sc.squeeze(0))
        lbl_lst.append(np.squeeze(label, 0))

    return mu_ft_lst, mu_sc_lst, lbl_lst


def ss_eval(epoch, dataloader, args, logger, model, device):
    vid_preds = []
    frm_preds = []
    vid_lens = []
    labels = []
    # model.eval()
    # smoth filter

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Testing test data point %d of %d' % (num + 1, len(dataloader)))
        if isinstance(sample, list):
            features = sample[1].numpy()
            label = sample[2].numpy()
            vid_len = sample[3].numpy()
        elif isinstance(sample, dict):
            features = sample['data'].numpy()
            label = sample['labels'].numpy()
            vid_len = sample['vid_len'].numpy()

        features = torch.from_numpy(features).float().to(device)
        cls_atts = []

        with torch.no_grad():
            o_out, m_out, mu_out = model(Variable(features))
            vid_pred = o_out[0] * 0.6 + m_out[0] * 0.4
            # sharpen the prediction value
            frm_pred_ori = F.softmax(o_out[3], -1) * args.frm_coef + F.softmax(m_out[3], -1) * (1 - args.frm_coef)
            # converted predictions
            # frm_pred = utils.compute_weighted_prediction(vid_pred[0].detach().cpu().numpy(),
            #                                              frm_pred_ori[0].detach().cpu().numpy(),args).to(frm_pred_ori.device)
            frm_pred = frm_pred_ori
            vid_att = o_out[2]

            frm_pred = frm_pred * vid_att[..., None]
            vid_pred = np.squeeze(vid_pred.cpu().data.numpy(), axis=0)
            frm_pred = np.squeeze(frm_pred.cpu().data.numpy(), axis=0)
            label = np.squeeze(label, axis=0)

        vid_preds.append(vid_pred)
        frm_preds.append(frm_pred)
        vid_lens.append(vid_len)
        labels.append(label)

    vid_preds = np.array(vid_preds)
    frm_preds = np.array(frm_preds)
    vid_lens = np.array(vid_lens)
    labels = np.array(labels)

    cmap = cmAP(vid_preds, labels)
    dmap, iou = dtmAP(vid_preds, frm_preds, vid_lens, dataloader.dataset.path_to_annotations, args)

    sum = 0
    count = 0
    print('Classification map %f' % cmap)
    for item in list(zip(iou, dmap)):
        print('Detection map @ %f = %f' % (item[0], item[1]))
        sum = sum + item[1]
        count += 1

    # logger.log_value('Test Classification mAP', cmap, epoch)
    logger.add_scalar('mAP', cmap, epoch)
    for item in list(zip(dmap, iou)):
        # logger.log_value('Test Detection1 mAP @ IoU = ' + str(item[1]), item[0], epoch)
        logger.add_scalar('mAP@' + str(item[1]), item[0], epoch)

    print('average map = %f' % (sum / count))
    logger.add_scalar('mAP@0.1:0.7', (sum / count), epoch)

    utils.write_results_to_file(args, dmap, sum / count, cmap, epoch)
