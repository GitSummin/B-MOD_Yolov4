import os
import numpy as np

# OMP 에러 해결을 위한 환경 변수 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    center1 = pred[:, :2]
    center2 = target[:, :2]

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2]  + eps
    h1 = pred[:, 3]  + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


def bcd_loss(pred, target, tau=0.5):
    eps = 1e-7

    mu_p = pred[:, :2]
    mu_t = target[:, :2]

    # Directly construct the squared widths and heights as diagonal elements
    sigma_p_diag = pred[:, 2:] ** 2
    sigma_t_diag = target[:, 2:] ** 2

    delta = mu_p - mu_t
    sigma_diag = 0.5 * (sigma_p_diag + sigma_t_diag)

    # Calculate determinant and inverse for 2x2 matrices analytically
    det_sigma_p = sigma_p_diag[:, 0] * sigma_p_diag[:, 1] + eps
    det_sigma_t = sigma_t_diag[:, 0] * sigma_t_diag[:, 1] + eps
    det_sigma = sigma_diag[:, 0] * sigma_diag[:, 1] + eps

    # Inverse of a diagonal matrix is simply the inverse of each diagonal element
    sigma_inv_diag = 1.0 / sigma_diag + eps

    # Compute terms analytically
    # term1 = torch.log(det_sigma / torch.sqrt(det_sigma_p * det_sigma_t) + eps)  # 수정된 부분
    term1 = torch.log((det_sigma / torch.sqrt(det_sigma_p * det_sigma_t)) + eps)
    # term1 = torch.log(det_sigma / torch.sqrt(det_sigma_p * det_sigma_t)).unsqueeze(-1) + eps
    term2 = (delta ** 2 * sigma_inv_diag).sum(dim=1) + eps
        

    dis = 0.5 * term1 + 0.125 * term2
    bcd_dis = dis.clamp(min=0)

    # loss = 1 - 1 / (tau + torch.log1p(bcd_dis))
    loss = torch.log1p(bcd_dis) / (tau + torch.log1p(bcd_dis))


    return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    #print(device)
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            
            
            ### Setting IoU ###
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False).squeeze()  # iou(prediction, target)            
            lbox += (1.0 - iou).mean() # iou_loss
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
            ### ----------- ###
            '''
            ### Setting NWD ###
            nwd = wasserstein_loss(pbox, tbox[i]).squeeze()
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False).squeeze()  # iou(prediction, target)
            lbox += (1 - model.gr) * (1.0 - nwd).mean() + model.gr * (1.0 - iou).mean()  # nwd loss
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
            ### ----------- ###
            '''
            '''
            ### Setting BCD ###
            bcd_loss_value = bcd_loss(pbox, tbox[i], tau=0.5) 

            # bcd_loss 결과가 텐서가 아닐 경우 텐서로 변환
            if not isinstance(bcd_loss_value, torch.Tensor):
                bcd_loss_value = torch.tensor(bcd_loss_value, dtype=torch.float32, device=device)

            # bcd_loss 결과 텐서에 squeeze() 적용
            bcd_loss_value = bcd_loss_value.squeeze()

            lbox += bcd_loss_value.mean() # bcd_loss

            # BCD 손실을 사용해 객체 탐지 점수 계산 및 조정
            objectness_score = 1 / torch.exp(bcd_loss_value)
            objectness_score = objectness_score.float()
            tobj = tobj.float()
            tobj[b, a, gj, gi] = objectness_score
            ### ----------- ###
            '''

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i].clamp(0, model.nc - 1)] = cp  # Ensuring indices are valid
                lcls += BCEcls(ps[:, 5:], t)  # BCE 

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()



def build_targets(p, targets, model):
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    g = 0.5  # offset
    multi_gpu = is_parallel(model)
    for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            na = anchors.shape[0]  # number of anchors
            # at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
            at = torch.arange(na, device=targets.device).view(na, 1).repeat(1, nt)  # anchor tensor
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        #indices.append((b, a, gj, gi))  # image, anchor, grid indices
        # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        # indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))  # image, anchor, grid indices
        indices.append((b, a, gj.clamp(0, gain[3] - 1).long(), gi.clamp(0, gain[2] - 1).long()))
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


