from __future__ import annotations

import math
import numpy as np
import torch
from .ops import xyxy2xywh

def inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        box1, box2 = xyxy2xywh(box1), xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2

    # Inner-IoU
    inter = (inner_b1_x2.minimum(inner_b2_x2) - inner_b1_x1.maximum(inner_b2_x1)).clamp_(0) * \
            (inner_b1_y2.minimum(inner_b2_y2) - inner_b1_y1.maximum(inner_b2_y1)).clamp_(0)
    inner_union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / inner_union


class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''
    
    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True
 
    def __init__(self, iou):
        self.iou = iou
        self._update(self)
    
    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()
    
    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


class DynamicWiseIoU:
    """
    动态非单调 Wise-IoU（基于 WIoU v3 思路），配合批次动态均值实现“去极端、聚中等”聚焦。
    适用于密集小目标场景，能减弱简单样本与离群样本的影响。
    """

    iou_mean = 1.0
    momentum = 1 - 0.5 ** (1 / 7000)  # 与 WIoU_Scale 保持一致的 EMA 默认值
    eps = 1e-7

    @classmethod
    def update_mean(cls, loss_tensor: torch.Tensor) -> None:
        """更新 IoU 损失的 EMA 均值，detach 避免梯度污染。"""
        cls.iou_mean = (1 - cls.momentum) * cls.iou_mean + cls.momentum * loss_tensor.detach().mean().item()

    @staticmethod
    def _to_xywh(box, xywh: bool):
        if xywh:
            return box
        return xyxy2xywh(box)

    @classmethod
    def loss(cls, box1, box2, xywh: bool = True, gamma: float = 1.9, delta: float = 3.0,
             use_inner: bool = False, inner_ratio: float = 0.7) -> torch.Tensor:
        """
        计算动态非单调 Wise-IoU 回归项。

        Args:
            box1 (Tensor): 预测框，形状 (..., 4)。
            box2 (Tensor): 真值框，形状与 box1 相同。
            xywh (bool): 输入是否为 xywh 格式，否则视为 xyxy。
            gamma (float): 非单调聚焦系数的底数。
            delta (float): 非单调聚焦系数的偏移。
        """
        # 基础 IoU 损失
        if use_inner:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=inner_ratio)
        else:
            iou = new_bbox_iou(box1, box2, xywh=xywh, alpha=1, Focal=False, CIoU=False, DIoU=False, GIoU=False,
                               SIoU=False, EIoU=False, WIoU=False, MPDIoU=False, ShapeIou=False, PIouV1=False,
                               PIouV2=False, UIoU=False, Inner_iou=False)
        loss_iou = 1 - iou
        cls.update_mean(loss_iou)

        # 动态离群度 beta 及非单调聚焦项（与 WIoU v3 相同公式）
        beta = loss_iou.detach() / (cls.iou_mean + cls.eps)
        alpha = delta * torch.pow(gamma, beta - delta)
        focus = torch.clamp(beta / alpha, min=0.0)

        # 空间注意项 ℛ：使用包围框对角长度归一化中心距
        box1_xywh, box2_xywh = cls._to_xywh(box1, xywh), cls._to_xywh(box2, xywh)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1_xywh.chunk(4, -1), box2_xywh.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
        cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp_min(cls.eps)
        ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp_min(cls.eps)
        center_dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
        spatial_weight = torch.exp(-center_dist / (cw ** 2 + ch ** 2 + cls.eps))

        return focus * spatial_weight * loss_iou


def _xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def nwd_loss(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, C: float = 13.0, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 Normalized Gaussian Wasserstein Distance (NWD) 损失。
    
    NWD 将边界框建模为二维高斯分布，通过计算 Wasserstein 距离来衡量相似度。
    相比传统 IoU，NWD 对小目标更友好，在重叠极小的情况下仍能提供有效梯度。
    
    Args:
        box1 (Tensor): 预测框，形状 (..., 4)
        box2 (Tensor): 真值框，形状与 box1 相同
        xywh (bool): 输入是否为 xywh 格式，否则视为 xyxy
        C (float): 归一化常数，与数据集相关，常用值：9,11,13,15,17
        eps (float): 数值稳定项
        
    Returns:
        Tensor: NWD 值，范围 [0,1]，值越大表示越相似
        
    References:
        NWD: A Normalized Gaussian Wasserstein Distance for Small Object Detection
    """
    # 确保输入为 xywh 格式
    if not xywh:
        box1 = xyxy2xywh(box1)
        box2 = xyxy2xywh(box2)
    
    # 提取坐标 (cx, cy, w, h)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    
    # 计算高斯分布的均值向量差 ||m1 - m2||_2^2
    # 按照论文公式：使用 (cx, cy, w/2, h/2) 作为高斯分布参数
    mean_diff_x = x1 - x2
    mean_diff_y = y1 - y2
    mean_diff_w = w1 / 2 - w2 / 2  
    mean_diff_h = h1 / 2 - h2 / 2
    
    # 计算 Wasserstein-2 距离的平方
    # W_2^2 = ||(cx_a, cy_a, w_a/2, h_a/2) - (cx_b, cy_b, w_b/2, h_b/2)||_2^2
    wasserstein_2_squared = (mean_diff_x ** 2 + mean_diff_y ** 2 + 
                           mean_diff_w ** 2 + mean_diff_h ** 2)
    
    # 归一化 Wasserstein 距离：NWD = exp(-sqrt(W_2^2)/C)
    wasserstein_2 = torch.sqrt(wasserstein_2_squared + eps)
    nwd = torch.exp(-wasserstein_2 / C)
    
    return nwd


def OBC(pred_box: torch.Tensor,
        gt_box: torch.Tensor,
        neighbor_boxes: torch.Tensor | None,
        xywh: bool = True,
        lambda_iog: float = 1.0,
        mu: float = 0.5,
        sigma: float = 0.1,
        eps: float = 1e-7) -> torch.Tensor:
    """
    遮挡边界修正损失（OBC）。对预测框侵占邻居真值的重叠与过近中心距离进行惩罚。

    Args:
        pred_box (Tensor): 预测框，形状 (N, 4) 或 (…, 4)。
        gt_box (Tensor): 对应目标框，形状与 pred_box 相同，用于对齐 batch 维度。
        neighbor_boxes (Tensor | None): 非目标邻居框，形状 (N, K, 4)。若无邻居可传 None。
        xywh (bool): 输入格式是否为 xywh。
        lambda_iog (float): IoG 惩罚权重。
        mu (float): 中心距排斥权重。
        sigma (float): 排斥安全距离阈值（单位与坐标一致，可按特征图/归一化尺度设置）。
        eps (float): 数值稳定项。
    """
    if neighbor_boxes is None or neighbor_boxes.numel() == 0:
        return pred_box.new_tensor(0.0)

    # 对齐形状与格式
    if neighbor_boxes.dim() == 2:
        neighbor_boxes = neighbor_boxes.unsqueeze(0)
    if not xywh:
        pred_box = xyxy2xywh(pred_box)
        gt_box = xyxy2xywh(gt_box)
        neighbor_boxes = xyxy2xywh(neighbor_boxes)

    pred_xyxy = _xywh_to_xyxy(pred_box)
    neigh_xyxy = _xywh_to_xyxy(neighbor_boxes)

    # IoG 惩罚项：对每个预测匹配邻居最大 IoG
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    nx1, ny1, nx2, ny2 = neigh_xyxy.unbind(-1)
    inter_w = (px2[..., None] - px1[..., None]).minimum(nx2) - (px1[..., None]).maximum(nx1)
    inter_h = (py2[..., None] - py1[..., None]).minimum(ny2) - (py1[..., None]).maximum(ny1)
    inter = inter_w.clamp_min(0) * inter_h.clamp_min(0)
    neigh_area = (nx2 - nx1).clamp_min(eps) * (ny2 - ny1).clamp_min(eps)
    iog = inter / (neigh_area + eps)
    max_iog, max_idx = iog.max(dim=-1)

    # 中心距离排斥：仅对与最大 IoG 的邻居计算
    pred_xywh = pred_box
    neigh_xywh = neighbor_boxes.gather(-2, max_idx.unsqueeze(-1).repeat(1, 1, 4)).squeeze(-2)
    cx_p, cy_p, _, _ = pred_xywh.unbind(-1)
    cx_n, cy_n, _, _ = neigh_xywh.unbind(-1)
    center_dist = torch.sqrt((cx_p - cx_n) ** 2 + (cy_p - cy_n) ** 2 + eps)
    dist_penalty = torch.relu(sigma - center_dist)

    return lambda_iog * max_iog + mu * dist_penalty


def dynamic_wiou_obc_loss(pred_box: torch.Tensor,
                          gt_box: torch.Tensor,
                          neighbor_boxes: torch.Tensor | None,
                          dfl_loss: torch.Tensor | float = 0.0,
                          xywh: bool = True,
                          gamma: float = 1.9,
                          delta: float = 3.0,
                          lambda_iog: float = 1.0,
                          mu: float = 0.5,
                          sigma: float = 0.1,
                          use_inner: bool = False,
                          inner_ratio: float = 0.7) -> torch.Tensor:

    """
    L_total = L_D-WIoU + L_DFL + L_OBC
    """
    wise_loss = DynamicWiseIoU.loss(pred_box, gt_box, xywh=xywh, gamma=gamma, delta=delta,
                                    use_inner=use_inner, inner_ratio=inner_ratio)
    obc_loss = OBC(pred_box, gt_box, neighbor_boxes, xywh=xywh,

                   lambda_iog=lambda_iog, mu=mu, sigma=sigma)
    dfl_loss = torch.as_tensor(dfl_loss, device=wise_loss.device, dtype=wise_loss.dtype)
    return wise_loss + dfl_loss + obc_loss

def new_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIoU=False, WIoU=False,
                  MPDIoU=False, ShapeIou=False, PIouV1=False, PIouV2=False, UIoU=False, Inner_iou=False, NWD=False,
                  Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7,
                  feat_w=640, feat_h=640, ratio=0.7, ShapeIou_scale=0, PIou_Lambda=1.3, epoch=600,
                  WIoU_OBC=False, neighbor_boxes=None, dfl_loss=0.0, w_gamma=1.9, w_delta=3.0,
                  obc_lambda=1.0, obc_mu=0.5, obc_sigma=0.1, nwd_C=13.0):
    """
    计算bboxes iou
    Args: 
        box1: predict bboxes
        box2: target bboxes
        xywh: 将bboxes转换为xyxy的形式
        GIoU: 为True时计算GIoU LOSS (yolov8自带)
        DIoU: 为True时计算DIoU LOSS (yolov8自带)
        CIoU: 为True时计算CIoU LOSS (yolov8自带,默认使用)
        SIoU: 为True时计算SIoU LOSS (新增)
        EIoU: 为True时计算EIoU LOSS (新增)
        WIoU: 为True时计算WIoU LOSS (新增)
        MPDIoU: 为True时计算MPDIoU LOSS (新增)
        ShapeIou: 为True时计算ShapeIou LOSS (新增)
        PIouV1/V2: 为True时计算Powerful-IoU LOSS (新增)
        UIoU: 为True时计算Unified-IoU LOSS (新增)
        Inner_iou: 为True时计算InnerIou LOSS (新增)
        NWD: 为True时计算Normalized Gaussian Wasserstein Distance LOSS (新增)
        Focal: 对IOU损失乘以系数=IOU**gamma,以使回归过程专注于高质量锚框,参考Focal-EIoU Loss
        alpha: AlphaIoU中的alpha参数,默认为1,为1时则为普通的IoU,如果想采用AlphaIoU,论文alpha默认值为3,此时设置CIoU=True则为AlphaCIoU
        gamma: Focal-EIoU中指数系数
        scale: scale为True时,WIoU会乘以一个系数
        eps: 防止除0
        feat_w/h: 特征图大小
        ratio: Inner-IoU对应的是尺度因子,通常取范围为[0.5,1.5],原文中VOC数据集对应的Inner-CIoU和Inner-SIoU设置在[0.7,0.8]之间有较大提升，
        数据集中大目标多则设置<1,小目标多设置>1
        ShapeIou_scale: 为ShapeIou的缩放因子,与数据集中目标的大小相关
        PIou_Lambda: 为Powerful-IoU的超参数
        epoch: 为Unified-IoU的超参数,训练轮数
        nwd_C: NWD归一化常数,与数据集相关,常用值:9,11,13,15,17
    Returns:
        iou
    """
    if WIoU_OBC:
        return dynamic_wiou_obc_loss(box1, box2, neighbor_boxes=neighbor_boxes, dfl_loss=dfl_loss, xywh=xywh,
                                     gamma=w_gamma, delta=w_delta, lambda_iog=obc_lambda, mu=obc_mu, sigma=obc_sigma,
                                     use_inner=Inner_iou, inner_ratio=ratio)

    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
 
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    
    if UIoU:
        # Unified-IoU https://arxiv.org/pdf/2408.06636
        # define the center point for scaling
        bb1_xc = x1
        bb1_yc = y1
        bb2_xc = x2
        bb2_yc = y2
        # attenuation mode of hyperparameter "u_ratio"[原链接为ratio]
        linear = True
        cosine = False
        fraction = False 
        # assuming that the total training epochs are 300, the "u_ratio" changes from 2 to 0.5
        if linear:
            u_ratio = -0.005 * epoch + 2
        elif cosine:
            u_ratio = 0.75 * math.cos(math.pi * epoch / 300) + 1.25
        elif fraction:
            u_ratio = 200 / (epoch + 100)
        else:
            u_ratio = 0.5
        ww1, hh1, ww2, hh2 = w1 * u_ratio, h1 * u_ratio, w2 * u_ratio, h2 * u_ratio
        bb1_x1, bb1_x2, bb1_y1, bb1_y2 = bb1_xc - (ww1 / 2), bb1_xc + (ww1 / 2), bb1_yc - (hh1 / 2), bb1_yc + (hh1 / 2)
        bb2_x1, bb2_x2, bb2_y1, bb2_y2 = bb2_xc - (ww2 / 2), bb2_xc + (ww2 / 2), bb2_yc - (hh2 / 2), bb2_yc + (hh2 / 2)
        # assign the value back to facilitate subsequent calls
        w1, h1, w2, h2 = ww1, hh1, ww2, hh2
        b1_x1, b1_x2, b1_y1, b1_y2 = bb1_x1, bb1_x2, bb1_y1, bb1_y2
        b2_x1, b2_x2, b2_y1, b2_y2 = bb2_x1, bb2_x2, bb2_y1, bb2_y2
        CIoU = True  
        
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
 
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        self = WIoU_Scale(1 - (inter / union))
 
    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter/(union + eps), alpha) # alpha iou https://arxiv.org/abs/2110.13675
    
    # Normalized Gaussian Wasserstein Distance
    if NWD:
        nwd_value = nwd_loss(box1, box2, xywh=xywh, C=nwd_C, eps=eps)
        if Focal:
            # NWD 损失：1 - NWD (因为 NWD 值越大表示越相似，损失应该越小)
            return 1 - nwd_value, torch.pow(inter/(union + eps), gamma)  # Focal_NWD
        else:
            return 1 - nwd_value  # NWD Loss
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha) # SIou
            elif WIoU and alpha == 1:
                if Inner_iou:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((rho2 / c2)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2)) # WIoU v1   
            
            if Inner_iou and alpha == 1:
                iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
            if Focal:
                return iou - rho2 / c2, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:
                return iou - rho2 / c2  # DIoU
        
        c_area = cw * ch + eps  # convex area
        if Inner_iou and alpha == 1:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    
    elif MPDIoU and alpha == 1:
        # MPDIoU https://arxiv.org/pdf/2307.07662v1
        sq_sum = (feat_w ** 2) + (feat_h ** 2)  # 对应输入image的宽高
        d12 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d22 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        if Inner_iou:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            raise RuntimeError("MPDIoU do not support Focal.")
        return iou - (d12 / sq_sum) - (d22 / sq_sum)
    
    elif ShapeIou and alpha == 1:
        # ShapeIou https://arxiv.org/pdf/2312.17663
        ww = 2 * torch.pow(w2, ShapeIou_scale) / (torch.pow(w2, ShapeIou_scale) + torch.pow(h2, ShapeIou_scale))
        hh = 2 * torch.pow(h2, ShapeIou_scale) / (torch.pow(w2, ShapeIou_scale) + torch.pow(h2, ShapeIou_scale))
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2
    
        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        if Inner_iou:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            raise RuntimeError("ShapeIou do not support Focal.")
        return iou - distance - 0.5 * ( shape_cost)

    elif (PIouV1 or PIouV2) and alpha == 1:
        # Powerful-IoU https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640
        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        L_v1 = 1 - iou - torch.exp(-P ** 2) + 1

        if Focal:
            raise RuntimeError("PIou do not support Focal.")
        if PIouV1:
            return L_v1
        if PIouV2:
            q = torch.exp(-P)
            x = q * PIou_Lambda
            return 3 * x * torch.exp(-x ** 2) * L_v1
            
    
    if Inner_iou and alpha == 1:
        iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU

