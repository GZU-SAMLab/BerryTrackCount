"""
计算与YOLO对齐的Precision和Recall指标
在最大F1点计算Precision和Recall，与YOLO11的计算方式一致
"""

import numpy as np
import torch
from typing import Tuple, Optional


def smooth(y: np.ndarray, f: float = 0.1) -> np.ndarray:
    """
    Box filter of fraction f (与YOLO的smooth函数对齐)
    
    Args:
        y: 输入数组
        f: 平滑因子
        
    Returns:
        平滑后的数组
    """
    if len(y) <= 1:
        return y
    nf = max(1, round(len(y) * f * 2) // 2 + 1)
    if nf % 2 == 0:
        nf += 1
    p = np.ones(nf // 2) * y[0]
    yp = np.concatenate((p, y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_yolo_aligned_metrics(coco_eval, iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    从COCO评估器中计算与YOLO对齐的Precision和Recall
    
    Args:
        coco_eval: COCO评估器对象（pycocotools.cocoeval.COCOeval）
        iou_threshold: IoU阈值，0.5表示mAP@0.5
        
    Returns:
        (precision, recall): 在最大F1点计算的Precision和Recall
    """
    if not hasattr(coco_eval, 'eval') or coco_eval.eval is None:
        return 0.0, 0.0
    
    try:
        precisions = coco_eval.eval['precision']
        if not isinstance(precisions, np.ndarray):
            precisions = np.array(precisions)
        
        # precision维度: [T x R x K x A x M]
        # T=IoU阈值, R=recall阈值, K=类别, A=面积范围, M=最大检测数
        # 找到IoU=0.5对应的索引（通常是第一个）
        iou_idx = 0  # IoU=0.5通常是第一个阈值
        
        # 提取IoU=0.5时的precision数据: [R x K x A x M]
        precision_50 = precisions[iou_idx, :, :, 0, -1]  # 所有面积范围，maxDets=100
        
        num_recall_thresholds = precision_50.shape[0]
        num_classes = precision_50.shape[1]
        
        # 构建recall阈值数组（从0到1）
        recall_thresholds = np.linspace(0, 1, num_recall_thresholds)
        
        # 构建置信度阈值数组（1000个点，与YOLO对齐）
        x = np.linspace(0, 1, 1000)
        
        # 为每个类别计算最大F1点对应的Precision和Recall
        precision_list = []
        recall_list = []
        eps = 1e-16
        
        for k in range(num_classes):
            # 提取该类别的precision曲线（按recall阈值）
            p_curve_raw = precision_50[:, k]  # [R]
            # 过滤无效值
            valid_mask = p_curve_raw > -1
            if not np.any(valid_mask):
                continue
            
            # 获取有效的precision和对应的recall阈值
            valid_p = p_curve_raw[valid_mask]
            valid_r = recall_thresholds[valid_mask]
            
            # 构建插值后的precision和recall曲线（按置信度阈值，与YOLO对齐）
            p_curve_interp = np.interp(x, valid_r, valid_p, left=1.0, right=0.0)
            r_curve_interp = np.interp(x, valid_r, valid_r, left=0.0, right=1.0)
            
            # 计算F1曲线（与YOLO对齐）
            f1_curve = 2 * p_curve_interp * r_curve_interp / (p_curve_interp + r_curve_interp + eps)
            
            # 平滑F1曲线（与YOLO对齐）
            if len(f1_curve) > 1:
                f1_smooth = smooth(f1_curve, 0.1)
            else:
                f1_smooth = f1_curve
            
            # 找到最大F1点（与YOLO对齐：f1_curve.mean(0).argmax()）
            max_f1_idx = np.argmax(f1_smooth)
            if max_f1_idx >= len(p_curve_interp):
                max_f1_idx = len(p_curve_interp) - 1
            
            # 提取该点的Precision和Recall（与YOLO对齐）
            precision_list.append(float(p_curve_interp[max_f1_idx]))
            recall_list.append(float(r_curve_interp[max_f1_idx]))
        
        # 计算平均值（与YOLO的mean_results对齐：self.p.mean()和self.r.mean()）
        precision_mean = np.mean(precision_list) if len(precision_list) > 0 else 0.0
        recall_mean = np.mean(recall_list) if len(recall_list) > 0 else 0.0
        
        return precision_mean, recall_mean
    
    except Exception as e:
        print(f"⚠️  计算YOLO对齐指标时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0


def get_model_params_flops(model, input_size: Tuple[int, int] = (640, 640)) -> Tuple[float, float]:
    """
    计算模型的参数量和FLOPs
    
    Args:
        model: PyTorch模型
        input_size: 输入图像尺寸 (height, width)
        
    Returns:
        (params_M, flops_G): 参数量（百万）和FLOPs（十亿）
    """
    # 计算参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_M = params / 1e6
    
    # 计算FLOPs
    flops_G = 0.0
    try:
        import thop
        from copy import deepcopy
        
        # DINO模型输入是标准RGB图像（3通道）
        # 尝试从模型的backbone第一层获取输入通道数，如果失败则使用3
        input_channels = 3  # 默认RGB
        try:
            # 尝试从backbone的第一层获取输入通道数
            if hasattr(model, 'backbone'):
                backbone = model.backbone
                # 查找第一个卷积层
                for name, module in backbone.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        # 第一个Conv2d层的输入通道数就是模型的输入通道数
                        input_channels = module.in_channels
                        break
            # 如果backbone不存在，尝试从模型的第一层获取
            elif hasattr(model, 'module') and hasattr(model.module, 'backbone'):
                backbone = model.module.backbone
                for name, module in backbone.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        input_channels = module.in_channels
                        break
        except:
            # 如果获取失败，使用默认值3（RGB）
            input_channels = 3
        
        # 获取设备
        p = next(model.parameters())
        device = p.device
        
        # 创建输入张量 (B, C, H, W)
        dummy_input = torch.randn(1, input_channels, input_size[0], input_size[1]).to(device)
        
        # 计算FLOPs
        flops, _ = thop.profile(deepcopy(model), inputs=(dummy_input,), verbose=False)
        flops_G = flops / 1e9 * 2  # thop返回的是MACs，需要乘以2得到FLOPs
    except ImportError:
        print("⚠️  thop未安装，无法计算FLOPs。请安装: pip install thop")
    except Exception as e:
        print(f"⚠️  计算FLOPs时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return params_M, flops_G

