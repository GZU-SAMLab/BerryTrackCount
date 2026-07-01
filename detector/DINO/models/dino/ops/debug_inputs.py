#!/usr/bin/env python3
"""
调试脚本：检查 MultiScaleDeformableAttention 的输入数据
用于诊断索引越界错误
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight):
    """检查输入数据的形状和值"""
    print("=" * 60)
    print("检查 MultiScaleDeformableAttention 输入数据")
    print("=" * 60)
    
    # 检查 value
    print(f"\n1. value shape: {value.shape}")
    print(f"   value dtype: {value.dtype}")
    print(f"   value device: {value.device}")
    print(f"   value is contiguous: {value.is_contiguous()}")
    
    batch, spatial_size, num_heads, channels = value.shape
    print(f"   batch={batch}, spatial_size={spatial_size}, num_heads={num_heads}, channels={channels}")
    
    # 检查 spatial_shapes
    print(f"\n2. spatial_shapes shape: {spatial_shapes.shape}")
    print(f"   spatial_shapes dtype: {spatial_shapes.dtype}")
    print(f"   spatial_shapes device: {spatial_shapes.device}")
    print(f"   spatial_shapes is contiguous: {spatial_shapes.is_contiguous()}")
    print(f"   spatial_shapes values:")
    spatial_shapes_cpu = spatial_shapes.cpu()
    num_levels = spatial_shapes.shape[0]
    total_spatial = 0
    for i in range(num_levels):
        h, w = spatial_shapes_cpu[i, 0].item(), spatial_shapes_cpu[i, 1].item()
        level_size = h * w
        total_spatial += level_size
        print(f"     Level {i}: H={h}, W={w}, size={level_size}")
    print(f"   Total spatial size: {total_spatial}")
    
    # 验证 spatial_size
    if total_spatial != spatial_size:
        print(f"\n   ❌ ERROR: spatial_size ({spatial_size}) != sum of level sizes ({total_spatial})")
        print(f"   This will cause index out of bounds!")
        return False
    else:
        print(f"   ✓ spatial_size matches sum of level sizes")
    
    # 检查 level_start_index
    print(f"\n3. level_start_index shape: {level_start_index.shape}")
    print(f"   level_start_index dtype: {level_start_index.dtype}")
    print(f"   level_start_index device: {level_start_index.device}")
    print(f"   level_start_index is contiguous: {level_start_index.is_contiguous()}")
    print(f"   level_start_index values:")
    level_start_index_cpu = level_start_index.cpu()
    for i in range(num_levels):
        val = level_start_index_cpu[i].item()
        print(f"     Level {i}: {val}")
    
    # 验证 level_start_index
    if level_start_index.shape[0] != num_levels:
        print(f"\n   ❌ ERROR: level_start_index size ({level_start_index.shape[0]}) != num_levels ({num_levels})")
        return False
    
    # 检查 level_start_index 是否在范围内
    max_level_start = level_start_index_cpu.max().item()
    if max_level_start >= spatial_size:
        print(f"\n   ❌ ERROR: max level_start_index ({max_level_start}) >= spatial_size ({spatial_size})")
        return False
    
    # 检查 sampling_loc
    print(f"\n4. sampling_loc shape: {sampling_loc.shape}")
    print(f"   sampling_loc dtype: {sampling_loc.dtype}")
    print(f"   sampling_loc device: {sampling_loc.device}")
    print(f"   sampling_loc is contiguous: {sampling_loc.is_contiguous()}")
    expected_shape = (batch, -1, num_heads, num_levels, -1, 2)
    print(f"   Expected shape pattern: {expected_shape}")
    
    # 检查 attn_weight
    print(f"\n5. attn_weight shape: {attn_weight.shape}")
    print(f"   attn_weight dtype: {attn_weight.dtype}")
    print(f"   attn_weight device: {attn_weight.device}")
    print(f"   attn_weight is contiguous: {attn_weight.is_contiguous()}")
    
    print("\n" + "=" * 60)
    print("所有检查通过 ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # 示例：创建测试数据
    print("创建测试数据...")
    batch = 2
    num_levels = 4
    num_query = 900
    num_heads = 8
    num_point = 4
    channels = 256
    
    # 创建 spatial_shapes (示例：4个尺度)
    spatial_shapes = torch.tensor([
        [80, 80],   # Level 0
        [40, 40],   # Level 1
        [20, 20],   # Level 2
        [10, 10],   # Level 3
    ], dtype=torch.long).cuda()
    
    # 计算 spatial_size
    total_spatial = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()
    
    # 创建 level_start_index
    level_start_index = torch.cat((
        spatial_shapes.new_zeros((1,)),
        (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
    )).long()
    
    # 创建 value
    value = torch.randn(batch, total_spatial, num_heads, channels).cuda()
    
    # 创建 sampling_loc
    sampling_loc = torch.randn(batch, num_query, num_heads, num_levels, num_point, 2).cuda()
    
    # 创建 attn_weight
    attn_weight = torch.randn(batch, num_query, num_heads, num_levels, num_point).cuda()
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # 检查输入
    if check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight):
        print("\n尝试导入并测试...")
        try:
            from functions.ms_deform_attn_func import MSDeformAttnFunction
            print("✓ 成功导入 MSDeformAttnFunction")
            
            im2col_step = 2
            output = MSDeformAttnFunction.apply(
                value, spatial_shapes, level_start_index, 
                sampling_loc, attn_weight, im2col_step
            )
            print(f"✓ 前向传播成功，输出形状: {output.shape}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

