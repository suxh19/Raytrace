#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用静态CT几何配置进行射线追踪演示
128个源点 + 768个探测器像素
使用向量化操作，避免循环
使用PyTorch代替NumPy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from raytrace import raytrace
from skimage import data
from scipy.ndimage import zoom

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载静态CT几何配置（先用numpy加载，再转为torch）
print("\n正在加载静态CT几何配置...")
sources = torch.from_numpy(np.load('staticCTGeometry/source_positions.npy')).float().to(device)
detectors = torch.from_numpy(np.load('staticCTGeometry/detector_positions.npy')).float().to(device)

print(f"源点数量: {sources.shape[0]}")
print(f"探测器像素数量: {detectors.shape[0]}")
print(f"总投影数量: {sources.shape[0] * detectors.shape[0]}")

# 生成 Shepp-Logan 模体
print("\n正在生成 Shepp-Logan 模体...")
phantom_2d = data.shepp_logan_phantom()
phantom_size = phantom_2d.shape[0]

# 体积配置
vol_size = (1, 512, 512)  # (D, H, W)
vol_spacing_DHW = (10, 0.5, 0.5)  # (D, H, W) 顺序
vol_center = torch.tensor([0.0, 0.0, 0.0], device=device)

# 计算 vol_start（按 x, y, z 顺序）
vol_start = vol_center - torch.tensor([
    (vol_size[2] - 1) * vol_spacing_DHW[2] / 2,  # x (W方向)
    (vol_size[1] - 1) * vol_spacing_DHW[1] / 2,  # y (H方向)
    (vol_size[0] - 1) * vol_spacing_DHW[0] / 2,  # z (D方向)
], device=device)

# raytrace 函数需要 (W, H, D) 顺序的 spacing
vol_spacing = (vol_spacing_DHW[2], vol_spacing_DHW[1], vol_spacing_DHW[0])

# 创建3D体积（先用numpy处理图像，再转为torch）
vol_np = np.zeros(vol_size, dtype=np.float32)
scale_factor = 512 / phantom_size
phantom_resized = zoom(phantom_2d, scale_factor, order=3)
for z in range(vol_size[0]):
    vol_np[z, :, :] = phantom_resized

# 转换为torch tensor
vol = torch.from_numpy(vol_np).float().to(device)

# 转换为3D坐标
sources_3d = torch.zeros((sources.shape[0], 3), device=device)
sources_3d[:, 0] = sources[:, 0]
sources_3d[:, 1] = sources[:, 1]
sources_3d[:, 2] = 0.0

detectors_3d = torch.zeros((detectors.shape[0], 3), device=device)
detectors_3d[:, 0] = detectors[:, 0]
detectors_3d[:, 1] = detectors[:, 1]
detectors_3d[:, 2] = 0.0

print("\n使用向量化方法生成所有射线，避免循环")
print("正在生成所有 128×768=98304 条射线的源点和目标点...")

# 使用torch的repeat_interleave和repeat
total_rays = sources.shape[0] * detectors.shape[0]

# 生成所有源点：每个源点重复768次
ray_sources = sources_3d.repeat_interleave(detectors.shape[0], dim=0)

# 生成所有目标点：所有探测器重复128次
ray_dests = detectors_3d.repeat(sources.shape[0], 1)

print(f"生成的射线数量: {total_rays}")
print(f"ray_sources 形状: {ray_sources.shape}")
print(f"ray_dests 形状: {ray_dests.shape}")

# 验证射线配对是否正确
print("\n验证前几条射线的配对:")
print("射线 0-7: 源点 0 -> 探测器 0-7")
for i in range(8):
    src = ray_sources[i][:2].cpu().numpy()
    dst = ray_dests[i][:2].cpu().numpy()
    print(f"  射线 {i}: 源点 {src} -> 探测器 {dst}")

print("射线 768-775: 源点 1 -> 探测器 0-7")
for i in range(768, 776):
    src_idx = i // detectors.shape[0]
    det_idx = i % detectors.shape[0]
    src = ray_sources[i][:2].cpu().numpy()
    dst = ray_dests[i][:2].cpu().numpy()
    print(f"  射线 {i}: 源点 {src} -> 探测器 {dst}")

# 执行射线追踪
print("\n正在计算完整投影矩阵...")

# 一次性计算所有射线（raytrace现在支持torch输入输出）
rpl_flat = raytrace(ray_sources, ray_dests, vol, vol_start, vol_spacing)

# 重塑为 128×768 的投影矩阵
projections = rpl_flat.reshape((sources.shape[0], detectors.shape[0]))

print(f"完成! 投影矩阵形状: {projections.shape}")
print(f"投影值范围: [{projections.min().item():.2f}, {projections.max().item():.2f}]")

# 保存投影数据（转换为numpy保存）
projections_np = projections.cpu().numpy()
np.save('test/full_projections.npy', projections_np)
print(f"完整投影矩阵已保存到: test/full_projections.npy")

# 显示投影矩阵
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
im = ax.imshow(projections_np, aspect='auto', cmap='gray', origin='lower')
ax.set_xlabel('Detector Pixel Index')
ax.set_ylabel('Source Index')
ax.set_title('CT Projection Matrix (128x768) - PyTorch Version')
plt.colorbar(im, ax=ax)
plt.savefig('test/static_ct_raytrace_torch.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n总结:")
print("✓ 使用 PyTorch 代替 NumPy")
print("✓ 支持 GPU 加速（如果可用）")
print("✓ 使用向量化操作避免 Python 循环")
print("✓ 一次性生成所有 98304 条射线")
print("✓ 高效计算完整投影矩阵")
print("✓ 内存使用优化")
print(f"✓ 使用设备: {device}")
