import logging
import numpy as np

from .raytrace_ext import raytrace as raytrace_numpy
try:
    from .torch_native import raytrace_torch as _raytrace_torch
    _has_torch_native = True
except Exception:
    _has_torch_native = False

def enableDebugOutput():
    """Setup the library logger from a user application"""
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(name)s (%(levelname)s) | %(message)s"))
    module_logger.addHandler(sh)

def raytrace(sources, dests, vol, vol_start, vol_spacing, stop_early=-1):
    """
    支持torch和numpy的raytrace接口
    
    自动检测输入类型并返回相应类型的结果：
    - 如果输入是torch.Tensor，返回torch.Tensor
    - 如果输入是np.ndarray，返回np.ndarray
    
    Args:
        sources: (N, 3) 射线起点坐标 (x, y, z)
        dests: (N, 3) 射线终点坐标 (x, y, z)
        vol: (D, H, W) 体积数据（深度, 高度, 宽度）
        vol_start: (3,) 体积起始坐标 (x, y, z)
        vol_spacing: (3,) 体素间距 (spacing_x, spacing_y, spacing_z) = (spacing_W, spacing_H, spacing_D)
            注意：必须使用 (x, y, z) 顺序，对应 (W, H, D)
        stop_early: 提前停止阈值
    
    Returns:
        (N,) 每条射线的路径长度
    
    坐标系统说明：
        - vol 形状为 (D, H, W)
        - 但 vol_start 和 vol_spacing 使用 (x, y, z) = (W, H, D) 顺序
        - 例如：如果 vol_spacing_DHW = (10, 0.5, 0.5)，则应传入
          vol_spacing = (0.5, 0.5, 10)
    """
    try:
        import torch
        is_torch = isinstance(sources, torch.Tensor)
    except ImportError:
        is_torch = False
    if is_torch and _has_torch_native and sources.is_cuda and dests.is_cuda and vol.is_cuda:
        return _raytrace_torch(sources, dests, vol, vol_start, vol_spacing, stop_early)
    if is_torch:
        device = sources.device
        dtype = sources.dtype
        sources_np = sources.detach().cpu().numpy()
        dests_np = dests.detach().cpu().numpy()
        vol_np = vol.detach().cpu().numpy()
        if isinstance(vol_start, torch.Tensor):
            vol_start = tuple(vol_start.detach().cpu().numpy())
        if isinstance(vol_spacing, torch.Tensor):
            vol_spacing = tuple(vol_spacing.detach().cpu().numpy())
        result_np = raytrace_numpy(sources_np, dests_np, vol_np, vol_start, vol_spacing, stop_early)
        result = torch.from_numpy(result_np).to(device=device, dtype=dtype)
        return result
    else:
        return raytrace_numpy(sources, dests, vol, vol_start, vol_spacing, stop_early)
