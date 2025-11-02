import logging
import numpy as np

from .raytrace_ext import raytrace as raytrace_numpy, beamtrace as beamtrace_numpy
from .geometry import rotateAroundAxisAtOriginRHS, inverseRotateBeamAtOriginRHS

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
        sources: (N, 3) 射线起点坐标
        dests: (N, 3) 射线终点坐标
        vol: (D, H, W) 体积数据
        vol_start: (3,) 体积起始坐标
        vol_spacing: (3,) 体素间距
        stop_early: 提前停止阈值
    
    Returns:
        (N,) 每条射线的路径长度
    """
    try:
        import torch
        is_torch = isinstance(sources, torch.Tensor)
    except ImportError:
        is_torch = False
    
    if is_torch:
        # 保存原始设备和数据类型
        device = sources.device
        dtype = sources.dtype
        
        # 转换为numpy进行计算
        sources_np = sources.detach().cpu().numpy()
        dests_np = dests.detach().cpu().numpy()
        vol_np = vol.detach().cpu().numpy()
        
        # 处理vol_start和vol_spacing
        if isinstance(vol_start, torch.Tensor):
            vol_start = tuple(vol_start.detach().cpu().numpy())
        if isinstance(vol_spacing, torch.Tensor):
            vol_spacing = tuple(vol_spacing.detach().cpu().numpy())
        
        # 调用numpy版本
        result_np = raytrace_numpy(sources_np, dests_np, vol_np, vol_start, vol_spacing, stop_early)
        
        # 转换回torch
        result = torch.from_numpy(result_np).to(device=device, dtype=dtype)
        return result
    else:
        # 直接调用numpy版本
        return raytrace_numpy(sources, dests, vol, vol_start, vol_spacing, stop_early)

def beamtrace(sad, det_dims, det_center, det_spacing, det_pixelsize, det_azi, det_zen, det_ang, vol, vol_start, vol_spacing, stop_early=-1):
    """
    支持torch和numpy的beamtrace接口
    
    自动检测输入类型并返回相应类型的结果：
    - 如果vol是torch.Tensor，返回torch.Tensor
    - 如果vol是np.ndarray，返回np.ndarray
    """
    try:
        import torch
        is_torch = isinstance(vol, torch.Tensor)
    except ImportError:
        is_torch = False
    
    if is_torch:
        # 保存原始设备和数据类型
        device = vol.device
        dtype = vol.dtype
        
        # 转换为numpy进行计算
        vol_np = vol.detach().cpu().numpy()
        
        # 处理其他可能的tensor参数
        if isinstance(vol_start, torch.Tensor):
            vol_start = tuple(vol_start.detach().cpu().numpy())
        if isinstance(vol_spacing, torch.Tensor):
            vol_spacing = tuple(vol_spacing.detach().cpu().numpy())
        if isinstance(det_center, torch.Tensor):
            det_center = tuple(det_center.detach().cpu().numpy())
        
        # 调用numpy版本
        result_np = beamtrace_numpy(sad, det_dims, det_center, det_spacing, det_pixelsize, 
                                   det_azi, det_zen, det_ang, vol_np, vol_start, vol_spacing, stop_early)
        
        # 转换回torch
        result = torch.from_numpy(result_np).to(device=device, dtype=dtype)
        return result
    else:
        # 直接调用numpy版本
        return beamtrace_numpy(sad, det_dims, det_center, det_spacing, det_pixelsize, 
                             det_azi, det_zen, det_ang, vol, vol_start, vol_spacing, stop_early)
