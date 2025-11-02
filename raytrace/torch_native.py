import os
import torch
from torch.utils.cpp_extension import load


def _build_extension():
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, 'torch_ext')
    include_dir = os.path.join(base_dir, 'src')

    ext = load(
        name='raytrace_torch_ext',
        sources=[
            os.path.join(src_dir, 'raytrace_torch.cc'),
            os.path.join(src_dir, 'raytrace_torch_kernel.cu'),
        ],
        extra_include_paths=[include_dir],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False,
    )
    return ext


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        try:
            _ext = _build_extension()
        except Exception as e:
            raise RuntimeError(f'Failed to build CUDA extension: {e}') from e
    if _ext is None:
        raise RuntimeError('Extension is None after build')
    return _ext


def _to_float3(param):
    """将参数转换为长度为3的浮点数列表"""
    if torch.is_tensor(param):
        param = param.detach().cpu().tolist()
    if len(param) != 3:
        raise ValueError(f'parameter must be length-3 sequence (x, y, z), got {len(param)}')
    return [float(param[0]), float(param[1]), float(param[2])]

def _validate_shape(tensor, expected_shape, name):
    """验证张量形状"""
    if tensor.ndim != len(expected_shape):
        raise ValueError(f'{name} must have {len(expected_shape)} dimensions, got {tensor.ndim}')
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f'{name} shape[{i}] must be {expected}, got {actual}')

def raytrace_torch(sources, dests, vol, vol_start, vol_spacing, stop_early: float = -1.0):
    # 验证输入类型和设备
    if not all(torch.is_tensor(t) for t in (sources, dests, vol)):
        raise TypeError('sources, dests, vol must be torch.Tensor')
    if not all(t.device.type == 'cuda' for t in (sources, dests, vol)):
        raise ValueError('sources, dests, vol must be CUDA tensors')
    if not all(t.dtype == torch.float32 for t in (sources, dests, vol)):
        raise TypeError('sources, dests, vol must be float32')
    
    # 验证形状
    _validate_shape(sources, (None, 3), 'sources')
    _validate_shape(dests, (None, 3), 'dests')
    _validate_shape(vol, (None, None, None), 'vol')
    
    # 转换参数为浮点数列表
    start = _to_float3(vol_start)
    spacing = _to_float3(vol_spacing)
    
    # 调用 C++ 扩展
    ext = _get_ext()
    return ext.raytrace_forward(
        sources.contiguous(), dests.contiguous(), vol.contiguous(),
        start[0], start[1], start[2],
        spacing[0], spacing[1], spacing[2],
        float(stop_early)
    )
