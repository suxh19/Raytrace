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
        _ext = _build_extension()
    return _ext


def raytrace_torch(sources, dests, vol, vol_start, vol_spacing, stop_early: float = -1.0):
    if not (torch.is_tensor(sources) and torch.is_tensor(dests) and torch.is_tensor(vol)):
        raise TypeError('sources, dests, vol must be torch.Tensor')

    if sources.device.type != 'cuda' or dests.device.type != 'cuda' or vol.device.type != 'cuda':
        raise ValueError('sources, dests, vol must be CUDA tensors')

    if sources.dtype != torch.float32 or dests.dtype != torch.float32 or vol.dtype != torch.float32:
        raise TypeError('sources, dests, vol must be float32')

    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError('sources must have shape [N, 3]')
    if dests.ndim != 2 or dests.shape[1] != 3:
        raise ValueError('dests must have shape [N, 3]')
    if vol.ndim != 3:
        raise ValueError('vol must have shape [D, H, W]')

    # Normalize vol_start / vol_spacing to Python floats (x,y,z)
    if torch.is_tensor(vol_start):
        vol_start = vol_start.detach().cpu().tolist()
    if torch.is_tensor(vol_spacing):
        vol_spacing = vol_spacing.detach().cpu().tolist()
    if len(vol_start) != 3 or len(vol_spacing) != 3:
        raise ValueError('vol_start and vol_spacing must be length-3 sequences (x, y, z)')

    sources_c = sources.contiguous()
    dests_c = dests.contiguous()
    vol_c = vol.contiguous()

    ext = _get_ext()
    rpl = ext.raytrace_forward(
        sources_c, dests_c, vol_c,
        float(vol_start[0]), float(vol_start[1]), float(vol_start[2]),
        float(vol_spacing[0]), float(vol_spacing[1]), float(vol_spacing[2]),
        float(stop_early),
    )
    return rpl
