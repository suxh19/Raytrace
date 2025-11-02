import torch
from .torch_native import raytrace_torch

def _validate_inputs(vols, sources, dests):
    """验证输入张量的类型、设备和数据类型"""
    if not all(torch.is_tensor(t) for t in (vols, sources, dests)):
        raise TypeError('vols, sources, dests must be torch.Tensor')
    if not all(t.device.type == 'cuda' for t in (vols, sources, dests)):
        raise ValueError('vols, sources, dests must be CUDA tensors')
    if not all(t.dtype == torch.float32 for t in (vols, sources, dests)):
        raise TypeError('tensors must be float32')

def _generate_ray_pairs(sources, dests):
    """生成所有 source-dest 射线对: [num_sources*num_dests, 3]"""
    num_sources, num_dests = sources.shape[0], dests.shape[0]
    ray_sources = sources.repeat_interleave(num_dests, dim=0)
    ray_dests = dests.repeat(num_sources, 1)
    return ray_sources, ray_dests, num_sources, num_dests

def _expand_to_batch(tensor, batch_size):
    """将 [N,3] 扩展为 [B,N,3]"""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).expand(batch_size, -1, -1), tensor.shape[0]
    elif tensor.ndim == 3 and tensor.shape[0] == batch_size:
        return tensor, tensor.shape[1]
    raise ValueError(f'tensor must be [N,3] or [B,N,3] with B={batch_size}')

class CTProjector(torch.nn.Module):
    def __init__(self, stop_early: float = -1.0):
        super().__init__()
        self.stop_early = float(stop_early)
    
    def forward(self, vols, sources, dests, vol_start, vol_spacing, num_streams: int = 1, return_shape=None):
        _validate_inputs(vols, sources, dests)
        
        # 单体积情况: [D,H,W]
        if vols.ndim == 3:
            ray_sources, ray_dests, num_sources, num_dests = _generate_ray_pairs(sources, dests)
            rpl = raytrace_torch(ray_sources, ray_dests, vols, vol_start, vol_spacing, self.stop_early)
            return rpl if return_shape == 'flat' else rpl.view(num_sources, num_dests)
        
        # 批量体积情况: [B,D,H,W]
        if vols.ndim != 4:
            raise ValueError('vols must be [D,H,W] or [B,D,H,W]')
        
        B = vols.shape[0]
        sources_b, num_sources = _expand_to_batch(sources, B)
        dests_b, num_dests = _expand_to_batch(dests, B)
        
        out = torch.empty((B, num_sources * num_dests), device=vols.device, dtype=torch.float32)
        num_streams = max(1, int(num_streams))
        
        # 批处理投影
        streams = [torch.cuda.Stream(device=vols.device) for _ in range(num_streams)] if num_streams > 1 else [None]
        for i in range(B):
            stream_ctx = torch.cuda.stream(streams[i % num_streams]) if streams[0] else None
            with stream_ctx if stream_ctx else torch.no_grad():
                src, dst = sources_b[i].contiguous(), dests_b[i].contiguous()
                ray_sources, ray_dests, _, _ = _generate_ray_pairs(src, dst)
                out[i] = raytrace_torch(ray_sources, ray_dests, vols[i].contiguous(), 
                                       vol_start, vol_spacing, self.stop_early)
        
        if num_streams > 1:
            torch.cuda.synchronize(vols.device)
        
        return out if return_shape == 'flat' else out.view(B, num_sources, num_dests)
