import torch
from .torch_native import raytrace_torch

def _generate_ray_pairs(sources, dests):
    """Generate all source-dest pairs: [num_sources, num_dests, 3]"""
    num_sources = sources.shape[0]
    num_dests = dests.shape[0]
    ray_sources = sources.repeat_interleave(num_dests, dim=0)
    ray_dests = dests.repeat(num_sources, 1)
    return ray_sources, ray_dests, num_sources, num_dests

class CTProjector(torch.nn.Module):
    def __init__(self, stop_early: float = -1.0):
        super().__init__()
        self.stop_early = float(stop_early)
    def forward(self, vols, sources, dests, vol_start, vol_spacing, num_streams: int = 1, return_shape=None):
        if not (torch.is_tensor(vols) and torch.is_tensor(sources) and torch.is_tensor(dests)):
            raise TypeError('vols, sources, dests must be torch.Tensor')
        if vols.device.type != 'cuda' or sources.device.type != 'cuda' or dests.device.type != 'cuda':
            raise ValueError('vols, sources, dests must be CUDA tensors')
        if vols.dtype != torch.float32 or sources.dtype != torch.float32 or dests.dtype != torch.float32:
            raise TypeError('tensors must be float32')
        if vols.ndim == 3:
            ray_sources, ray_dests, num_sources, num_dests = _generate_ray_pairs(sources, dests)
            rpl = raytrace_torch(ray_sources, ray_dests, vols, vol_start, vol_spacing, self.stop_early)
            if return_shape == 'flat':
                return rpl
            return rpl.view(num_sources, num_dests)
        if vols.ndim != 4:
            raise ValueError('vols must be [D,H,W] or [B,D,H,W]')
        B = vols.shape[0]
        if sources.ndim == 2 and sources.shape[-1] == 3:
            sources_b = sources.unsqueeze(0).expand(B, -1, -1)
            num_sources = sources.shape[0]
        elif sources.ndim == 3 and sources.shape[-1] == 3 and sources.shape[0] == B:
            sources_b = sources
            num_sources = sources.shape[1]
        else:
            raise ValueError('sources must be [N,3] or [B,N,3]')
        if dests.ndim == 2 and dests.shape[-1] == 3:
            dests_b = dests.unsqueeze(0).expand(B, -1, -1)
            num_dests = dests.shape[0]
        elif dests.ndim == 3 and dests.shape[-1] == 3 and dests.shape[0] == B:
            dests_b = dests
            num_dests = dests.shape[1]
        else:
            raise ValueError('dests must be [N,3] or [B,N,3]')
        N = num_sources * num_dests
        out = torch.empty((B, N), device=vols.device, dtype=torch.float32)
        num_streams = max(1, int(num_streams))
        if num_streams == 1:
            for i in range(B):
                ray_sources, ray_dests, _, _ = _generate_ray_pairs(sources_b[i], dests_b[i])
                rpl_i = raytrace_torch(ray_sources, ray_dests, vols[i], vol_start, vol_spacing, self.stop_early)
                out[i] = rpl_i
        else:
            streams = [torch.cuda.Stream(device=vols.device) for _ in range(num_streams)]
            for i in range(B):
                s = streams[i % num_streams]
                with torch.cuda.stream(s):
                    ray_sources, ray_dests, _, _ = _generate_ray_pairs(sources_b[i].contiguous(), dests_b[i].contiguous())
                    rpl_i = raytrace_torch(ray_sources, ray_dests, vols[i].contiguous(), vol_start, vol_spacing, self.stop_early)
                    out[i] = rpl_i
            torch.cuda.synchronize(vols.device)
        if return_shape == 'flat':
            return out
        return out.view(B, num_sources, num_dests)
