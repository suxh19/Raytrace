import time
import numpy as np
import torch
from skimage import data
from scipy.ndimage import zoom
from raytrace.projector import CTProjector


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type != 'cuda':
        raise SystemExit('CUDA is required for batch test')

    print('\nload static CT geometry ...')
    sources = torch.from_numpy(np.load('staticCTGeometry/source_positions.npy')).float().to(device)
    detectors = torch.from_numpy(np.load('staticCTGeometry/detector_positions.npy')).float().to(device)
    print(f"sources: {sources.shape[0]}")
    print(f"detectors: {detectors.shape[0]}")

    print('\nprepare volume batch ...')
    phantom_2d = data.shepp_logan_phantom()
    phantom_size = phantom_2d.shape[0]
    vol_size = (1, 512, 512)
    vol_spacing_DHW = (1, 0.5, 0.5)

    vol_center = torch.tensor([0.0, 0.0, 0.0], device=device)
    vol_start = vol_center - torch.tensor([
        (vol_size[2] - 1) * vol_spacing_DHW[2] / 2,
        (vol_size[1] - 1) * vol_spacing_DHW[1] / 2,
        (vol_size[0] - 1) * vol_spacing_DHW[0] / 2,
    ], device=device)
    vol_spacing = (vol_spacing_DHW[2], vol_spacing_DHW[1], vol_spacing_DHW[0])

    vol_np = np.zeros(vol_size, dtype=np.float32)
    scale_factor = 512 / phantom_size
    phantom_resized = np.asarray(zoom(phantom_2d, scale_factor, order=3), dtype=np.float32)
    for z in range(vol_size[0]):
        vol_np[z, :, :] = phantom_resized
    vol = torch.from_numpy(vol_np).float().to(device)

    B = 4
    vols = vol.unsqueeze(0).repeat(B, 1, 1, 1).contiguous()
    for i in range(B):
        vols[i].mul_(1.0 + 0.05 * i)
    print(f"vols: {vols.shape}")

    print('\nbuild geometry ...')
    sources_3d = torch.stack([sources[:, 0], sources[:, 1], torch.zeros_like(sources[:, 0])], dim=1)
    detectors_3d = torch.stack([detectors[:, 0], detectors[:, 1], torch.zeros_like(detectors[:, 0])], dim=1)
    print(f"sources_3d: {sources_3d.shape}, detectors_3d: {detectors_3d.shape}")

    projector = CTProjector(stop_early=-1.0).to(device)

    print('\nwarmup ...')
    torch.cuda.synchronize()
    _ = projector(vols[:1], sources_3d, detectors_3d, vol_start, vol_spacing, num_streams=1)
    torch.cuda.synchronize()

    print('\nrun batch with num_streams=1 ...')
    t0 = time.time()
    out1 = projector(vols, sources_3d, detectors_3d, vol_start, vol_spacing, num_streams=1)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"time: {(t1 - t0)*1000:.2f} ms, shape: {out1.shape}")

    print('\nrun batch with num_streams=2 ...')
    t0 = time.time()
    out2 = projector(vols, sources_3d, detectors_3d, vol_start, vol_spacing, num_streams=2)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"time: {(t1 - t0)*1000:.2f} ms, shape: {out2.shape}")

    print('\nvalidate outputs (close enough) ...')
    diff = (out1 - out2).abs().max().item()
    print(f"max |out1-out2|: {diff:.6f}")

    proj_np = out2.detach().cpu().numpy()
    np.save('test/batch_projections.npy', proj_np)
    print('saved: test/batch_projections.npy')


if __name__ == '__main__':
    main()
