import numpy as np
import matplotlib.pyplot as plt


def main():
    print('load batch projections ...')
    proj_np = np.load('test/batch_projections.npy')
    print(f'shape: {proj_np.shape}')

    B, num_sources, num_detectors = proj_np.shape
    print(f'batch: {B}, sources: {num_sources}, detectors: {num_detectors}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(B):
        ax = axes[i]
        im = ax.imshow(proj_np[i], aspect='auto', cmap='gray', origin='lower')
        ax.set_xlabel('Detector Index')
        ax.set_ylabel('Source Index')
        ax.set_title(f'Batch {i}: Projection Matrix ({num_sources}x{num_detectors})')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('test/batch_projections_subplot.png', dpi=150, bbox_inches='tight')
    print('saved: test/batch_projections_subplot.png')
    plt.show()

    print('\nstatistics:')
    for i in range(B):
        vmin = proj_np[i].min()
        vmax = proj_np[i].max()
        vmean = proj_np[i].mean()
        print(f'  batch {i}: min={vmin:.4f}, max={vmax:.4f}, mean={vmean:.4f}')


if __name__ == '__main__':
    main()
