# Repository Guidelines

## Project Structure & Module Organization
- `raytrace/` holds the Python package and Cython bridge (`raytrace_ext.pyx`, built `.so`); keep high-level APIs in `__init__.py`.
- `raytrace/src/` contains CUDA kernels (`raytrace.cu`, `geometry*.cu[h]`); update headers alongside implementations.
- `staticCTGeometry/` stores beam geometry `.npy` assets consumed by demos; treat as read-only source data.
- `test/` includes the CT demo script and generated artifacts (`full_projections.npy`, `static_ct_raytrace_torch.png`); clean large outputs before committing.
- Top-level `setup.py` and `pyproject.toml` define the Cython build; `build/` is transient build output.

## Build, Test, and Development Commands
- `pip install -e .` builds the CUDA extension in-place; ensure `nvcc` and `CUDAHOME` point to the same toolkit.
- `python setup.py build_ext --inplace` forces a rebuild of `raytrace_ext` after kernel changes.
- `python test/demo_static_ct_raytrace.py` runs the static CT walkthrough, produces `test/static_ct_raytrace_torch.png`, and validates GPU support.

## Coding Style & Naming Conventions
- Python follows PEP 8 with 4-space indentation; expose public helpers via snake_case functions and document tensor shape expectations.
- C++/CUDA adopts 2-space continuation indent with camelCase device functions (`siddonRPL`, `cudaRayTrace`); mirror header and source naming.
- Keep file names lowercase with underscores; place new CUDA utilities in `raytrace/src` and add matching `.cuh` exports.
- use "uv run" to run the python codes

## Testing Guidelines
- Prefer lightweight smoke checks via `python test/demo_static_ct_raytrace.py`; run once on GPU and CPU fallback before submitting.
- Store generated `.npy/.png` artifacts under `test/` and avoid committing files larger than existing baselines.
- When adding automated tests, drop them alongside demos under `test/` and ensure they skip gracefully if CUDA is unavailable.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`); mirror the succinct, lowercase summaries already in history.
- Reference related issues in the body, describe GPU/CPU impacts, and attach refreshed demo screenshots when visuals change.
- PRs should state the CUDA version used, build/test commands executed, and any new assets or dependencies introduced.

## CUDA Environment & Troubleshooting
- Verify `nvcc --version` before builds; if CUDA paths are missing, export `CUDAHOME=/usr/local/cuda` to align headers and libs.
- After toolkit upgrades, clear stale artifacts (`rm -rf build raytrace.egg-info`) and rebuild to avoid ABI mismatches.

## language
- always response me in Chiense

