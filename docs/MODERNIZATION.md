# Implementation and acceptance status

The original commit (`8327c0c5a4f0fb9f6e67b172d450b9db6ebb2dec`) contained package docstrings, requirements and intended architecture only. After confirmation that no earlier implementation existed, this new application was built with independently runnable CPU workflows and optional accelerated training.

Implemented and generated-data tested:

- Rapid XYZ/RGB → anisotropic Gaussian covariance, orientation, scale, opacity and SH-DC appearance, with adaptive bounded voxel processing.
- Gaussian PLY, official-codec SPZ 3/4, legacy `.splat`, origin/axis metadata and tiled SPZ export.
- Gaussian PLY/SPZ/`.splat`/snapshot → centre points or deterministic covariance samples; point PLY, LAS/LAZ and XYZ exports with derived-data markers and validation.
- Large-world-coordinate forward/inverse roundtrips, opacity filtering, output budgets, covariance anisotropy and quaternion conventions.
- Actual calibrated pinhole image projection, alpha compositing, CPU RGB/opacity fitting, held-out views, real PNG/manifest parsing, input-change rejection and interrupted checkpoint recovery.
- GUI conversion-mode controls, background jobs, native Qt6/OpenGL previews, source/splat comparison, camera/section controls, saved Gaussian snapshots and resumable file batches.
- Missing-GPU handling, scratch-disk checks, incomplete-import recovery and locked import caches.

The CUDA/gsplat source backend is implemented but **has not executed in this environment**. Source licence/version checks and a missing-backend test do not establish CUDA correctness, performance or successful training. The default PyInstaller build intentionally excludes Torch/CUDA.

Current budgets: at most one million solver/imported Gaussians, 100k preview points, a 256 MiB snapshot restore budget, 5m default inverse points (up to 20m explicitly configured), 256 calibrated views and 512 MiB decoded RGB images. CPU appearance fitting additionally limits Gaussian count, image pixels and total footprint pixels. The input cloud is disk-backed, but bounded geometry/feature arrays, some color conversions and review summaries still consume RAM. This is not a complete arbitrary-scale distributed reconstruction system.

The 3D preview shows Gaussian centres. A separate calibrated-camera renderer composites real anisotropic Gaussian footprints. Interactive full-scene GPU Gaussian rasterization, progressive frustum-based LOD and mid-file scene reconstruction checkpoints remain future work. Batch recovery is at completed-file granularity; appearance recovery is at saved optimization iterations.

Inverse conversion produces derived points, not original LiDAR returns. SPZ without origin metadata cannot recover source georeferencing. Higher-order SH colors are reduced to the direction-neutral DC component only for explicit point extraction; higher-order Gaussian re-export is rejected. Original survey inputs are preserved.

The benchmark in `benchmarks/results.json` is a synthetic curved surface. There is no prior executable baseline. Native-codec roundtrips are not independent external-viewer acceptance, and synthetic held-out views are not a real-building photometric-quality benchmark. Windows deployment, ReCap consumption, hardware VRAM/FPS, real calibrated scenes and signed release packages require separate validation.

The user supplied inline logo references. Matching Snap/Segment files already exist in those repositories; an exact downloadable AV Splat PNG/SVG is still needed at `logos/av_splat.png`. The asset hook and packaging route are prepared without inventing artwork.
