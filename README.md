# AV Splat

Artificial Vision's local **Point → Splat and Splat → Point** conversion and reconstruction studio. The previous repository contained only documentation and package docstrings; this is a new implementation built after confirmation that no earlier application existed.

## Install and run

Use 64-bit Python 3.12:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python launch.py
```

`requirements.txt` provides CPU conversion and desktop processing. `requirements-spz.txt` adds the **official Niantic codec**, pinned to a source commit, and is included by requirements-dev.txt. Building that optional native extension needs a C++ compiler; on the cloud Linux image use `CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_BUILD_PARALLEL_LEVEL=2` for installation. Package downloads and build verification stay enabled.

The application launches offline without installing dependencies. Native 3D preview requires an OpenGL desktop. For headless checks set `QT_QPA_PLATFORM=offscreen AV_HEADLESS=1`.

## Point → Splat

Import PLY, LAS/LAZ, E57 or XYZ/XYZRGB, confirm metre units, choose a Gaussian budget and voxel size, then **Build Rapid Splat**. Neighbourhood covariance produces real anisotropic scales and orientations; RGB and an explicit opacity initialize the remaining Gaussian parameters. Missing RGB requires an explicit neutral-color option, including E57 files with partially uncolored scans.

Rapid mode is geometric initialization, **not optimized photorealistic reconstruction**. Original LiDAR is retained as the survey source. Solver voxel size adapts to fit the Gaussian budget. Full input coordinates remain disk-backed; the bounded representative cloud drives the geometry. The viewer samples Gaussian centres and labels them as such; it does not pretend the point preview is a full splat rasterizer.

Export degree-zero Gaussian PLY, compressed SPZ v3, legacy 32-byte `.splat`, or tiled SPZ delivery. The Python SPZ adapter also supports and tests version 4. Gaussian PLY preserves source XYZ in float64 and uses documented log-scale/logit-opacity/wxyz conventions. SPZ and legacy `.splat` use local origins and explicit axis conversion; **retain the `.av.json` sidecars** to recover source coordinates. Tiled SPZ manifests use bounded local frames and bounding boxes.

## Splat → Point

Import Gaussian PLY, SPZ, legacy `.splat`, or `.avsplat` and select **Splat → Point**. Gaussian inputs are detected automatically; Gaussian PLY is distinguished from ordinary point PLY by its actual fields.

- **Gaussian centres:** one point per selected Gaussian, preserving its stored centre and direction-neutral DC RGB.
- **Covariance samples:** deterministic synthetic points drawn inside anisotropic Gaussian ellipsoids, truncated at three standard deviations. Set samples per Gaussian, opacity threshold and output point budget.

Export point PLY, LAS/LAZ or XYZ. Derived LAS contains an `av_derived` marker; PLY/LAS retain Gaussian IDs and opacity as attributes. LAS coordinate quantization is verified and reported. XYZ is coordinates-only. Native point-cloud intensity, return numbers, stations and survey accuracy cannot be reconstructed from Gaussian parameters, so no such data is invented.

SPZ source origin/axes are restored when a valid AV sidecar is present. Without it, points stay in stored local RUB coordinates and the report explicitly marks the missing origin/georeferencing. Legacy `.splat` sidecars likewise restore the source frame. No unit conversion is guessed. Colors from higher-order SH inputs use the DC component; directional appearance is not measured per-point RGB. Gaussian re-export of higher-order inputs is blocked rather than silently discarding that appearance.

**This is a conversion to derived visualization points, not recovery of the original survey.** Point → Splat → Point can be lossy through decimation, filtering, quantization and synthetic sampling.

```sh
python launch.py --input colored.las --output scene.spz --project scene.avsplat
python launch.py --direction splat-to-point --input scene.spz --output derived.ply
python launch.py --direction splat-to-point --input scene.splat --point-method sample --samples-per-gaussian 8 --max-points 5000000 --output sampled.laz
```

## Calibrated imagery

After Rapid Splat, choose a camera manifest and fit calibrated imagery. Pinhole inputs use `schema_version: 1`, `convention: "opencv_world_to_camera"`, `units: "metres"`, actual RGB image paths, pixel intrinsics `K` and rigid world-to-camera matrices:

```json
{"schema_version":1,"convention":"opencv_world_to_camera","units":"metres","cameras":[{"image":"view1.png","K":[[500,0,320],[0,500,240],[0,0,1]],"world_to_camera":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"distortion":[0,0,0,0]}]}
```

The snippet shows one record; **at least two calibrated views are required**. The last view is held out, never trained. Images must be undistorted; nonzero distortion is rejected. RGB files, poses, image dimensions and source fingerprints are checked. Unregistered images are not accepted as calibrated observations.

**CPU reference:** actual anisotropic Gaussian projection and depth-ordered alpha compositing optimize RGB and opacity with fixed LiDAR geometry, scales and orientations. It has explicit small-scene/image/footprint budgets, reports training and held-out error, and checkpoints atomically. Matching interrupted jobs can resume exactly. It is not a scalable photorealistic production engine.

**Optional CUDA/gsplat:** requirements-gpu.txt pins reviewed-source gsplat separately. The source backend optimizes RGB, opacity, scale and orientation with positions fixed, one camera per batch, a VRAM preflight, numerical checks, held-out error and safe `weights_only` checkpoint loading. **CUDA execution is unverified in this cloud task**, and the default packaged application is CPU-only. It must be tested on a supported NVIDIA/CUDA machine before production use. No external model weights or mandatory cloud services are used.

```sh
python launch.py --input colored.ply --max-gaussians 5000 --calibration cameras.json --checkpoint fit.npz --iterations 50 --output fitted.ply
# Resume identical settings/input geometry/calibration with --resume.
```

## Batch and project persistence

`python launch.py --batch jobs.json --state jobs.state.json` runs a sequential resumable queue. Each job has a unique ID, input/output paths relative to the manifest, and a config. Completed unchanged outputs are skipped; changed source fingerprints or configuration reprocess the relevant job.

```json
{"schema_version":1,"jobs":[{"id":"to-splat","input":"colored.ply","output":"scene.spz","config":{"voxel_size":0.02}},{"id":"to-point","input":"existing.spz","output":"derived.ply","config":{"direction":"splat-to-point","method":"centres","opacity_threshold":0.05}}]}
```

Batch outputs may not overwrite inputs or the manifest. This is a file-job queue, not a dependency graph: chained jobs must be run separately. `.avsplat` projects store Gaussian arrays and provenance without pickle. Import caches and processing intermediates use a writable scratch workspace; `AV_WORKSPACE` overrides its location. Save projects, checkpoints and outputs outside ephemeral caches when durable retention is required.

## Validation and packaging

```sh
python -m unittest discover -s tests -v
python launch.py --smoke-test
python benchmarks/rapid.py
python -m PyInstaller --noconfirm AV_Splat.spec
# Linux: dist/AV_Splat/AV_Splat --smoke-test
# Windows: dist/AV_Splat/AV_Splat.exe --smoke-test
```

See [measured baseline](benchmarks/results.json) and [implementation status](docs/MODERNIZATION.md). The small synthetic benchmark records processing time and actual output sizes; no speedup over a nonexistent prior engine is claimed. Windows packaging, real-image reconstruction quality, physical GPU performance, rendering FPS and independent viewer interoperability remain unverified.

The interface uses the AV Scan/scanAR glass palette and layout patterns. Original logo artwork is loaded from `logos/av_splat.png` and included by the packaging recipe when supplied. The inline chat logo is visible but not exposed as a downloadable workspace asset; exact-file integration remains pending. No replacement logo is generated. Third-party licences and pinned codec/backend sources are documented in [licenses/README.md](licenses/README.md).

Completed import caches are read-only and versioned. Setting the same writable `AV_WORKSPACE` for AV Segment and AV Splat lets them reuse compatible imports under the shared `imports/` directory, with cross-process writer locks. Their processing jobs use separate work folders. Close applications before manually clearing a scratch cache.

## Ownership

Proprietary / All rights reserved. Third-party components retain their respective licences.
