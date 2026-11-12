# AV Splat

**3D Gaussian Splatting and Reality Capture Processing Suite**

AV Splat is a high-performance desktop and batch processing tool for 3D Gaussian Splatting (3DGS) workflows. Part of the AV reality-capture suite alongside **AV Snap** and **AV Segment**, AV Splat provides modular pipelines for training, rendering, format conversion, and real-time visualization of radiance fields from terrestrial laser scans and photogrammetry.

---

## Architecture

The project is structured into modular components:

```
AV_splat/
├── batch/       # Batch processing pipelines and multi-scene automation
├── core/        # Core mathematical models, rasterization, and training loops
├── formats/     # Import/export handlers (PLY, E57, COLMAP, Nerfstudio)
├── gui/         # Interactive desktop viewer and PySide6 UI
└── utils/       # Geometry, camera calibration, coordinate transforms, and metrics
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/callen0817/av-splat.git
cd av-splat

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Modules

- **Core (`core/`)**: Tensor operations, Gaussian representations (positions, covariances, spherical harmonics, opacities), and CUDA-accelerated rasterization.
- **Formats (`formats/`)**: Parsers and exporters for standard point clouds (`.ply`, `.e57`) and camera trajectories.
- **GUI (`gui/`)**: Desktop interface for real-time inspection, crop box editing, and rendering preview.
- **Batch (`batch/`)**: Automated scripts for multi-dataset processing and optimization queues.
- **Utils (`utils/`)**: Rigid-body transforms, camera models (pinhole, fisheye), and quality assessment metrics.

---

## License

Proprietary / All rights reserved.
