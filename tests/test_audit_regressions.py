"""Regression tests for Wave 1–3 audit fixes."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('AV_HEADLESS', '1')
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image

from av_common.data import Dataset, load_dataset
from av_common.e57 import normalize_e57_colors
from av_common.exports import protect_source, write_ply
from av_common.geometry import voxelize
from av_common.queue import run_queue
from batch.queue import run as run_batch
from core.cameras import Camera, load_cameras
from core.gpu import GpuConfig
from core.rapid import GaussianScene, RapidConfig, rapid_splat
from core.to_points import splat_to_points
from formats.gaussian import export_ply, protect, read_ply, save_project
from formats.import_splats import read_splat
from formats.legacy_splat import export_splat
from formats.spz_export import export_tiles


class _FakeNode:
    def __init__(self, values):
        self._values = values

    def isDefined(self, key):
        return key in self._values

    def __getitem__(self, key):
        val = self._values[key]
        return val if isinstance(val, _FakeNode) else SimpleNamespace(value=lambda v=val: v)


class AuditRegressionTests(unittest.TestCase):
    def _sample_scene(self, metadata=None):
        n = 16
        rng = np.random.default_rng(123)
        means = rng.uniform(0, 1, size=(n, 3))
        colors = rng.uniform(0.1, 0.9, size=(n, 3)).astype('float32')
        scales = np.full((n, 3), -3.0, dtype='float32')
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype='float32'), (n, 1))
        logits = np.full(n, 1.5, dtype='float32')
        return GaussianScene(means, colors, scales, quats, logits, metadata if metadata is not None else {'schema_version': 1})

    def test_read_ply_normalizes_quaternions_and_clips_extremes(self):
        with tempfile.TemporaryDirectory() as d:
            ply_path = Path(d) / 'external_3dgs.ply'
            fields = {
                'x': np.array([0.0, 1.0], dtype='float64'),
                'y': np.array([0.0, 2.0], dtype='float64'),
                'z': np.array([0.0, 3.0], dtype='float64'),
                'f_dc_0': np.array([0.0, 0.5], dtype='float32'),
                'f_dc_1': np.array([0.0, -0.5], dtype='float32'),
                'f_dc_2': np.array([0.0, 0.25], dtype='float32'),
                'opacity': np.array([-80.0, 75.0], dtype='float32'),
                'scale_0': np.array([-35.0, 25.0], dtype='float32'),
                'scale_1': np.array([-5.0, 0.0], dtype='float32'),
                'scale_2': np.array([-2.0, 1.0], dtype='float32'),
                'rot_0': np.array([2.0, 0.0], dtype='float32'),
                'rot_1': np.array([0.0, 3.0], dtype='float32'),
                'rot_2': np.array([0.0, 4.0], dtype='float32'),
                'rot_3': np.array([0.0, 0.0], dtype='float32'),
            }
            write_ply(ply_path, fields)
            scene = read_ply(ply_path)
            np.testing.assert_allclose(np.linalg.norm(scene.quaternions, axis=1), [1.0, 1.0], atol=1e-6)
            self.assertEqual(float(scene.opacity_logits.min()), -30.0)
            self.assertEqual(float(scene.opacity_logits.max()), 30.0)
            self.assertEqual(float(scene.log_scales.min()), -20.0)
            self.assertEqual(float(scene.log_scales.max()), 10.0)

            bad_ply = Path(d) / 'zero_quat.ply'
            fields['rot_0'] = np.array([0.0, 1.0], dtype='float32')
            write_ply(bad_ply, fields)
            with self.assertRaises(ValueError):
                read_ply(bad_ply)

    def test_e57_color_normalization_8bit_16bit_float_and_limits(self):
        c8 = normalize_e57_colors(np.array([[0, 128, 255]], dtype='uint8'))
        np.testing.assert_allclose(c8, [[0.0, 128 / 255, 1.0]], atol=1e-6)

        c16 = normalize_e57_colors(np.array([[0, 32768, 65535]], dtype='uint16'))
        np.testing.assert_allclose(c16, [[0.0, 32768 / 65535, 1.0]], atol=1e-6)

        cfloat = normalize_e57_colors(np.array([[0.0, 0.25, 1.0]], dtype='float32'))
        np.testing.assert_allclose(cfloat, [[0.0, 0.25, 1.0]], atol=1e-6)

        c16_float_buf = normalize_e57_colors(np.array([[0.0, 32768.0, 65535.0]], dtype='float64'))
        np.testing.assert_allclose(c16_float_buf, [[0.0, 32768 / 65535, 1.0]], atol=1e-6)

        header = SimpleNamespace(
            node=_FakeNode({
                'colorLimits': _FakeNode({
                    'colorRedMinimum': 0.0, 'colorRedMaximum': 65535.0,
                    'colorGreenMinimum': 0.0, 'colorGreenMaximum': 65535.0,
                    'colorBlueMinimum': 0.0, 'colorBlueMaximum': 65535.0,
                })
            })
        )
        clim = normalize_e57_colors(np.array([[0.0, 65535.0, 32767.5]], dtype='float64'), header=header)
        np.testing.assert_allclose(clim, [[0.0, 1.0, 0.5]], atol=1e-6)

    def test_none_source_metadata_handling(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scene = self._sample_scene({'schema_version': 1, 'source': None})
            protect(scene, d / 'out.ply')
            pts = splat_to_points(scene, d / 'pts')
            self.assertIsNone(pts.metadata['source'])
            protect_source(Dataset(pts.points, pts.colors, metadata={'source': None}), d / 'pts_out.ply')
            out_ply = d / 'nested' / 'out.ply'
            export_ply(out_ply, scene)
            self.assertTrue(out_ply.exists())
            Path(str(out_ply) + '.av.json').write_text(json.dumps({'source': None}), encoding='utf-8')
            reloaded = read_ply(out_ply)
            self.assertIsNone(reloaded.metadata['crs'])
            proj = d / 'sub' / 'scene.avsplat'
            save_project(proj, scene)
            self.assertTrue(proj.exists())

    def test_protect_missing_source_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scene = self._sample_scene({'schema_version': 1, 'source': {'source': {'path': str(d / 'deleted.ply'), 'size': 1, 'mtime_ns': 1}}})
            with self.assertRaises(ValueError):
                protect(scene, d / 'out.ply')

    def test_batch_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sub = d / 'jobdir'
            sub.mkdir()
            for bad_job in [
                {'id': 'escape_in', 'input': '../outside.ply', 'output': 'out.ply'},
                {'id': 'escape_out', 'input': 'in.ply', 'output': '../outside.ply'},
                {'id': 'abs_in', 'input': str(d / 'abs.ply'), 'output': 'out.ply'},
                {'id': 'abs_out', 'input': 'in.ply', 'output': str(d / 'abs_out.ply')},
            ]:
                manifest = sub / 'jobs.json'
                manifest.write_text(json.dumps({'schema_version': 1, 'jobs': [bad_job]}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    run_queue(manifest, sub / 'state.json', lambda *args: None)

    def test_camera_manifest_path_traversal_and_K10_validation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            outside = d / 'secret.png'
            Image.new('RGB', (16, 16), (128, 128, 128)).save(outside)
            sub = d / 'calib'
            sub.mkdir()
            for bad_image in ['../secret.png', str(outside)]:
                manifest = sub / 'cameras.json'
                manifest.write_text(json.dumps({
                    'schema_version': 1,
                    'convention': 'opencv_world_to_camera',
                    'units': 'metres',
                    'cameras': [{'image': bad_image, 'K': [[10, 0, 8], [0, 10, 8], [0, 0, 1]], 'world_to_camera': np.eye(4).tolist()}],
                }), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_cameras(manifest)

            bad_K = np.array([[10.0, 0.0, 8.0], [1.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
            cam = Camera(bad_K, np.eye(4), np.zeros((16, 16, 3), dtype='float32'), str(outside))
            with self.assertRaises(ValueError):
                cam.validate()

    def test_export_tiles_partitioning_and_version_forwarding(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scene = self._sample_scene()
            scene.means[:8, 0] += 0.0
            scene.means[8:, 0] += 1000.0
            calls = []

            def fake_export_spz(path, tile, cancel=lambda: False, source_axes='RFU', version=3):
                calls.append((Path(path).name, len(tile.means), source_axes, version))
                Path(path).write_bytes(b'fake')
                return str(path)

            with patch('formats.spz_export.export_spz', side_effect=fake_export_spz):
                manifest_path = export_tiles(d / 'nested' / 'tiles', scene, source_axes='RDF', tile_size=512.0, version=4)
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            self.assertEqual(manifest['spz_version'], 4)
            self.assertEqual(sum(t['gaussians'] for t in manifest['tiles']), len(scene.means))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(v == 4 and ax == 'RDF' for _, _, ax, v in calls))

    def test_gpu_config_validate(self):
        self.assertIsInstance(GpuConfig().validate(), GpuConfig)
        with self.assertRaises(ValueError):
            GpuConfig(iterations=0).validate()
        with self.assertRaises(ValueError):
            GpuConfig(learning_rate=0.5).validate()
        with self.assertRaises(ValueError):
            GpuConfig(checkpoint_every=0).validate()
        with self.assertRaises(ValueError):
            GpuConfig(max_image_side=0).validate()

    def test_splat_missing_source_axes_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scene = self._sample_scene()
            splat_path = d / 'scene.splat'
            export_splat(splat_path, scene, source_axes='RFU')
            sidecar = Path(str(splat_path) + '.av.json')
            sidecar.write_text(json.dumps({'stored_axes': 'RUB', 'local_origin_source_axes': [0, 0, 0]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                read_splat(splat_path)

    def test_readonly_cache_uses_mmap_mode_r(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            points = np.random.default_rng(7).uniform(size=(30, 3))
            fields = {name: points[:, i] for i, name in enumerate('xyz')}
            fields['intensity'] = np.arange(30, dtype='uint16')
            src = d / 'cloud.ply'
            write_ply(src, fields)
            load_dataset(src, d / 'cache')
            reloaded = load_dataset(src, d / 'cache')
            self.assertEqual(getattr(reloaded.points.base, 'mode', getattr(reloaded.points, 'mode', None)), 'r')
            self.assertEqual(getattr(reloaded.extras['intensity'].base, 'mode', getattr(reloaded.extras['intensity'], 'mode', None)), 'r')

    def test_batch_splat_output_and_voxelize_retry(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            points = np.random.default_rng(8).uniform(size=(200, 3))
            vc = voxelize(points, d / 'vox', size=0.01, max_voxels=30)
            self.assertLessEqual(len(vc.points), 30)
            fields = {name: points[:, i] for i, name in enumerate('xyz')}
            for ch in ['red', 'green', 'blue']:
                fields[ch] = np.full(200, 128, dtype='uint8')
            write_ply(d / 'in.ply', fields)
            manifest = d / 'jobs.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'jobs': [{'id': 'splat_job', 'input': 'in.ply', 'output': 'out.splat', 'config': {'voxel_size': 0.05}}]}), encoding='utf-8')
            outcomes = run_batch(manifest, d / 'state.json')
            self.assertEqual(outcomes[0]['status'], 'done')
            self.assertTrue((d / 'out.splat').exists())

    def test_gui_filter_extension_and_cuda_iterations(self):
        from PySide6.QtWidgets import QApplication
        from gui.app import SplatStudio, _apply_filter_extension

        app = QApplication.instance() or QApplication([])
        self.assertEqual(_apply_filter_extension('/tmp/out', 'SPZ v3 (*.spz)'), '/tmp/out.spz')
        self.assertEqual(_apply_filter_extension('/tmp/out.ply', 'SPZ v3 (*.spz)'), '/tmp/out.ply')
        w = SplatStudio()
        self.assertEqual(w.iterations.maximum(), 5000)
        w.backend.setCurrentIndex(1)
        app.processEvents()
        self.assertEqual(w.iterations.maximum(), 100_000)
        w.close()
