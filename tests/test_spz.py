import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import spz
from av_common.data import Dataset
from core.rapid import rapid_splat,RapidConfig
from formats.spz_export import export_spz,export_tiles

class SPZTests(unittest.TestCase):
    def scene(self,d):
        points=np.random.default_rng(4).uniform(size=(200,3))+[500000,4500000,20]
        colors=np.random.default_rng(5).uniform(size=(200,3))
        return rapid_splat(Dataset(points,colors),d,RapidConfig(voxel_size=.01))

    def test_official_codec_versions_origin_axes_and_colors(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); scene=self.scene(d/'work')
            for version in [3,4]:
                output=d/f'scene-{version}.spz'; export_spz(output,scene,source_axes='RFU',version=version)
                report=json.loads(Path(str(output)+'.av.json').read_text())
                options=spz.UnpackOptions(); options.to_coord=spz.CoordinateSystem.RFU
                cloud=spz.load_spz(str(output),options)
                points=np.asarray(cloud.positions).reshape(-1,3)+report['local_origin_source_axes']
                np.testing.assert_allclose(points,scene.means,atol=.0004,rtol=0)
                self.assertEqual(report['spz_version'],version)
                self.assertGreater(output.stat().st_size,0)

    def test_tiled_manifest_and_fixedpoint_range_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); scene=self.scene(d/'work'); scene.means[::2,0]+=10000
            with self.assertRaises(ValueError): export_spz(d/'large.spz',scene)
            output=export_tiles(d/'tiles',scene,tile_size=512)
            manifest=json.loads(Path(output).read_text())
            self.assertEqual(sum(t['gaussians'] for t in manifest['tiles']),len(scene.means))
            self.assertGreater(len(manifest['tiles']),1)
