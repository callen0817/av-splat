import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from av_common.data import Dataset
from core.rapid import GaussianScene,rapid_splat,RapidConfig
from core.to_points import splat_to_points,PointConfig
from formats.gaussian import export_ply
from formats.spz_export import export_spz
from formats.legacy_splat import export_splat
from formats.import_splats import load_splats,read_splat
from formats.points import export_points


class InverseTests(unittest.TestCase):
    def scene(self,d):
        data=Dataset(np.random.default_rng(9).uniform(size=(200,3))+[500000,4500000,30],np.random.default_rng(10).uniform(size=(200,3)))
        return rapid_splat(data,d,RapidConfig(voxel_size=.005))

    def test_point_to_splat_to_point_all_formats_world_coordinates(self):
        import laspy
        from plyfile import PlyData
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); original=self.scene(d/'rapid')
            for extension,writer,tolerance in [('ply',export_ply,1e-9),('spz',export_spz,.0004),('splat',export_splat,.00001)]:
                with self.subTest(extension=extension):
                    path=d/('scene.'+extension); writer(path,original)
                    scene=load_splats(path); points=splat_to_points(scene,d/('points-'+extension))
                    np.testing.assert_allclose(points.points,original.means,atol=tolerance,rtol=0)
                    np.testing.assert_allclose(points.colors,original.colors,atol=.005,rtol=0)
                    self.assertTrue(points.metadata['not_original_survey'])
                    point_ply=d/('points-'+extension+'.ply'); export_points(point_ply,points)
                    raw=PlyData.read(point_ply)['vertex'].data
                    self.assertIn('gaussian_id',raw.dtype.names); self.assertEqual(len(raw),len(original.means))
                    output=d/('points-'+extension+'.laz'); export_points(output,points)
                    actual=laspy.read(output); np.testing.assert_allclose(actual.x,points.points[:,0],atol=.000051,rtol=0)
                    self.assertTrue(np.all(actual.av_derived==1))
                    self.assertTrue(Path(str(output)+'.av.json').exists())
                    with self.assertRaises(ValueError): export_points(path,points)

    def test_covariance_sampling_determinism_and_opacity_filter(self):
        q=Rotation.from_euler('z',90,degrees=True).as_quat()[[3,0,1,2]]
        scene=GaussianScene(np.array([[0.,0.,0.],[10.,0.,0.]]),np.array([[1.,0.,0.],[0.,1.,0.]],dtype='float32'),np.log(np.array([[.1,.01,.001],[.1,.01,.001]])),np.tile(q,(2,1)),np.array([3.,-20.]),{'schema_version':1})
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); config=PointConfig(method='sample',samples_per_gaussian=100,opacity_threshold=.1)
            a=splat_to_points(scene,d/'a',config); b=splat_to_points(scene,d/'b',config)
            np.testing.assert_array_equal(a.points,b.points); self.assertEqual(len(a.points),100)
            self.assertGreater(np.var(a.points[:,1]),np.var(a.points[:,0])*50)
            self.assertTrue(np.all(a.extras['gaussian_id']==0))
            with self.assertRaises(MemoryError): splat_to_points(scene,d/'small',PointConfig(method='sample',samples_per_gaussian=100,max_points=10))
            with self.assertRaises(InterruptedError): splat_to_points(scene,d/'cancel',cancel=lambda:True)

    def test_missing_sidecar_is_explicit_and_corrupt_legacy_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); scene=self.scene(d/'rapid'); path=d/'local.spz'; export_spz(path,scene)
            Path(str(path)+'.av.json').unlink()
            imported=load_splats(path)
            self.assertEqual(imported.metadata['coordinate_frame'],'SPZ local RUB coordinates')
            self.assertTrue(any('No AV sidecar' in message for message in imported.metadata['warnings']))
            bad=d/'broken.splat'; bad.write_bytes(b'broken')
            with self.assertRaises(ValueError): read_splat(bad)
