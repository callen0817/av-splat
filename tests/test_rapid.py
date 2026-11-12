import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from av_common.data import Dataset
from core.rapid import rapid_splat,RapidConfig
from formats.gaussian import export_ply,read_ply,save_project,load_project


class RapidTests(unittest.TestCase):
    def data(self):
        a,b=np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20))
        points=np.column_stack([a.ravel(),b.ravel(),np.zeros(a.size)])
        colors=np.column_stack([a.ravel(),b.ravel(),np.full(a.size,.3)])
        return Dataset(points,colors)

    def test_covariance_orientation_and_gaussian_ply_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); scene=rapid_splat(self.data(),d/'work',RapidConfig(voxel_size=.01))
            rotation=Rotation.from_quat(scene.quaternions[:,[1,2,3,0]]).as_matrix()
            covariance=rotation@np.array([np.diag(s**2) for s in np.exp(scene.log_scales)])@np.transpose(rotation,(0,2,1))
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance)>0))
            self.assertLess(np.mean(covariance[:,2,2]),np.mean(covariance[:,0,0])*.1)
            self.assertFalse(scene.metadata['optimized_with_images'])
            output=d/'scene.ply'; export_ply(output,scene); actual=read_ply(output)
            np.testing.assert_allclose(actual.means,scene.means,atol=0); np.testing.assert_allclose(actual.colors,scene.colors,atol=1e-6)
            np.testing.assert_allclose(actual.log_scales,scene.log_scales)
            np.testing.assert_allclose(np.abs(np.sum(actual.quaternions*scene.quaternions,axis=1)),1,atol=1e-6)
            snapshot=d/'project.avsplat'; save_project(snapshot,scene); restored=load_project(snapshot)
            np.testing.assert_array_equal(restored.means,scene.means)

    def test_budget_cancellation_and_uncolored_input(self):
        with tempfile.TemporaryDirectory() as d:
            data=self.data()
            with self.assertRaises(InterruptedError): rapid_splat(data,d,cancel=lambda:True)
            with self.assertRaises(ValueError): rapid_splat(Dataset(data.points),d)
            scene=rapid_splat(Dataset(data.points),d,RapidConfig(allow_uncolored=True,max_gaussians=100))
            self.assertLessEqual(len(scene.means),100); np.testing.assert_allclose(scene.colors,.5)
            with self.assertRaises(ValueError): RapidConfig(opacity=1).validate()
