import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from core.rapid import GaussianScene
from core.cameras import Camera,load_cameras
from core.appearance import render,optimize,AppearanceConfig


def scene():
    return GaussianScene(np.array([[-.15,0,2.],[.15,0,2.]]),np.array([[.2,.3,.4],[.3,.2,.2]],dtype='float32'),np.full((2,3),np.log(.1),dtype='float32'),np.array([[1.,0,0,0],[1.,0,0,0]],dtype='float32'),np.full(2,2.,dtype='float32'),{'schema_version':1,'mode':'rapid'})


def cameras(target):
    result=[]
    for x in [0.,.02]:
        K=np.array([[40.,0,8.],[0,40.,8.],[0,0,1.]])
        transform=np.eye(4); transform[0,3]=x
        camera=Camera(K,transform,np.zeros((16,16,3)),str(x)).validate()
        camera.image=render(target,camera)[0].astype('float32'); result.append(camera)
    return result


class AppearanceTests(unittest.TestCase):
    def test_calibrated_fit_heldout_and_checkpoint_recovery(self):
        initial=scene(); target=scene(); target.colors=np.array([[.8,.2,.1],[.1,.7,.9]],dtype='float32')
        views=cameras(target)
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'checkpoint.npz'; config=AppearanceConfig(iterations=30,learning_rate=.8,checkpoint_every=5)
            fitted=optimize(initial,views,'calibration-v1',checkpoint,config)
            report=fitted.metadata['appearance']
            self.assertLess(report['heldout_mse'],report['initial_heldout_mse']*.2)
            np.testing.assert_array_equal(fitted.means,initial.means)
            resumed=optimize(initial,views,'calibration-v1',checkpoint,config,resume=True)
            np.testing.assert_array_equal(resumed.colors,fitted.colors)
            with self.assertRaises(ValueError): optimize(initial,views,'different-cameras',checkpoint,config,resume=True)
            self.assertTrue(fitted.metadata['geometry_fixed'])

    def test_calibration_validation_cancellation_and_limits(self):
        with self.assertRaises(ValueError): Camera(np.eye(3),np.diag([2,1,1,1]),np.zeros((16,16,3)),'').validate()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InterruptedError): optimize(scene(),cameras(scene()),'id',Path(d)/'checkpoint.npz',cancel=lambda:True)
            with self.assertRaises(MemoryError): optimize(scene(),cameras(scene()),'id',Path(d)/'checkpoint.npz',AppearanceConfig(max_gaussians=1))
            path=Path(d)/'bad.json'; path.write_text(json.dumps({'schema_version':1,'units':'metres','convention':'unknown'}))
            with self.assertRaises(ValueError): load_cameras(path)

    def test_actual_calibration_files_and_image_change_detection(self):
        from PIL import Image
        from core.cameras import verify_camera_sources
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); reference=scene(); views=cameras(reference); records=[]
            for index,camera in enumerate(views):
                image=d/f'view-{index}.png'; Image.fromarray(np.rint(np.clip(camera.image,0,1)*255).astype('uint8')).save(image)
                records.append({'image':image.name,'K':camera.K.tolist(),'world_to_camera':camera.world_to_camera.tolist(),'distortion':[0,0,0,0]})
            manifest=d/'cameras.json'; manifest.write_text(json.dumps({'schema_version':1,'convention':'opencv_world_to_camera','units':'metres','cameras':records}))
            loaded,identity=load_cameras(manifest,max_side=16)
            self.assertEqual(loaded[0].image.shape,(16,16,3)); self.assertEqual(len(identity),64)
            verify_camera_sources(loaded)
            Image.fromarray(np.zeros((16,16,3),dtype='uint8')).save(d/'view-0.png')
            with self.assertRaises(ValueError): verify_camera_sources(loaded)

    def test_interruption_and_resume_match_uninterrupted_fit(self):
        initial=scene(); target=scene(); target.colors=np.array([[.8,.2,.1],[.1,.7,.9]],dtype='float32'); views=cameras(target)
        config=AppearanceConfig(iterations=20,learning_rate=.8,checkpoint_every=5)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); stopped=[False]
            def progress(message): stopped[0]=True
            with self.assertRaises(InterruptedError): optimize(initial,views,'same-calibration',d/'partial.npz',config,cancel=lambda:stopped[0],progress=progress)
            self.assertTrue((d/'partial.npz').exists())
            resumed=optimize(initial,views,'same-calibration',d/'partial.npz',config,resume=True)
            full=optimize(initial,views,'same-calibration',d/'full.npz',config)
            np.testing.assert_allclose(resumed.colors,full.colors,atol=1e-7)
            np.testing.assert_allclose(resumed.opacity_logits,full.opacity_logits,atol=1e-7)
            self.assertEqual(resumed.metadata['appearance']['initial_heldout_mse'],full.metadata['appearance']['initial_heldout_mse'])

    def test_known_pinhole_projection_axes_and_behind_camera_rejection(self):
        one=GaussianScene(np.array([[0.,0.,2.]]),np.array([[1.,0.,0.]],dtype='float32'),np.full((1,3),np.log(.05)),np.array([[1.,0.,0.,0.]]),np.array([2.]),{'schema_version':1})
        K=np.array([[40.,0.,8.],[0.,40.,8.],[0.,0.,1.]])
        camera=Camera(K,np.eye(4),np.zeros((16,16,3)),'').validate()
        image,_=render(one,camera); row,column=np.unravel_index(np.argmax(image[:,:,0]),(16,16))
        self.assertIn(column,[7,8]); self.assertIn(row,[7,8])
        camera.world_to_camera[0,3]=.2; camera.world_to_camera[1,3]=.2
        image,_=render(one,camera); row,column=np.unravel_index(np.argmax(image[:,:,0]),(16,16))
        self.assertIn(column,[11,12]); self.assertIn(row,[11,12])
        camera.world_to_camera[2,3]=-3
        image,records=render(one,camera)
        self.assertFalse(records); self.assertEqual(float(image.sum()),0.)
