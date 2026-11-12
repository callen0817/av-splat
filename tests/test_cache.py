import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from av_common.data import load_dataset,allocate
from av_common.exports import write_ply

class CacheTests(unittest.TestCase):
    def test_incomplete_import_rewrites_partial_attributes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); points=np.arange(60,dtype=float).reshape(-1,3); values=np.arange(20,dtype='uint16')
            fields={name:points[:,i] for i,name in enumerate('xyz')}; fields['intensity']=values
            source=d/'source.ply'; write_ply(source,fields); load_dataset(source,d/'cache')
            marker=next((d/'cache').glob('*/ready.json')); marker.unlink()
            attribute=marker.parent/'attr_intensity.npy'
            damaged=np.load(attribute,mmap_mode='r+'); damaged[:]=0; damaged.flush(); del damaged
            recovered=load_dataset(source,d/'cache')
            np.testing.assert_array_equal(recovered.extras['intensity'],values)
            with self.assertRaises(ValueError): recovered.points[0,0]=999
            with self.assertRaises(ValueError): recovered.extras['intensity'][0]=999

    def test_insufficient_scratch_space_fails_before_array_creation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('av_common.data.shutil.disk_usage',return_value=SimpleNamespace(free=10)):
                with self.assertRaises(MemoryError): allocate(d,'too_large',(100,3))
            self.assertFalse((Path(d)/'too_large.npy').exists())

    def test_streamed_xyz_and_rgb_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); points=np.array([[1.,2.,3.],[4.,5.,6.]])
            xyz=d/'points.xyz'; np.savetxt(xyz,points)
            data=load_dataset(xyz,d/'cache'); np.testing.assert_array_equal(data.points,points); self.assertIsNone(data.colors)
            rgb=d/'colors.xyzrgb'; np.savetxt(rgb,np.column_stack([points,[[255,0,128],[0,255,64]]]))
            colored=load_dataset(rgb,d/'cache'); np.testing.assert_allclose(colored.colors,[[1,0,128/255],[0,1,64/255]],atol=1e-7)
