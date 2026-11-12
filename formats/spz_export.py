"""Niantic SPZ adapter: pinned official codec, explicit axis and origin metadata."""
import gzip
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import tempfile
import numpy as np
from scipy.special import expit
from av_common.data import atomic_json,check_cancel
from core.rapid import GaussianScene
from formats.gaussian import protect,SH_C0


def available(): return importlib.util.find_spec('spz') is not None


def export_spz(path,scene,cancel=lambda:False,source_axes='RFU',version=3):
    scene.validate(); protect(scene,path)
    if scene.metadata.get('source_sh_degree',0)>0: raise ValueError('Higher-order SH re-export would discard appearance. Use explicit point conversion instead.')
    if not available(): raise RuntimeError('Install requirements-spz.txt for the pinned official Niantic codec. Gaussian PLY remains available without it.')
    if version not in {3,4} or source_axes not in {'RFU','RDF','RUB'}: raise ValueError('Choose SPZ version 3/4 and an explicit RFU/RDF/RUB source axis convention.')
    import spz
    origin=(scene.means.min(0)+scene.means.max(0))/2
    local=scene.means-origin
    if np.max(np.abs(local))*4096>=2**23-1: raise ValueError('Scene exceeds SPZ fixed-point range. Use tiled SPZ export or Gaussian PLY.')
    if scene.log_scales.min()<-10 or scene.log_scales.max()>5.9: raise ValueError('Gaussian scales exceed the SPZ quantization range; use PLY to retain them.')
    cloud=spz.GaussianCloud(); cloud.sh_degree=0; cloud.antialiased=False
    cloud.positions=local.astype('float32').ravel(); cloud.scales=scene.log_scales.astype('float32').ravel()
    cloud.rotations=scene.quaternions[:,[1,2,3,0]].astype('float32').ravel()
    cloud.alphas=scene.opacity_logits.astype('float32'); cloud.colors=((scene.colors-.5)/SH_C0).astype('float32').ravel()
    options=spz.PackOptions(); options.version=version; options.from_coord=getattr(spz.CoordinateSystem,source_axes)
    path=Path(path).resolve(); fd,temp=tempfile.mkstemp(prefix='.av-',suffix='.spz',dir=path.parent); os.close(fd)
    try:
        check_cancel(cancel)
        if not spz.save_spz(cloud,options,temp): raise RuntimeError('Official SPZ encoder failed.')
        if version==3:
            with gzip.open(temp,'rb') as stream: header=stream.read(16)
        else:
            with open(temp,'rb') as stream: header=stream.read(16)
        magic,actual_version=struct.unpack('<II',header[:8])
        if magic!=0x5053474e or actual_version!=version: raise RuntimeError('SPZ file magic/version mismatch.')
        unpack=spz.UnpackOptions(); unpack.to_coord=getattr(spz.CoordinateSystem,source_axes)
        decoded=spz.load_spz(temp,unpack)
        if decoded.num_points!=len(scene.means): raise RuntimeError('SPZ point count changed.')
        actual=np.asarray(decoded.positions).reshape(-1,3)
        if not np.allclose(actual,local,atol=1/4096+.00015,rtol=0): raise RuntimeError('SPZ position quantization exceeded the expected bound.')
        colors=np.asarray(decoded.colors).reshape(-1,3)*SH_C0+.5
        if not np.allclose(colors,scene.colors,atol=.005,rtol=0): raise RuntimeError('SPZ color quantization exceeded the expected bound.')
        if not np.allclose(np.asarray(decoded.scales).reshape(-1,3),scene.log_scales,atol=.033,rtol=0): raise RuntimeError('SPZ log-scales exceeded quantization tolerance.')
        if not np.allclose(expit(np.asarray(decoded.alphas)),expit(scene.opacity_logits),atol=.0021,rtol=0): raise RuntimeError('SPZ opacity exceeded quantization tolerance.')
        rotations=np.asarray(decoded.rotations).reshape(-1,4)[:,[3,0,1,2]]
        if not np.all(np.abs(np.sum(rotations*scene.quaternions,axis=1))>.999): raise RuntimeError('SPZ orientation failed validation.')
        check_cancel(cancel); os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    report={**scene.metadata,'export_format':'Niantic SPZ','spz_version':version,'source_axes':source_axes,'stored_axes':'RUB by SPZ convention','local_origin_source_axes':origin.tolist(),'position_fractional_bits':12,
            'restore_source_coordinates':'decode to source_axes, then add local_origin_source_axes; retain this sidecar','codec_source':'https://github.com/nianticlabs/spz','codec_commit':'855173912dee2a48cb539657f46c4ade775f48c9'}
    atomic_json(str(path)+'.av.json',report)
    return str(path)


def export_tiles(directory,scene,cancel=lambda:False,source_axes='RFU',tile_size=512.):
    scene.validate()
    if not np.isfinite(tile_size) or not 1<=tile_size<=1024: raise ValueError('SPZ tile size must be 1–1024 metres.')
    destination=Path(directory).resolve()
    if destination.exists(): raise ValueError('Choose a new tile directory; existing projects are never replaced.')
    temporary=Path(tempfile.mkdtemp(prefix='.av-tiles-',dir=destination.parent))
    try:
        keys=np.floor(scene.means/tile_size).astype('int64'); unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        tiles=[]
        for i,key in enumerate(unique):
            check_cancel(cancel); mask=inverse==i
            tile=GaussianScene(scene.means[mask],scene.colors[mask],scene.log_scales[mask],scene.quaternions[mask],scene.opacity_logits[mask],scene.metadata)
            name=f'tile_{i:06d}.spz'; export_spz(temporary/name,tile,cancel,source_axes)
            tiles.append({'file':name,'gaussians':int(mask.sum()),'minimum_source':tile.means.min(0).tolist(),'maximum_source':tile.means.max(0).tolist(),'metadata':name+'.av.json'})
        atomic_json(temporary/'manifest.avspz.json',{'schema_version':1,'source_axes':source_axes,'tile_size_m':tile_size,'gaussians':len(scene.means),'tiles':tiles,'source':scene.metadata.get('source')})
        check_cancel(cancel); os.replace(temporary,destination)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return str(destination/'manifest.avspz.json')
