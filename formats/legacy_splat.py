"""Legacy 32-byte web .splat adapter with explicit source-to-RUB conversion."""
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation
from av_common.data import atomic_json,check_cancel
from formats.gaussian import protect


def axis_matrix(source_axes):
    matrices={'RFU':[[1,0,0],[0,0,1],[0,-1,0]],'RDF':[[1,0,0],[0,-1,0],[0,0,-1]],'RUB':np.eye(3)}
    if source_axes not in matrices: raise ValueError('Choose RFU, RDF or RUB source axes.')
    return np.asarray(matrices[source_axes],dtype=float)


def export_splat(path,scene,cancel=lambda:False,source_axes='RFU'):
    scene.validate(); protect(scene,path)
    if scene.metadata.get('source_sh_degree',0)>0: raise ValueError('Legacy .splat export cannot preserve higher-order SH; use explicit point conversion instead.')
    transform=axis_matrix(source_axes); origin=(scene.means.min(0)+scene.means.max(0))/2
    local=(scene.means-origin)@transform.T
    rotations=Rotation.from_quat(scene.quaternions[:,[1,2,3,0]]).as_matrix()
    q=Rotation.from_matrix(transform[None]@rotations).as_quat()[:,[3,0,1,2]]
    dtype=np.dtype([('position','<f4',(3,)),('scale','<f4',(3,)),('rgba','u1',(4,)),('rotation','u1',(4,))])
    path=Path(path).resolve(); fd,temp=tempfile.mkstemp(prefix='.av-',suffix='.splat',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            for start in range(0,len(scene.means),100_000):
                check_cancel(cancel); end=min(start+100_000,len(scene.means)); block=np.empty(end-start,dtype)
                block['position']=local[start:end]; block['scale']=np.exp(scene.log_scales[start:end]); block['rgba'][:,:3]=np.rint(scene.colors[start:end]*255).astype('uint8')
                block['rgba'][:,3]=np.rint(expit(scene.opacity_logits[start:end])*255).astype('uint8'); block['rotation']=np.clip(np.rint(q[start:end]*128+128),0,255).astype('uint8')
                stream.write(block.tobytes())
        raw=np.memmap(temp,dtype=dtype,mode='r')
        if len(raw)!=len(scene.means) or not np.array_equal(raw['position'],local.astype('float32')) or not np.isfinite(raw['scale']).all(): raise RuntimeError('Legacy .splat record validation failed.')
        error=float(np.max(np.abs(raw['position'].astype('float64')-local)))
        del raw; check_cancel(cancel); os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    atomic_json(str(path)+'.av.json',{**scene.metadata,'export_format':'legacy 32-byte .splat','source_axes':source_axes,'stored_axes':'RUB','local_origin_source_axes':origin.tolist(),'quaternion_order':'wxyz','max_position_rounding_m':error,'sh_degree':0,
               'restore_source_coordinates':'Apply the inverse source-axis conversion to stored positions, then add local_origin_source_axes.'})
    return str(path)
