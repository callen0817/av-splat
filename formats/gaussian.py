"""Graphdeco-compatible degree-zero Gaussian PLY and explicit project snapshots."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from av_common.data import atomic_json,check_cancel,fingerprint
from av_common.exports import write_ply
from core.rapid import GaussianScene

SH_C0=.28209479177387814


def protect(scene,path):
    source=scene.metadata.get('source',{}).get('source',{})
    if source and fingerprint(source['path'])!=source: raise ValueError('Original source changed; regenerate or restore against the original data.')
    if source and (Path(source['path']).resolve()==Path(path).resolve() or (Path(path).exists() and os.path.samefile(source['path'],path))):
        raise ValueError('Do not overwrite original survey data.')


def export_ply(path,scene,cancel=lambda:False):
    scene.validate(); protect(scene,path)
    if scene.metadata.get('source_sh_degree',0)>0: raise ValueError('Higher-order SH Gaussian re-export is not supported. Use explicit Splat → Point conversion for DC-color point extraction.')
    n=len(scene.means); zeros=np.zeros(n,dtype='float32')
    fields={k:scene.means[:,i].astype('float64') for i,k in enumerate(['x','y','z'])}
    fields.update(nx=zeros,ny=zeros,nz=zeros)
    for i in range(3): fields[f'f_dc_{i}']=((scene.colors[:,i]-.5)/SH_C0).astype('float32')
    fields['opacity']=scene.opacity_logits.astype('float32')
    for i in range(3): fields[f'scale_{i}']=scene.log_scales[:,i].astype('float32')
    for i in range(4): fields[f'rot_{i}']=scene.quaternions[:,i].astype('float32')
    write_ply(path,fields,cancel)
    report={**scene.metadata,'export_format':'Gaussian PLY','sh_degree':0,'coordinate_frame':'original source coordinates','color_encoding':'degree-zero spherical harmonics; f_dc=(RGB-0.5)/C0','scale_encoding':'natural logarithm','opacity_encoding':'logit','quaternion_order':'wxyz'}
    atomic_json(str(path)+'.av.json',report)
    return str(path)


def read_ply(path):
    from plyfile import PlyData
    fields=PlyData.read(path,mmap='r')['vertex'].data
    if len(fields)>1_000_000: raise MemoryError('Gaussian PLY exceeds the one-million-primitive interactive budget. Load a tile or lower LOD.')
    required=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    if not all(k in fields.dtype.names for k in required): raise ValueError('Not a degree-zero Gaussian PLY.')
    means=np.column_stack([fields[k] for k in ['x','y','z']])
    colors=np.clip(np.column_stack([fields[f'f_dc_{i}'] for i in range(3)])*SH_C0+.5,0,1)
    scales=np.column_stack([fields[f'scale_{i}'] for i in range(3)])
    q=np.column_stack([fields[f'rot_{i}'] for i in range(4)])
    from formats.import_splats import _sidecar
    report=_sidecar(path)
    rest=[name for name in fields.dtype.names if name.startswith('f_rest_')]
    if len(rest) not in {0,9,24,45,72}: raise ValueError('Invalid spherical-harmonic coefficient count in Gaussian PLY.')
    degree=int(round(np.sqrt(len(rest)/3+1)-1)) if rest else 0
    metadata={'schema_version':1,'mode':'imported Gaussian PLY','source':{'source':fingerprint(path),'format':'Gaussian PLY'},'source_file':str(Path(path).resolve()),'source_sh_degree':degree,'crs':report.get('source',{}).get('crs'),'coordinate_frame':'source Gaussian PLY coordinates','warnings':['Point conversion uses direction-neutral DC RGB; higher-order view-dependent appearance is not recovered as measured point colors.']}
    return GaussianScene(means,colors,scales,q,np.asarray(fields['opacity']),metadata).validate()


def save_project(path,scene,cancel=lambda:False):
    scene.validate(); protect(scene,path)
    if scene.metadata.get('source_sh_degree',0)>0: raise ValueError('This project format currently retains degree-zero appearance only; use explicit point conversion for higher-order input.')
    path=Path(path); fd,temp=tempfile.mkstemp(prefix='.av-splat-',suffix='.avsplat',dir=path.resolve().parent); os.close(fd)
    try:
        check_cancel(cancel)
        with open(temp,'wb') as stream:
            np.savez_compressed(stream,means=scene.means,colors=scene.colors,log_scales=scene.log_scales,quaternions=scene.quaternions,opacity_logits=scene.opacity_logits,metadata=np.array(json.dumps(scene.metadata,allow_nan=False)))
        check_cancel(cancel); os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def load_project(path):
    import zipfile
    # Prevent unreasonable decompression allocations from an untrusted snapshot.
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist())>256*1024**2: raise ValueError('Snapshot exceeds the 256 MiB restore budget.')
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata']))
        if metadata.get('schema_version')!=1: raise ValueError('Unsupported Gaussian project version.')
        scene=GaussianScene(*(np.array(saved[k]) for k in ['means','colors','log_scales','quaternions','opacity_logits']),metadata).validate()
    return scene
