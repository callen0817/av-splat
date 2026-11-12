"""Gaussian import adapters. Point conversion uses direction-neutral DC RGB."""
import gzip
import json
from pathlib import Path
import struct
import numpy as np
from scipy.special import logit
from av_common.data import fingerprint
from core.rapid import GaussianScene
from formats.gaussian import SH_C0


def is_gaussian_ply(path):
    fields=set()
    with open(path,'rb') as stream:
        for _ in range(2048):
            line=stream.readline(4096)
            if not line: break
            if line.strip()==b'end_header': break
            pieces=line.split()
            if len(pieces)==3 and pieces[0]==b'property': fields.add(pieces[-1])
    return {b'f_dc_0',b'scale_0',b'rot_0',b'opacity'}.issubset(fields)


def _sidecar(path):
    sidecar=Path(str(path)+'.av.json')
    if not sidecar.exists(): return {}
    if sidecar.stat().st_size>16*1024**2: raise ValueError('Splat sidecar exceeds the metadata budget.')
    return json.loads(sidecar.read_text())


def read_spz(path,restore_origin=True):
    import spz
    path=Path(path).resolve()
    if path.stat().st_size>256*1024**2: raise MemoryError('SPZ exceeds the 256 MiB import budget. Import a tile.')
    with open(path,'rb') as stream: prefix=stream.read(16)
    if prefix[:2]==b'\x1f\x8b':
        with gzip.open(path,'rb') as stream:
            prefix=stream.read(16)
            # Reject gzip bombs before the native decoder reads the full payload.
            count=16
            while True:
                block=stream.read(1024**2)
                if not block: break
                count+=len(block)
                if count>256*1024**2: raise MemoryError('Decompressed SPZ exceeds the 256 MiB budget.')
    if len(prefix)<16: raise ValueError('Truncated SPZ header.')
    magic,version,n=struct.unpack('<III',prefix[:12])
    if magic!=0x5053474e or version not in {1,2,3,4} or not 1<=n<=1_000_000:
        raise ValueError('Unsupported SPZ version or point count; import a valid tile with at most one million Gaussians.')
    report=_sidecar(path); axes=report.get('source_axes','RUB') if restore_origin else 'RUB'
    if axes not in {'RFU','RDF','RUB'}: raise ValueError('Unknown source-axis metadata.')
    options=spz.UnpackOptions(); options.to_coord=getattr(spz.CoordinateSystem,axes)
    cloud=spz.load_spz(str(path),options)
    if cloud.num_points!=n: raise ValueError('SPZ decode/count validation failed.')
    means=np.asarray(cloud.positions,dtype='float64').reshape(-1,3)
    origin=np.asarray(report.get('local_origin_source_axes',[0,0,0]),dtype=float) if restore_origin else np.zeros(3)
    if origin.shape!=(3,) or not np.isfinite(origin).all(): raise ValueError('Invalid SPZ origin metadata.')
    means+=origin
    metadata={'schema_version':1,'mode':'imported SPZ','source':{'source':fingerprint(path),'format':'Gaussian SPZ'},'source_axes':axes,'coordinate_frame':'restored source coordinates' if report and restore_origin else 'SPZ local RUB coordinates',
              'restored_origin':origin.tolist(),'source_sh_degree':int(cloud.sh_degree),'spz_version':version,'crs':report.get('source',{}).get('crs') if report and restore_origin else None,
              'warnings':['Gaussian-derived data is not original survey geometry. RGB is the direction-neutral spherical-harmonic DC component.']}
    if not report: metadata['warnings'].append('No AV sidecar: original origin, units and georeferencing cannot be recovered; points remain in stored local RUB coordinates.')
    colors=np.clip(np.asarray(cloud.colors).reshape(-1,3)*SH_C0+.5,0,1)
    return GaussianScene(means,colors,np.asarray(cloud.scales).reshape(-1,3),np.asarray(cloud.rotations).reshape(-1,4)[:,[3,0,1,2]],np.clip(np.asarray(cloud.alphas),-30,30),metadata).validate()


def read_splat(path):
    path=Path(path).resolve()
    if path.stat().st_size==0 or path.stat().st_size%32: raise ValueError('Legacy .splat records must be exactly 32 bytes each.')
    n=path.stat().st_size//32
    if n>1_000_000: raise MemoryError('Legacy .splat import is limited to one million Gaussians per tile.')
    dtype=np.dtype([('position','<f4',(3,)),('scale','<f4',(3,)),('rgba','u1',(4,)),('rotation','u1',(4,))])
    raw=np.memmap(path,dtype=dtype,mode='r')
    if not np.isfinite(raw['scale']).all() or np.any(raw['scale']<=0): raise ValueError('Invalid legacy Gaussian scales.')
    q=(raw['rotation'].astype('float64')-128)/128
    norm=np.linalg.norm(q,axis=1)
    if np.any((norm<.25)|(norm>1.5)): raise ValueError('Invalid legacy wxyz quaternion encoding.')
    q/=norm[:,None]
    report=_sidecar(path); origin=np.asarray(report.get('local_origin_source_axes',[0,0,0]),float)
    if origin.shape!=(3,) or not np.isfinite(origin).all(): raise ValueError('Invalid legacy origin metadata.')
    positions=raw['position'].astype('float64')
    if report.get('stored_axes')=='RUB':
        from formats.legacy_splat import axis_matrix
        from scipy.spatial.transform import Rotation
        transform=axis_matrix(report['source_axes'])
        positions=positions@transform
        matrices=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        q=Rotation.from_matrix(transform.T[None]@matrices).as_quat()[:,[3,0,1,2]]
    metadata={'schema_version':1,'mode':'imported legacy .splat','source':{'source':fingerprint(path),'format':'legacy .splat'},'coordinate_frame':'restored source coordinates' if report else 'legacy file coordinates; axes/units unspecified','source_axes':report.get('source_axes','unspecified'),
              'warnings':['Decoded Gaussian centres/scales are visualization data, not recovered LiDAR measurements.'],'crs':report.get('source',{}).get('crs')}
    return GaussianScene(positions+origin,raw['rgba'][:,:3].astype('float32')/255,np.log(raw['scale']).copy(),q.astype('float32'),logit(np.clip(raw['rgba'][:,3].astype('float64')/255,1e-12,1-1e-12)).astype('float32'),metadata).validate()


def load_splats(path):
    extension=Path(path).suffix.lower()
    if extension=='.spz': return read_spz(path)
    if extension=='.splat': return read_splat(path)
    if extension=='.avsplat':
        from formats.gaussian import load_project
        return load_project(path)
    if extension=='.ply':
        from formats.gaussian import read_ply
        return read_ply(path)
    raise ValueError('Supported Gaussian inputs are PLY, SPZ, .splat and .avsplat.')
