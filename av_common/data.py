"""Memory-mapped point clouds and cancellable chunked readers."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np


def check_cancel(cancel):
    if cancel(): raise InterruptedError('Processing cancelled; source data is unchanged.')


def fingerprint(path):
    path=Path(path).resolve(); stat=path.stat()
    return {'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.av-',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream: json.dump(data,stream,indent=2,allow_nan=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@dataclass
class Dataset:
    points: np.ndarray
    colors: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if self.points.ndim!=2 or self.points.shape[1]!=3 or not len(self.points):
            raise ValueError('A nonempty N × 3 point cloud is required.')
        for start in range(0,len(self.points),200_000):
            points=self.points[start:start+200_000]
            if not np.isfinite(points).all(): raise ValueError('Point coordinates contain NaN or infinity.')
        if self.colors is not None:
            if self.colors.shape!=self.points.shape: raise ValueError('RGB count must equal XYZ count.')
            for start in range(0,len(self.colors),200_000):
                color=self.colors[start:start+200_000]
                if not np.isfinite(color).all() or color.min()<0 or color.max()>1:
                    raise ValueError('RGB values must be finite and normalized to [0,1].')
        return self


def allocate(folder,name,shape,dtype='float64'):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    needed=int(np.prod(shape))*np.dtype(dtype).itemsize
    if needed>shutil.disk_usage(folder).free*.8: raise MemoryError('Insufficient scratch disk space for this dataset.')
    array=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
    if needed>=8*1024**2 and hasattr(os,'posix_fallocate'):
        import errno
        descriptor=os.open(array.filename,os.O_RDWR)
        try:
            try: os.posix_fallocate(descriptor,0,os.fstat(descriptor).st_size)
            except OSError as error:
                if error.errno not in {errno.EOPNOTSUPP,errno.ENOSYS}:
                    raise MemoryError('Cannot reserve scratch disk space for the mapped array.') from error
        finally: os.close(descriptor)
    return array


def _load_ply(path,folder,cancel):
    from plyfile import PlyData
    if Path(path).stat().st_size>512*1024**2:
        with open(path,'rb') as stream:
            if b'format ascii' in stream.read(256):
                raise ValueError('Convert large ASCII PLY to binary PLY or LAS for bounded-memory import.')
    ply=PlyData.read(path,mmap='r')
    vertex=ply['vertex'].data
    if not all(k in vertex.dtype.names for k in ['x','y','z']): raise ValueError('PLY must contain x, y and z vertex fields.')
    if not len(vertex): raise ValueError('PLY has no vertices.')
    if any(vertex.dtype[k].kind=='O' for k in vertex.dtype.names): raise ValueError('List-valued vertex attributes are not supported.')
    points=allocate(folder,'points',(len(vertex),3))
    has_color=all(k in vertex.dtype.names for k in ['red','green','blue'])
    colors=allocate(folder,'colors',(len(vertex),3),'float32') if has_color else None
    for start in range(0,len(vertex),200_000):
        check_cancel(cancel); end=min(start+200_000,len(vertex))
        points[start:end]=np.column_stack([vertex[k][start:end] for k in ['x','y','z']])
        if has_color:
            rgb=np.column_stack([vertex[k][start:end] for k in ['red','green','blue']])
            maximum=np.iinfo(rgb.dtype).max if rgb.dtype.kind in 'ui' else 1
            colors[start:end]=rgb/maximum
    extras={k:vertex[k] for k in vertex.dtype.names if k not in {'x','y','z','red','green','blue'}}
    return Dataset(points,colors,extras,{'format':'PLY','units':'metres assumed','crs':None})


def _load_xyz(path,folder,cancel):
    def rows():
        with open(path,encoding='utf-8-sig') as stream:
            while True:
                line=stream.readline(65537)
                if not line: break
                if len(line)>65536: raise ValueError('XYZ row exceeds the supported text-line budget.')
                stripped=line.strip()
                if stripped and not stripped.startswith('#'): yield stripped
    count=0; columns=None
    for line in rows():
        if count%100_000==0: check_cancel(cancel)
        width=len(line.split())
        if width not in {3,6} or (columns is not None and columns!=width): raise ValueError('XYZ requires consistent XYZ or XYZRGB rows (3 or 6 columns).')
        columns=width; count+=1
    if not count: raise ValueError('XYZ contains no points.')
    points=allocate(folder,'points',(count,3))
    colors=allocate(folder,'colors',(count,3),'float32') if columns==6 else None
    batch=[]; offset=0; color_max=0.
    def write(lines):
        nonlocal offset,color_max
        check_cancel(cancel); values=np.loadtxt(lines,ndmin=2)
        if not np.isfinite(values).all(): raise ValueError('XYZ values must be finite.')
        points[offset:offset+len(values)]=values[:,:3]
        if colors is not None:
            if values[:,3:].min()<0 or values[:,3:].max()>255: raise ValueError('XYZRGB colors must be normalized RGB or 0–255 RGB.')
            colors[offset:offset+len(values)]=values[:,3:]; color_max=max(color_max,float(values[:,3:].max()))
        offset+=len(values)
    for line in rows():
        batch.append(line)
        if len(batch)==100_000: write(batch); batch=[]
    if batch: write(batch)
    if offset!=count: raise ValueError('XYZ source changed while importing.')
    if colors is not None and color_max>1:
        for start in range(0,count,200_000): check_cancel(cancel); colors[start:start+200_000]/=255
    metadata={'format':'XYZRGB' if colors is not None else 'XYZ','units':'source units; confirm metres before geometry processing','crs':None}
    sidecar=Path(str(path)+'.av.json')
    if sidecar.exists() and sidecar.stat().st_size<16*1024**2:
        record=json.loads(sidecar.read_text())
        if record.get('count')==count:
            for key in ['crs','coordinate_frame','not_original_survey','units']:
                if key in record: metadata[key]=record[key]
    return Dataset(points,colors,{},metadata)


def _load_las(path,folder,cancel,progress):
    import laspy
    with laspy.open(path) as reader:
        count=reader.header.point_count
        if count==0: raise ValueError('LAS contains no points.')
        points=allocate(folder,'points',(count,3))
        has_color='red' in set(reader.header.point_format.dimension_names)
        colors=allocate(folder,'colors',(count,3),'float32') if has_color else None
        intensity=allocate(folder,'intensity',(count,),'uint16')
        offset=0
        for chunk in reader.chunk_iterator(200_000):
            check_cancel(cancel); end=offset+len(chunk)
            points[offset:end]=np.column_stack([chunk.x,chunk.y,chunk.z])
            intensity[offset:end]=chunk.intensity
            if has_color: colors[offset:end]=np.column_stack([chunk.red,chunk.green,chunk.blue])/65535
            offset=end; progress(f'Importing {offset:,} / {count:,} points')
        crs=reader.header.parse_crs()
        return Dataset(points,colors,{'intensity':intensity},{'format':'LAS','units':'metres assumed; confirm source CRS units','crs':crs.to_wkt() if crs else None})


def _load_e57(path,folder,cancel):
    root=Path(__file__).resolve().parent.parent
    command=[sys.executable,'--io-worker'] if getattr(sys,'frozen',False) else [sys.executable,'-m','av_common.e57']
    process=subprocess.Popen(command+[str(path),str(folder)],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        while True:
            check_cancel(cancel)
            try: stdout,stderr=process.communicate(timeout=.2); break
            except subprocess.TimeoutExpired: pass
        if process.returncode: raise RuntimeError('E57 import failed: '+stderr[-2500:])
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    metadata=json.loads((folder/'e57.json').read_text())
    n=metadata['count']
    points=np.load(folder/'points.npy',mmap_mode='r+')[:n]
    colors=np.load(folder/'colors.npy',mmap_mode='r+')[:n] if metadata['has_colors'] else None
    extras={'intensity':np.load(folder/'intensity.npy',mmap_mode='r+')[:n]}
    return Dataset(points,colors,extras,metadata)


def _load_dataset_unlocked(path,cache_root,cancel=lambda:False,progress=lambda message:None):
    path=Path(path).resolve(); stamp=fingerprint(path)
    key=hashlib.sha256(json.dumps({'cache_schema_version':1,'source':stamp},sort_keys=True).encode()).hexdigest()[:24]
    folder=Path(cache_root)/key
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'ready.json'
    # Complete retained imports can be reused without reparsing the source.
    if marker.exists():
        metadata=json.loads(marker.read_text()); n=metadata['count']
        points=np.load(folder/'points.npy',mmap_mode='r+')[:n]
        colors=np.load(folder/'colors.npy',mmap_mode='r+')[:n] if metadata['has_colors'] else None
        extras={}
        for name in metadata.get('cached_extras',[]): extras[name]=np.load(folder/('attr_'+name+'.npy'),mmap_mode='r+')[:n]
        return Dataset(points,colors,extras,metadata).validate()
    extension=path.suffix.lower()
    if extension=='.ply': data=_load_ply(path,folder,cancel)
    elif extension in {'.las','.laz'}: data=_load_las(path,folder,cancel,progress)
    elif extension=='.e57': data=_load_e57(path,folder,cancel)
    elif extension in {'.xyz','.xyzrgb'}: data=_load_xyz(path,folder,cancel)
    else: raise ValueError('Supported inputs are PLY, LAS/LAZ, E57 and XYZ/XYZRGB. Convert proprietary formats with a supported reader.')
    data.validate(); check_cancel(cancel)
    if fingerprint(path)!=stamp: raise ValueError('Source changed during import; retry with a stable file.')
    # Cache all scalar attributes, including custom PLY fields.
    for name,values in list(data.extras.items()):
        if not name.replace('_','').isalnum(): raise ValueError('Unsupported attribute name.')
        output=folder/('attr_'+name+'.npy')
        # A missing ready marker means every attribute must be rewritten;
        # an existing attribute file may belong to an interrupted prior import.
        cached=allocate(folder,'attr_'+name,(len(data.points),),values.dtype)
        for start in range(0,len(values),200_000):
            check_cancel(cancel); cached[start:start+200_000]=values[start:start+200_000]
        cached.flush(); data.extras[name]=cached
    data.metadata.update(cache_schema_version=1,source=stamp,count=len(data.points),has_colors=data.colors is not None,cached_extras=list(data.extras))
    for array in [data.points,data.colors,*data.extras.values()]:
        if hasattr(array,'flush'): array.flush()
    atomic_json(marker,data.metadata)
    return data


def verify_source(data):
    stamp=data.metadata.get('source')
    if stamp is not None and fingerprint(stamp['path'])!=stamp:
        raise ValueError('Source has changed. Reload before saving or exporting derived results.')


def load_dataset(path,cache_root,cancel=lambda:False,progress=lambda message:None):
    from filelock import FileLock,Timeout
    path=Path(path).resolve()
    lock_root=Path(cache_root)/'locks'; lock_root.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(str(path).encode()).hexdigest()[:24]+'.lock'
    lock=FileLock(lock_root/name)
    while True:
        check_cancel(cancel)
        try: lock.acquire(timeout=.2); break
        except Timeout: progress('Waiting for another import of this source to finish…')
    try:
        data=_load_dataset_unlocked(path,cache_root,cancel,progress)
        for array in [data.points,data.colors,*data.extras.values()]:
            if array is not None: array.setflags(write=False)
        return data
    finally:
        lock.release()
