"""Atomic, chunked binary PLY output with verified scalar attributes."""
import os
from pathlib import Path
import tempfile
import numpy as np
from av_common.data import check_cancel,verify_source,atomic_json

PLY_TYPES={'f8':'double','f4':'float','u1':'uchar','i1':'char','u2':'ushort','i2':'short','u4':'uint','i4':'int'}


def write_ply(path,fields,cancel=lambda:False):
    from plyfile import PlyData
    if not fields or not all(name.replace('_','').isalnum() for name in fields): raise ValueError('Invalid PLY fields.')
    n=len(next(iter(fields.values())))
    if not n or any(np.asarray(value).ndim!=1 or len(value)!=n for value in fields.values()): raise ValueError('PLY fields must have equal nonzero lengths.')
    dtype=[]
    for name,values in fields.items():
        dt=np.asarray(values).dtype
        code=dt.kind+str(dt.itemsize)
        if code not in PLY_TYPES: raise ValueError(f'PLY scalar type {dt} is not supported for {name}.')
        dtype.append((name,'<'+code))
    path=Path(path).resolve(); fd,temp=tempfile.mkstemp(prefix='.av-',suffix='.ply',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            header=['ply','format binary_little_endian 1.0','comment Artificial Vision derivative; retain original survey data',f'element vertex {n}']
            header += [f'property {PLY_TYPES[np.dtype(kind).kind+str(np.dtype(kind).itemsize)]} {name}' for name,kind in dtype]
            stream.write(('\n'.join(header)+'\nend_header\n').encode('ascii'))
            for start in range(0,n,100_000):
                check_cancel(cancel); end=min(start+100_000,n); block=np.empty(end-start,dtype=dtype)
                for name,values in fields.items(): block[name]=values[start:end]
                stream.write(block.tobytes())
        check=PlyData.read(temp,mmap='r')['vertex'].data
        if len(check)!=n: raise RuntimeError('PLY export vertex count changed.')
        for start in range(0,n,100_000):
            check_cancel(cancel); end=min(start+100_000,n)
            for name,values in fields.items():
                if not np.array_equal(check[name][start:end],values[start:end],equal_nan=True):
                    raise RuntimeError(f'PLY field {name} failed export validation.')
        del check
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def cloud_fields(data):
    fields={key:data.points[:,i] for i,key in enumerate(['x','y','z'])}
    # PLY color quantization is explicit and independent of survey coordinates.
    if data.colors is not None:
        for i,name in enumerate(['red','green','blue']): fields[name]=np.rint(np.clip(data.colors[:,i],0,1)*255).astype('uint8')
    fields.update({name:value for name,value in data.extras.items() if name not in fields})
    return fields


def protect_source(data,path):
    verify_source(data)
    source=data.metadata.get('source',{}).get('path')
    if source and (Path(source).resolve()==Path(path).resolve() or (Path(path).exists() and os.path.samefile(source,path))):
        raise ValueError('Export must use a new path; original measurements are preserved.')
