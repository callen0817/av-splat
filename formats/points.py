"""Derived point exports with provenance and verified coordinate quantization."""
import os
from pathlib import Path
import tempfile
import numpy as np
from av_common.data import atomic_json,check_cancel,verify_source
from av_common.exports import write_ply,cloud_fields,protect_source


def export_points(path,data,cancel=lambda:False,las_scale=.0001):
    data.validate(); protect_source(data,path); path=Path(path).resolve(); ext=path.suffix.lower()
    if ext=='.ply':
        fields=cloud_fields(data); fields['av_derived']=np.broadcast_to(np.uint8(1),(len(data.points),)); write_ply(path,fields,cancel)
    elif ext in {'.las','.laz','.xyz'}:
        fd,temp=tempfile.mkstemp(prefix='.av-points-',suffix=ext,dir=path.parent); os.close(fd)
        try:
            if ext=='.xyz':
                with open(temp,'w') as stream:
                    for start in range(0,len(data.points),100_000):
                        check_cancel(cancel); np.savetxt(stream,data.points[start:start+100_000],fmt='%.12f')
                # Validate bounded chunks without loading the whole text file.
                with open(temp) as stream:
                    for start in range(0,len(data.points),100_000):
                        check_cancel(cancel); expected=data.points[start:start+100_000]
                        actual=np.loadtxt([stream.readline() for _ in range(len(expected))],ndmin=2)
                        if not np.allclose(actual,expected,atol=1e-8,rtol=0): raise RuntimeError('XYZ coordinates failed validation.')
            else:
                import laspy
                from pyproj import CRS
                if not np.isfinite(las_scale) or las_scale<=0: raise ValueError('LAS quantization scale must be positive.')
                minimum,maximum=data.points.min(0),data.points.max(0); origin=(minimum+maximum)/2
                if np.max((maximum-minimum)/2)/las_scale>np.iinfo('int32').max-1: raise ValueError('Scene exceeds LAS integer range at this precision. Use PLY or a coarser explicit scale.')
                header=laspy.LasHeader(point_format=3,version='1.2'); header.scales=np.full(3,las_scale); header.offsets=origin
                if data.metadata.get('crs'): header.add_crs(CRS.from_user_input(data.metadata['crs']))
                for name,values in data.extras.items(): header.add_extra_dim(laspy.ExtraBytesParams(name=name,type=values.dtype))
                header.add_extra_dim(laspy.ExtraBytesParams(name='av_derived',type='uint8'))
                with laspy.open(temp,mode='w',header=header) as writer:
                    for start in range(0,len(data.points),100_000):
                        check_cancel(cancel); points=data.points[start:start+100_000]
                        record=laspy.ScaleAwarePointRecord.zeros(len(points),header=header); record.x,record.y,record.z=points.T; record.classification=np.ones(len(points),dtype='uint8')
                        if data.colors is not None:
                            rgb=np.rint(np.clip(data.colors[start:start+len(points)],0,1)*65535).astype('uint16'); record.red,record.green,record.blue=rgb.T
                        for name,values in data.extras.items(): record[name]=values[start:start+len(points)]
                        record.av_derived=np.ones(len(points),dtype='uint8'); writer.write_points(record)
                with laspy.open(temp) as reader:
                    if reader.header.point_count!=len(data.points): raise RuntimeError('Derived LAS point count changed.')
                    offset=0
                    for record in reader.chunk_iterator(100_000):
                        check_cancel(cancel); xyz=np.column_stack([record.x,record.y,record.z]); expected=data.points[offset:offset+len(record)]
                        if not np.allclose(xyz,expected,atol=las_scale*.51,rtol=0): raise RuntimeError('Derived LAS coordinates failed validation.')
                        for name,values in data.extras.items():
                            if not np.array_equal(record[name],values[offset:offset+len(record)]): raise RuntimeError('Derived LAS attribute changed.')
                        offset+=len(record)
            check_cancel(cancel); os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    else: raise ValueError('Point outputs support PLY, LAS/LAZ and XYZ. E57 station metadata cannot be reconstructed from Gaussians.')
    verify_source(data)
    atomic_json(str(path)+'.av.json',{**data.metadata,'export':str(path),'format':ext,'las_quantization_native_units':las_scale if ext in {'.las','.laz'} else None,'xyz_is_coordinates_only':ext=='.xyz'})
    return str(path)
