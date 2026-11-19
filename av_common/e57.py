"""Chunked libE57 extraction in an isolated process; poses applied per scan."""
import sys
from pathlib import Path
import numpy as np
import pye57
from av_common.data import allocate,atomic_json


def _color_limits(header):
    node=getattr(header,'node',None)
    if node is None or not hasattr(node,'isDefined') or not node.isDefined('colorLimits'): return None
    limits=node['colorLimits']
    lows=[]; highs=[]
    for channel in ['Red','Green','Blue']:
        lo_key=f'color{channel}Minimum'; hi_key=f'color{channel}Maximum'
        if not limits.isDefined(hi_key): return None
        lo=float(limits[lo_key].value()) if limits.isDefined(lo_key) else 0.0
        hi=float(limits[hi_key].value())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: return None
        lows.append(lo); highs.append(hi)
    return np.asarray(lows,dtype='float64'),np.asarray(highs,dtype='float64')


def normalize_e57_colors(raw_rgb,header=None):
    raw=np.asarray(raw_rgb)
    if not len(raw): return np.empty((0,3),dtype='float32')
    limits=_color_limits(header) if header is not None else None
    if limits is not None:
        lo,hi=limits
        return np.clip((raw.astype('float64')-lo)/(hi-lo),0.0,1.0).astype('float32')
    if raw.dtype.kind in 'ui':
        scale=float(np.iinfo(raw.dtype).max)
        return np.clip(raw.astype('float64')/scale,0.0,1.0).astype('float32')
    values=raw.astype('float64')
    finite=values[np.isfinite(values)]
    peak=float(finite.max()) if finite.size else 0.0
    if peak<=1.0: scale=1.0
    elif peak<=255.0: scale=255.0
    else: scale=65535.0
    return np.clip(values/scale,0.0,1.0).astype('float32')


def extract(path,folder):
    folder=Path(folder)
    with pye57.E57(path) as reader:
        headers=[reader.get_header(i) for i in range(reader.scan_count)]
        count=sum(header.point_count for header in headers)
        if not count: raise ValueError('E57 contains no points.')
        any_scan_has_color=any(all('color'+channel in set(header.point_fields) for channel in ['Red','Green','Blue']) for header in headers)
        xyz=allocate(folder,'points',(count,3))
        rgb=allocate(folder,'colors',(count,3),'float32') if any_scan_has_color else None
        intensity=allocate(folder,'attr_intensity',(count,),'float64')
        offset=0; scans=[]; any_colors=False; all_colors=True
        for index,header in enumerate(headers):
            names=set(header.point_fields)
            cartesian=all('cartesian'+axis in names for axis in 'XYZ')
            coordinates=['cartesian'+axis for axis in 'XYZ'] if cartesian else ['sphericalRange','sphericalAzimuth','sphericalElevation']
            if not all(field in names for field in coordinates): raise ValueError('Unsupported E57 coordinates.')
            invalid='cartesianInvalidState' if cartesian else 'sphericalInvalidState'
            color=all('color'+channel in names for channel in ['Red','Green','Blue'])
            color_fields=['colorRed','colorGreen','colorBlue'] if color else []
            non_color_fields=coordinates+[key for key in ['intensity',invalid] if key in names]
            if color and hasattr(reader,'image_file') and hasattr(pye57,'libe57'):
                data,buffers=reader.make_buffers(non_color_fields,200_000)
                for field in color_fields:
                    arr=np.empty(200_000,dtype='float64')
                    buf=pye57.libe57.SourceDestBuffer(reader.image_file,field,arr,200_000,True,True)
                    data[field]=arr; buffers.append(buf)
            else:
                data,buffers=reader.make_buffers(non_color_fields+color_fields,200_000)
            stream=header.points.reader(buffers); start=offset
            try:
                while True:
                    n=stream.read()
                    if not n: break
                    positions=np.column_stack([data[key][:n] for key in coordinates])
                    if not cartesian:
                        from pye57.e57 import convert_spherical_to_cartesian
                        positions=convert_spherical_to_cartesian(positions)
                    valid=np.isfinite(positions).all(axis=1)
                    if invalid in data: valid &= data[invalid][:n]==0
                    positions=positions[valid]
                    if header.has_pose(): positions=reader.to_global(positions,header.rotation,header.translation)
                    end=offset+len(positions); xyz[offset:end]=positions
                    if rgb is not None:
                        if color:
                            raw_rgb=np.column_stack([data['color'+channel][:n][valid] for channel in ['Red','Green','Blue']])
                            rgb[offset:end]=normalize_e57_colors(raw_rgb,header)
                            any_colors=True
                        else: rgb[offset:end]=.5
                    intensity[offset:end]=data['intensity'][:n][valid] if 'intensity' in data else np.nan
                    offset=end
            finally: stream.close()
            all_colors &= color
            scans.append({'has_colors':color,'index':index,'start':start,'stop':offset,'rotation_wxyz':header.rotation.tolist(),'translation':header.translation.tolist()})
        if not offset: raise ValueError('E57 contains no valid finite points.')
        xyz.flush(); intensity.flush()
        if rgb is not None: rgb.flush()
        crs=reader.root['coordinateMetadata'].value() if reader.root.isDefined('coordinateMetadata') else None
        atomic_json(folder/'e57.json',{'format':'E57','count':offset,'raw_count':count,'has_colors':any_colors,'colors_complete':all_colors,'scans':scans,'crs':crs,'units':'metres'})


def main(args=None):
    args=sys.argv[1:] if args is None else args
    extract(args[0],args[1])

if __name__=='__main__': main()
