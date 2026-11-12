"""Chunked libE57 extraction in an isolated process; poses applied per scan."""
import json
import sys
from pathlib import Path
import numpy as np
import pye57
from av_common.data import allocate,atomic_json


def extract(path,folder):
    folder=Path(folder)
    with pye57.E57(path) as reader:
        count=sum(reader.get_header(i).point_count for i in range(reader.scan_count))
        if not count: raise ValueError('E57 contains no points.')
        xyz=allocate(folder,'points',(count,3))
        rgb=allocate(folder,'colors',(count,3),'float32')
        intensity=allocate(folder,'intensity',(count,),'float64')
        offset=0; scans=[]; any_colors=False; all_colors=True
        for index in range(reader.scan_count):
            header=reader.get_header(index); names=set(header.point_fields)
            cartesian=all('cartesian'+axis in names for axis in 'XYZ')
            coordinates=['cartesian'+axis for axis in 'XYZ'] if cartesian else ['sphericalRange','sphericalAzimuth','sphericalElevation']
            if not all(field in names for field in coordinates): raise ValueError('Unsupported E57 coordinates.')
            invalid='cartesianInvalidState' if cartesian else 'sphericalInvalidState'
            fields=coordinates+[key for key in ['colorRed','colorGreen','colorBlue','intensity',invalid] if key in names]
            buffers,data=None,None
            data,buffers=reader.make_buffers(fields,200_000)
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
                    color=all('color'+channel in names for channel in ['Red','Green','Blue'])
                    if color:
                        rgb[offset:end]=np.column_stack([data['color'+channel][:n][valid] for channel in ['Red','Green','Blue']])/255
                        any_colors=True
                    else: rgb[offset:end]=.5
                    intensity[offset:end]=data['intensity'][:n][valid] if 'intensity' in data else np.nan
                    offset=end
            finally: stream.close()
            all_colors &= all('color'+channel in names for channel in ['Red','Green','Blue'])
            scans.append({'has_colors':all('color'+channel in names for channel in ['Red','Green','Blue']),'index':index,'start':start,'stop':offset,'rotation_wxyz':header.rotation.tolist(),'translation':header.translation.tolist()})
        if not offset: raise ValueError('E57 contains no valid finite points.')
        xyz.flush(); rgb.flush(); intensity.flush()
        crs=reader.root['coordinateMetadata'].value() if reader.root.isDefined('coordinateMetadata') else None
        atomic_json(folder/'e57.json',{'format':'E57','count':offset,'raw_count':count,'has_colors':any_colors,'colors_complete':all_colors,'scans':scans,'crs':crs,'units':'metres'})


def main(args=None):
    args=sys.argv[1:] if args is None else args
    extract(args[0],args[1])

if __name__=='__main__': main()
