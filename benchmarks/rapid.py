"""Reproducible geometry/size baseline, not a production-quality claim."""
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from av_common.data import Dataset
from core.rapid import rapid_splat,RapidConfig
from formats.gaussian import export_ply
from formats.spz_export import export_spz
from core.to_points import splat_to_points,PointConfig
from _resources import ResourceMeter


def main():
    a,b=np.meshgrid(np.linspace(0,5,200),np.linspace(0,5,200)); points=np.column_stack([a.ravel(),b.ravel(),.15*np.sin(a.ravel())])
    colors=np.column_stack([a.ravel()/5,b.ravel()/5,np.full(a.size,.3)])
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        with ResourceMeter() as meter:
            start=time.perf_counter(); scene=rapid_splat(Dataset(points,colors),d/'work',RapidConfig(voxel_size=.03)); seconds=time.perf_counter()-start
            export_ply(d/'scene.ply',scene); export_spz(d/'scene.spz',scene)
            start=time.perf_counter(); centres=splat_to_points(scene,d/'centres'); centre_seconds=time.perf_counter()-start
            start=time.perf_counter(); sampled=splat_to_points(scene,d/'sampled',PointConfig(method='sample',samples_per_gaussian=4)); sample_seconds=time.perf_counter()-start
        record={'dataset':'40,000 colored points on a synthetic curved surface','source_points':len(points),'gaussians':len(scene.means),'rapid_seconds':seconds,'source_points_per_second':len(points)/seconds,
                'ply_bytes':(d/'scene.ply').stat().st_size,'spz_bytes':(d/'scene.spz').stat().st_size,'ply_to_spz_ratio':(d/'scene.ply').stat().st_size/(d/'scene.spz').stat().st_size,
                'before_comparison':'unavailable: repository contained only scaffolds','real_image_quality':'unmeasured','interactive_fps':'unmeasured','gpu_training':'unrun: CUDA unavailable','splat_to_centres_seconds':centre_seconds,'centre_points':len(centres.points),'covariance_sample_seconds':sample_seconds,'sampled_points':len(sampled.points),'resources':meter.report()}
        Path(__file__).with_name('results.json').write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))
if __name__=='__main__': main()
