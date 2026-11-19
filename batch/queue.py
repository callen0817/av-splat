from pathlib import Path
import tempfile
from av_common.queue import run_queue
from av_common.data import load_dataset
from core.rapid import RapidConfig,rapid_splat
from formats.gaussian import export_ply
from formats.legacy_splat import export_splat
from formats.spz_export import export_spz


def run(manifest,state,cancel=lambda:False,progress=lambda message:None):
    with tempfile.TemporaryDirectory(prefix='av-splat-batch-') as tmp_root:
        root=Path(tmp_root)
        def process(source,output,settings,stop,report):
            settings=dict(settings); direction=settings.pop('direction','point-to-splat')
            if direction=='splat-to-point':
                from formats.import_splats import load_splats
                from core.to_points import splat_to_points,PointConfig
                from formats.points import export_points
                with tempfile.TemporaryDirectory(dir=root) as job_dir:
                    points=splat_to_points(load_splats(source),Path(job_dir),PointConfig(**settings),stop,report)
                    export_points(output,points,stop)
                return
            if direction!='point-to-splat': raise ValueError('Unknown conversion direction.')
            data=load_dataset(source,root/'imports',stop,report)
            with tempfile.TemporaryDirectory(dir=root) as job_dir:
                scene=rapid_splat(data,Path(job_dir),RapidConfig(**settings),stop,report)
            ext=output.suffix.lower()
            if ext=='.spz': export_spz(output,scene,stop)
            elif ext=='.ply': export_ply(output,scene,stop)
            elif ext=='.splat': export_splat(output,scene,stop)
            else: raise ValueError('Rapid batch outputs must be Gaussian PLY, SPZ or .splat.')
        return run_queue(manifest,state,process,cancel,progress)
