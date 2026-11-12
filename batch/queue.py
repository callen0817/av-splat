from pathlib import Path
import tempfile
from av_common.queue import run_queue
from av_common.data import load_dataset
from core.rapid import RapidConfig,rapid_splat
from formats.gaussian import export_ply
from formats.spz_export import export_spz


def run(manifest,state,cancel=lambda:False,progress=lambda message:None):
    root=Path(tempfile.mkdtemp(prefix='av-splat-batch-'))
    def process(source,output,settings,stop,report):
        settings=dict(settings); direction=settings.pop('direction','point-to-splat')
        if direction=='splat-to-point':
            from formats.import_splats import load_splats
            from core.to_points import splat_to_points,PointConfig
            from formats.points import export_points
            points=splat_to_points(load_splats(source),Path(tempfile.mkdtemp(dir=root)),PointConfig(**settings),stop,report)
            export_points(output,points,stop); return
        if direction!='point-to-splat': raise ValueError('Unknown conversion direction.')
        data=load_dataset(source,root/'imports',stop,report)
        scene=rapid_splat(data,Path(tempfile.mkdtemp(dir=root)),RapidConfig(**settings),stop,report)
        if output.suffix.lower()=='.spz': export_spz(output,scene,stop)
        elif output.suffix.lower()=='.ply': export_ply(output,scene,stop)
        else: raise ValueError('Rapid batch outputs must be Gaussian PLY or SPZ.')
    return run_queue(manifest,state,process,cancel,progress)
