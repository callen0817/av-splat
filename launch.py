"""Offline Gaussian processing and AV Splat desktop entry point."""
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--io-worker':
        from av_common.e57 import main as worker
        worker(sys.argv[2:]); return 0
    import argparse
    parser=argparse.ArgumentParser(description='AV Splat Rapid conversion and calibrated appearance')
    parser.add_argument('--batch'); parser.add_argument('--state');
    parser.add_argument('--direction',choices=['point-to-splat','splat-to-point'],default='point-to-splat')
    parser.add_argument('--point-method',choices=['centres','sample'],default='centres'); parser.add_argument('--samples-per-gaussian',type=int,default=4); parser.add_argument('--opacity-threshold',type=float,default=.05); parser.add_argument('--max-points',type=int,default=5000000)
    parser.add_argument('--input'); parser.add_argument('--output'); parser.add_argument('--project'); parser.add_argument('--voxel',type=float,default=.02)
    parser.add_argument('--max-gaussians',type=int,default=250000); parser.add_argument('--allow-uncolored',action='store_true')
    parser.add_argument('--calibration'); parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu'); parser.add_argument('--checkpoint'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--smoke-test',action='store_true'); args=parser.parse_args()
    if args.batch:
        from batch.queue import run
        results=run(args.batch,args.state or args.batch+'.state.json',progress=print)
        print(results)
        return 1 if any(job['status']=='failed' for job in results) else 0
    if args.input and args.direction=='splat-to-point':
        import tempfile
        from formats.import_splats import load_splats
        from core.to_points import splat_to_points,PointConfig
        from formats.points import export_points
        scene=load_splats(args.input)
        points=splat_to_points(scene,Path(tempfile.mkdtemp(prefix='av-splat-to-point-')),PointConfig(method=args.point_method,samples_per_gaussian=args.samples_per_gaussian,opacity_threshold=args.opacity_threshold,max_points=args.max_points),progress=print)
        if args.output: export_points(args.output,points)
        print(f'{len(points.points):,} derived points; original survey returns are not recovered.'); return 0
    if args.input:
        import tempfile
        from av_common.data import load_dataset
        from core.rapid import rapid_splat,RapidConfig
        from formats.gaussian import export_ply,save_project
        from formats.spz_export import export_spz
        from formats.legacy_splat import export_splat
        folder=Path(tempfile.mkdtemp(prefix='av-splat-cli-'))
        data=load_dataset(args.input,folder/'imports',progress=print)
        scene=rapid_splat(data,folder/'rapid',RapidConfig(voxel_size=args.voxel,max_gaussians=args.max_gaussians,allow_uncolored=args.allow_uncolored),progress=print)
        if args.calibration:
            from core.cameras import load_cameras
            cameras,identity=load_cameras(args.calibration,max_side=128 if args.backend=='cpu' else 1024)
            if not args.checkpoint: parser.error('--checkpoint is required for calibrated optimization')
            if args.backend=='cpu':
                from core.appearance import optimize,AppearanceConfig
                scene=optimize(scene,cameras,identity,args.checkpoint,AppearanceConfig(iterations=args.iterations),progress=print,resume=args.resume)
            else:
                from core.gpu import train,GpuConfig
                scene=train(scene,cameras,identity,args.checkpoint,GpuConfig(iterations=args.iterations),progress=print,resume=args.resume)
        if args.output:
            if args.output.lower().endswith('.spz'): export_spz(args.output,scene)
            elif args.output.lower().endswith('.ply'): export_ply(args.output,scene)
            elif args.output.lower().endswith('.splat'): export_splat(args.output,scene)
            else: parser.error('Choose Gaussian PLY, SPZ or .splat output')
        if args.project: save_project(args.project,scene)
        print(f'{len(scene.means):,} Gaussians · {scene.metadata["mode"]}'); return 0
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtWidgets import QApplication,QMessageBox
    app=QApplication(sys.argv); app.setApplicationName('AV Splat'); app.setOrganizationName('Artificial Vision')
    from av_common.theme import BrandSplash
    splash=BrandSplash(); splash.show(); splash.status('Loading Gaussian geometry engine…'); app.processEvents()
    try:
        from gui.conversion import ConversionStudio as SplatStudio
        window=SplatStudio(); window.show(); splash.finish(window)
    except Exception as error:
        if args.smoke_test: print(str(error),file=sys.stderr)
        else: QMessageBox.critical(None,'AV Splat startup error',str(error))
        return 1
    if args.smoke_test:
        def check():
            try:
                assert not window.process_button.isEnabled() and not window.export_button.isEnabled()
                window.grab().save(os.environ.get('AV_SCREENSHOT','/tmp/av-splat.png')); print('PASS AV Splat GUI startup',flush=True); app.exit(0)
            except Exception:
                import traceback; traceback.print_exc(); app.exit(1)
        QTimer.singleShot(300,check)
    return app.exec()

if __name__=='__main__': sys.exit(main())
