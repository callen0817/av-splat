"""Bidirectional point ↔ Gaussian workflows on the reconstruction studio."""
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QWidget,QVBoxLayout,QComboBox,QSpinBox,QDoubleSpinBox,QFormLayout,QFileDialog
from av_common.data import Dataset,load_dataset,fingerprint
from av_common.studio import text
from core.to_points import splat_to_points,PointConfig
from formats.import_splats import load_splats,is_gaussian_ply
from formats.points import export_points
from formats.legacy_splat import export_splat
from formats.gaussian import export_ply
from formats.spz_export import export_spz
from gui.app import SplatStudio


class ConversionStudio(SplatStudio):
    def __init__(self):
        self.point_result=None; self.gaussian_input=None
        super().__init__()
        self.direction=QComboBox(); self.direction.addItems(['Point → Splat','Splat → Point']); self.toolbar.insertWidget(1,self.direction)
        # Preserve the reconstruction controls as one panel, alongside an explicit
        # inverse-conversion panel rather than mixing their settings together.
        self.reconstruction_panel=QWidget(); reconstruction=QVBoxLayout(self.reconstruction_panel); reconstruction.setContentsMargins(0,0,0,0)
        while self.right.count()>1:
            item=self.right.takeAt(1)
            if item.widget(): reconstruction.addWidget(item.widget())
            elif item.layout():
                child=item.layout(); child.setParent(None); reconstruction.addLayout(child)
            else: reconstruction.addItem(item)
        self.right.addWidget(self.reconstruction_panel)
        self.points_panel=QWidget(); points=QVBoxLayout(self.points_panel); points.setContentsMargins(0,0,0,0)
        points.addWidget(text('DERIVED POINT EXTRACTION','eyebrow'))
        self.point_method=QComboBox(); self.point_method.addItems(['Gaussian centres','Covariance samples · synthetic']); self.point_method.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); self.point_method.setMinimumContentsLength(8)
        self.samples=QSpinBox(); self.samples.setRange(1,100); self.samples.setValue(4)
        self.point_opacity=QDoubleSpinBox(); self.point_opacity.setRange(0,1); self.point_opacity.setSingleStep(.05); self.point_opacity.setValue(.05)
        self.point_budget=QSpinBox(); self.point_budget.setRange(1,20_000_000); self.point_budget.setValue(5_000_000)
        form=QFormLayout(); form.addRow('Point method',self.point_method); form.addRow('Samples / Gaussian',self.samples); form.addRow('Minimum opacity',self.point_opacity); form.addRow('Maximum points',self.point_budget); points.addLayout(form)
        points.addWidget(text('Centres preserve the stored Gaussian positions. Sampling generates new points inside the covariance ellipsoids. Neither reconstructs original LiDAR returns, intensity or station metadata.','muted'))
        points.addWidget(text('SPZ origin and source axes are restored when an AV sidecar is present. Without it, stored local coordinates are retained and reported.','muted')); points.addStretch()
        self.right.addWidget(self.points_panel); self.points_panel.hide()
        self.direction.currentIndexChanged.connect(self.direction_changed)
        self.point_method.currentIndexChanged.connect(self.invalidate_points)
        for control in [self.samples,self.point_opacity,self.point_budget]: control.valueChanged.connect(self.invalidate_points)
        self.direction_changed(0)

    def accept_result(self,result):
        self.point_result=None; self.gaussian_input=result
        super().accept_result(result)

    def invalidate_points(self,*args):
        self.point_result=None; self.refresh_actions(); self.update_view()

    def direction_changed(self,index):
        self.reconstruction_panel.setVisible(index==0); self.points_panel.setVisible(index==1)
        self.process_button.setText('Build Rapid Splat' if index==0 else 'Extract derived points')
        self.export_button.setText('Export splats…' if index==0 else 'Export points…')
        if index==1 and self.result is not None: self.gaussian_input=self.result
        self.refresh_actions(); self.update_view()

    def refresh_actions(self,*args):
        super().refresh_actions(*args)
        if hasattr(self,'direction'):
            inverse=self.direction.currentIndex()==1
            self.direction.setEnabled(not self.busy)
            if inverse:
                self.process_button.setEnabled(not self.busy and self.gaussian_input is not None)
                self.export_button.setEnabled(not self.busy and self.point_result is not None)
                self.save_button.setEnabled(False)

    def accept_data(self,data):
        self.point_result=None; self.gaussian_input=None
        super().accept_data(data)

    def accept_gaussian_input(self,scene):
        source=scene.metadata.get('source',{}).get('source') or {'path':'Gaussian scene'}
        data=Dataset(scene.means,scene.colors,metadata={'source':source,'units':'Gaussian file units; verify metres','crs':scene.metadata.get('crs')})
        self.accept_data(data); self.gaussian_input=scene; self.result=scene; self.direction.setCurrentIndex(1)
        self.source_label.setText(f'{Path(source["path"]).name}\n{len(scene.means):,} input Gaussians\n{scene.metadata.get("coordinate_frame","Gaussian source coordinates")}')
        for warning in scene.metadata.get('warnings',[]): self.log.append(warning)
        self.update_view(); self.refresh_actions()

    def import_data(self):
        if self.busy: return
        path,_=QFileDialog.getOpenFileName(self,'Import points or Gaussian splats','','Points and splats (*.ply *.las *.laz *.e57 *.xyz *.xyzrgb *.spz *.splat *.avsplat)')
        if not path: return
        def operation(cancel,progress):
            gaussian=Path(path).suffix.lower() in {'.spz','.splat','.avsplat'} or (path.lower().endswith('.ply') and is_gaussian_ply(path))
            if gaussian: return True,load_splats(path)
            return False,load_dataset(path,self.workspace/'imports',cancel,progress)
        def accept(value):
            if value[0]: self.accept_gaussian_input(value[1])
            else: self.accept_data(value[1]); self.direction.setCurrentIndex(0)
        self.run_job(operation,accept)

    def process(self):
        if not hasattr(self,'direction') or self.direction.currentIndex()==0: return super().process()
        if self.busy or self.gaussian_input is None: return
        config=PointConfig(method='centres' if self.point_method.currentIndex()==0 else 'sample',samples_per_gaussian=self.samples.value(),opacity_threshold=self.point_opacity.value(),max_points=self.point_budget.value())
        scene=self.gaussian_input; folder=self.fresh_folder('splat-to-point-')
        def accept(data): self.point_result=data; self.update_view(); self.refresh_actions()
        self.run_job(lambda cancel,progress:splat_to_points(scene,folder,config,cancel,progress),accept)

    def update_view(self,*args):
        if not hasattr(self,'direction') or self.direction.currentIndex()==0 or self.point_result is None:
            return super().update_view(*args)
        data=self.point_result; ids=np.linspace(0,len(data.points)-1,min(100000,len(data.points)),dtype='int64')
        self.image_label.hide(); self.viewer.show(); self.viewer.set_data(data.points[ids],data.colors[ids])
        self.metrics.setText(f'{len(data.points):,} derived points · {data.metadata["derivation"]}\nNot recovered survey measurements. Colors use the Gaussian DC component.')

    def open_project(self):
        if self.busy: return
        path,_=QFileDialog.getOpenFileName(self,'Open Gaussian scene','','Gaussian scenes (*.avsplat *.ply *.spz *.splat)')
        if path: self.run_job(lambda cancel,progress:load_splats(path),self.accept_gaussian_input)

    def export(self):
        if self.busy: return
        if self.direction.currentIndex()==1:
            if self.point_result is None: return
            path,_=QFileDialog.getSaveFileName(self,'Export derived points','','Point PLY (*.ply);;LAS (*.las);;LAZ (*.laz);;XYZ coordinates (*.xyz)')
            if path:
                data=self.point_result; self.run_job(lambda cancel,progress:export_points(path,data,cancel),lambda value:self.log.append('Validated derived-point export: '+value))
        elif self.result is not None:
            path,_=QFileDialog.getSaveFileName(self,'Export Gaussian splats','','Gaussian PLY (*.ply);;SPZ v3 (*.spz);;Legacy web splat (*.splat)')
            if path:
                scene=self.result; axes=self.axes.currentText().split(' ')[0]
                def operation(cancel,progress):
                    if path.lower().endswith('.spz'): return export_spz(path,scene,cancel,axes)
                    if path.lower().endswith('.splat'): return export_splat(path,scene,cancel,axes)
                    return export_ply(path,scene,cancel)
                self.run_job(operation,lambda value:self.log.append('Validated Gaussian export: '+value))
