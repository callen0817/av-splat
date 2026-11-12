import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QComboBox,QDoubleSpinBox,QSpinBox,QFormLayout,QCheckBox,QFileDialog,QLineEdit,QLabel,QMessageBox)
from av_common.studio import Studio,text
from av_common.data import Dataset,fingerprint
from core.rapid import rapid_splat,RapidConfig
from core.cameras import load_cameras
from core.appearance import optimize,AppearanceConfig,render
from core.gpu import train,GpuConfig
from formats.gaussian import export_ply,save_project,load_project,read_ply
from formats.spz_export import export_spz,export_tiles,available


class SplatStudio(Studio):
    def __init__(self):
        super().__init__('AV Splat','Reconstruction Studio')
        self.process_button.setText('Build Rapid Splat')
        self.view_mode.addItems(['Gaussian centres · RGB','Original point cloud','Gaussian scale','Opacity','Height'])
        self.image_label=QLabel(); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.hide(); self.center.addWidget(self.image_label,1)
        self.left.addWidget(text('PROJECT MODES','eyebrow'))
        self.left.addWidget(text('Rapid Splat\nGeometry-initialized Gaussians from source XYZ/RGB. No image optimization is implied.'))
        self.left.addWidget(text('Calibrated appearance\nKnown cameras and RGB views refine appearance. CPU reference mode keeps geometry fixed; CUDA mode also optimizes scale and orientation.'))
        self.left.addStretch(); self.left.addWidget(text('LiDAR remains the survey source. Splats are visualization derivatives.','muted'))
        form=QFormLayout(); self.quality=QComboBox(); self.quality.addItems(['Preview · 5k','Balanced · 100k','Detailed · 250k']); self.quality.setCurrentIndex(1)
        self.voxel=QDoubleSpinBox(); self.voxel.setRange(.001,5); self.voxel.setDecimals(3); self.voxel.setValue(.02)
        self.opacity=QDoubleSpinBox(); self.opacity.setRange(.01,.99); self.opacity.setValue(.85); self.opacity.setSingleStep(.05)
        form.addRow('Gaussian budget',self.quality); form.addRow('Voxel size · m',self.voxel); form.addRow('Initial opacity',self.opacity)
        self.right.addLayout(form); self.neutral=QCheckBox('Use neutral color if source has no RGB'); self.right.addWidget(self.neutral)
        self.right.addWidget(text('CALIBRATED IMAGERY','eyebrow'))
        self.calibration=QLineEdit(); self.calibration.setPlaceholderText('Camera manifest JSON')
        self.calibration_button=self.button('Choose camera manifest…',self.choose_calibration); self.right.addWidget(self.calibration); self.right.addWidget(self.calibration_button)
        self.backend=QComboBox(); self.backend.addItems(['CPU reference · small scenes','CUDA / gsplat · optional GPU'])
        self.right.addWidget(self.backend); self.iterations=QSpinBox(); self.iterations.setRange(1,5000); self.iterations.setValue(50)
        self.right.addWidget(text('Optimization iterations','muted')); self.right.addWidget(self.iterations)
        self.checkpoint=QLineEdit(str(self.workspace/'appearance.npz')); self.right.addWidget(text('Checkpoint path','muted')); self.right.addWidget(self.checkpoint)
        self.resume=QCheckBox('Resume matching checkpoint'); self.right.addWidget(self.resume)
        self.fit_button=self.button('Fit calibrated imagery',self.fit,True); self.render_button=self.button('Render calibrated view',self.render_camera)
        self.right.addWidget(self.fit_button); self.right.addWidget(self.render_button)
        self.axes=QComboBox(); self.axes.addItems(['RFU · X right / Y forward / Z up','RDF · X right / Y down / Z forward','RUB · X right / Y up / Z back'])
        self.right.addWidget(text('Source axes for SPZ export','muted')); self.right.addWidget(self.axes)
        self.tile_button=self.button('Export tiled SPZ directory…',self.export_tiled); self.right.addWidget(self.tile_button)
        self.right.addWidget(text('3D preview shows Gaussian centres. Calibrated view rendering composites full anisotropic Gaussians. Keep SPZ origin/axis sidecars.','muted'))
        self.right.addStretch(); self.refresh_actions()

    def refresh_actions(self,*args):
        super().refresh_actions()
        if hasattr(self,'fit_button'):
            for button in [self.fit_button,self.render_button,self.tile_button]: button.setEnabled(not self.busy and self.result is not None)

    def process(self):
        if self.data is None or not self.units.isChecked() or self.busy: return
        config=RapidConfig(voxel_size=self.voxel.value(),max_gaussians=[5000,100000,250000][self.quality.currentIndex()],opacity=self.opacity.value(),allow_uncolored=self.neutral.isChecked())
        data=self.data; folder=self.fresh_folder('rapid-')
        self.run_job(lambda cancel,progress:rapid_splat(data,folder,config,cancel,progress),self.accept_result)

    def accept_result(self,result): self.result=result; self.update_view(); self.refresh_actions()

    def update_view(self,*args):
        if self.data is None: return
        if hasattr(self,'image_label'): self.image_label.hide(); self.viewer.show()
        if self.result is None or self.view_mode.currentIndex()==1:
            ids=self.preview_ids(); points=self.data.points[ids]
            colors=self.data.colors[ids] if self.data.colors is not None else np.full((len(ids),3),.6)
        else:
            ids=np.linspace(0,len(self.result.means)-1,min(100000,len(self.result.means)),dtype='int64')
            points=self.result.means[ids]; colors=self.result.colors[ids]; mode=self.view_mode.currentIndex()
            if mode==2:
                value=np.exp(self.result.log_scales[ids]).max(axis=1); value=(value-value.min())/max(np.ptp(value),1e-9); colors=np.column_stack([value,.6*np.ones(len(value)),1-value])
            elif mode==3:
                from scipy.special import expit
                value=expit(self.result.opacity_logits[ids]); colors=np.column_stack([value,value,value])
        if self.view_mode.currentIndex()==4:
            value=(points[:,2]-points[:,2].min())/max(np.ptp(points[:,2]),1e-9); colors=np.column_stack([value,.6*np.ones(len(value)),1-value])
        self.viewer.set_data(points,colors)
        if self.result is not None:
            self.metrics.setText(f'{len(self.result.means):,} Gaussians · {self.result.metadata.get("mode","unknown")}\nSource measurements are preserved. Preview: centres, not rasterized appearance.')

    def choose_calibration(self):
        path,_=QFileDialog.getOpenFileName(self,'Open calibrated camera manifest','','Camera manifest (*.json)')
        if path: self.calibration.setText(path)

    def fit(self):
        if self.result is None or self.busy: return
        manifest=self.calibration.text().strip(); checkpoint=self.checkpoint.text().strip()
        if not manifest or not checkpoint:
            QMessageBox.warning(self,'Calibration required','Choose a calibrated camera manifest and checkpoint path.'); return
        from pathlib import Path
        if Path(checkpoint).resolve()==Path(manifest).resolve():
            QMessageBox.warning(self,'Invalid checkpoint','The checkpoint must not overwrite the camera manifest.'); return
        scene=self.result; backend=self.backend.currentIndex(); iterations=self.iterations.value(); resume=self.resume.isChecked()
        def operation(cancel,progress):
            cameras,identity=load_cameras(manifest,max_side=128 if backend==0 else 1024)
            if backend==0: return optimize(scene,cameras,identity,checkpoint,AppearanceConfig(iterations=iterations),cancel,progress,resume)
            return train(scene,cameras,identity,checkpoint,GpuConfig(iterations=iterations),cancel,progress,resume)
        self.run_job(operation,self.accept_result)

    def render_camera(self):
        if self.result is None or self.busy: return
        scene=self.result; manifest=self.calibration.text().strip()
        if not manifest: QMessageBox.warning(self,'Calibration required','Choose a calibrated camera manifest first.'); return
        def operation(cancel,progress):
            if len(scene.means)>5000: raise MemoryError('The CPU reference render is limited to 5,000 Gaussians. Choose Preview quality and rebuild.')
            cameras,_=load_cameras(manifest,max_side=256)
            return render(scene,cameras[0],cancel)[0]
        def accept(image):
            rgb=np.rint(np.clip(image,0,1)*255).astype('uint8')
            qimage=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(650,500,Qt.KeepAspectRatio,Qt.SmoothTransformation)); self.viewer.hide(); self.image_label.show()
            self.metrics.setText('Calibrated anisotropic Gaussian render · first camera. Change preview mode to return to 3D.')
        self.run_job(operation,accept)

    def save_project(self):
        if self.result is None or self.busy: return
        path,_=QFileDialog.getSaveFileName(self,'Save Gaussian project','','AV Splat project (*.avsplat)')
        if path:
            scene=self.result
            self.run_job(lambda cancel,progress:save_project(path,scene,cancel),lambda _:self.log.append('Gaussian project saved: '+path))

    def open_project(self):
        if self.busy: return
        path,_=QFileDialog.getOpenFileName(self,'Open Gaussian project','','AV Splat project (*.avsplat);;Gaussian PLY (*.ply)')
        if path:
            def operation(cancel,progress): return read_ply(path) if path.lower().endswith('.ply') else load_project(path)
            def accept(scene):
                self.accept_data(Dataset(scene.means,scene.colors,metadata={'units':'project source units; see metadata','source':fingerprint(path)}))
                self.accept_result(scene)
            self.run_job(operation,accept)

    def export(self):
        if self.result is None or self.busy: return
        path,_=QFileDialog.getSaveFileName(self,'Export Gaussian visualization derivative','','Gaussian PLY (*.ply);;SPZ v3 (*.spz)')
        if path:
            scene=self.result; axes=self.axes.currentText().split(' ')[0]
            self.run_job(lambda cancel,progress:export_spz(path,scene,cancel,axes) if path.lower().endswith('.spz') else export_ply(path,scene,cancel),lambda value:self.log.append('Validated Gaussian export: '+value))

    def export_tiled(self):
        if self.result is None or self.busy: return
        path,_=QFileDialog.getSaveFileName(self,'Choose a NEW tile directory name','','Tile directory (*)')
        if path:
            scene=self.result; axes=self.axes.currentText().split(' ')[0]
            self.run_job(lambda cancel,progress:export_tiles(path,scene,cancel,axes),lambda value:self.log.append('SPZ tile manifest: '+value))
