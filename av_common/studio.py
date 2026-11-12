"""Shared AV Suite Qt shell with responsive jobs and bounded OpenGL previews."""
import os
from pathlib import Path
import tempfile
import threading
import numpy as np
from PySide6.QtCore import Qt,QThread,Signal,QTimer
from PySide6.QtWidgets import (QMainWindow,QWidget,QFrame,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QSplitter,
    QComboBox,QCheckBox,QProgressBar,QTextEdit,QMessageBox,QFileDialog,QScrollArea)
from av_common.theme import STYLE,icon


class Job(QThread):
    progress=Signal(str); completed=Signal(object); error=Signal(str)
    def __init__(self,operation,parent=None):
        super().__init__(parent); self.operation=operation; self.stop=threading.Event()
    def run(self):
        try: result=self.operation(self.stop.is_set,self.progress.emit); self.completed.emit(result)
        except Exception as error: self.error.emit(str(error))
    def cancel(self): self.stop.set()


def text(value,role=None):
    label=QLabel(value); label.setWordWrap(True)
    if role: label.setObjectName(role)
    return label


def card(title):
    frame=QFrame(); frame.setObjectName('glass'); box=QVBoxLayout(frame); box.setContentsMargins(16,16,16,16); box.setSpacing(10)
    box.addWidget(text(title,'eyebrow')); return frame,box


class Viewer(QWidget):
    def __init__(self):
        super().__init__(); self.gl=None; self.item=None; self.points=None; self.colors=None
        box=QVBoxLayout(self); box.setContentsMargins(0,0,0,0)
        tools=QHBoxLayout(); self.camera=QComboBox(); self.camera.addItems(['Perspective','Top','Front','Side'])
        self.section=QComboBox(); self.section.addItems(['No section','Lower half X','Lower half Y','Lower half Z'])
        self.visible=QCheckBox('Show scene'); self.visible.setChecked(True)
        for widget in [self.camera,self.section,self.visible]: tools.addWidget(widget)
        box.addLayout(tools)
        self.hint=text('Import a point cloud to begin.\nOriginal survey geometry remains authoritative.','muted'); self.hint.setAlignment(Qt.AlignCenter); box.addWidget(self.hint,1)
        self.box=box; self.camera.currentIndexChanged.connect(self.render); self.section.currentIndexChanged.connect(self.render); self.visible.toggled.connect(self.render)

    def set_data(self,points,colors):
        self.points=np.asarray(points); self.colors=np.asarray(colors)
        if self.points.shape!=(len(self.colors),3): raise ValueError('Preview color/point counts differ.')
        if self.gl is None and os.environ.get('AV_HEADLESS')!='1':
            import pyqtgraph.opengl as gl
            self.module=gl; self.gl=gl.GLViewWidget(); self.gl.setBackgroundColor('#171a21'); self.box.addWidget(self.gl,1); self.hint.hide()
        self.render()

    def render(self,*args):
        if self.points is None: return
        if self.gl is None:
            self.hint.setText(f'{len(self.points):,} preview points loaded. Headless mode: a desktop/OpenGL display is required for 3D rendering.'); return
        if self.item is not None: self.gl.removeItem(self.item); self.item=None
        if not self.visible.isChecked() or not len(self.points): return
        center=(self.points.min(0)+self.points.max(0))/2
        mask=np.ones(len(self.points),bool)
        if self.section.currentIndex():
            axis=self.section.currentIndex()-1; mask=self.points[:,axis]<=center[axis]
        colors=self.colors[mask]
        if colors.shape[1]==3: colors=np.column_stack([colors,np.ones(len(colors))])
        self.item=self.module.GLScatterPlotItem(pos=self.points[mask]-center,color=colors,size=2,pxMode=True)
        self.item.setGLOptions('translucent'); self.gl.addItem(self.item)
        elevation,azimuth=[(30,45),(90,0),(0,0),(0,90)][self.camera.currentIndex()]
        self.gl.setCameraPosition(distance=max(float(np.ptp(self.points,axis=0).max())*1.6,1),elevation=elevation,azimuth=azimuth)


class Studio(QMainWindow):
    def __init__(self,product,subtitle):
        super().__init__(); self.product=product; self.data=None; self.result=None; self.job=None; self.busy=False; self.closing=False
        self.setWindowTitle(product+' — Artificial Vision '+subtitle); self.setWindowIcon(icon()); self.setStyleSheet(STYLE)
        self.resize(1440,920); self.setMinimumSize(1040,680)
        self.workspace=Path(os.environ.get('AV_WORKSPACE',Path(tempfile.gettempdir())/('av-'+product.lower().replace(' ','-'))))
        self.workspace.mkdir(parents=True,exist_ok=True)
        root=QWidget(); self.setCentralWidget(root); layout=QVBoxLayout(root); layout.setContentsMargins(20,16,20,16); layout.setSpacing(12)
        header,box=card('ARTIFICIAL VISION   /   SPATIAL TOOLS')
        title=QHBoxLayout(); title.addWidget(text(product,'title')); title.addWidget(text(subtitle,'muted')); title.addStretch(); title.addWidget(text('An Artificial Vision Product.','eyebrow')); box.addLayout(title)
        self.toolbar=QHBoxLayout(); self.actions=[]
        self.import_button=self.button('Import…',self.import_data); self.process_button=self.button('Process',self.process,True)
        self.open_button=self.button('Open project…',self.open_project); self.save_button=self.button('Save project…',self.save_project); self.export_button=self.button('Export…',self.export)
        for button in [self.import_button,self.process_button,self.open_button,self.save_button,self.export_button]: self.toolbar.addWidget(button)
        self.toolbar.addStretch(); box.addLayout(self.toolbar); layout.addWidget(header)
        splitter=QSplitter(Qt.Horizontal)
        left,self.left=card('01   /   PROJECT & DATASETS'); self.source_label=text('No dataset selected','muted'); self.left.addWidget(self.source_label)
        self.units=QCheckBox('Source coordinates use metres'); self.units.toggled.connect(self.refresh_actions); self.left.addWidget(self.units)
        splitter.addWidget(left)
        center,self.center=card('02   /   SPATIAL REVIEW'); self.view_mode=QComboBox(); self.center.addWidget(self.view_mode); self.view_mode.currentIndexChanged.connect(self.update_view)
        self.viewer=Viewer(); self.center.addWidget(self.viewer,1); self.metrics=text('Waiting for data','muted'); self.center.addWidget(self.metrics); splitter.addWidget(center)
        right,self.right=card('03   /   PROCESSING & REVIEW'); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(right); scroll.setMinimumWidth(290); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); splitter.addWidget(scroll)
        splitter.setSizes([260,850,320]); layout.addWidget(splitter,1)
        footer,logs=card('JOBS & DIAGNOSTICS'); row=QHBoxLayout(); self.progress=QProgressBar(); self.progress.setRange(0,1); self.progress.setValue(0)
        self.cancel_button=QPushButton('Cancel'); self.cancel_button.clicked.connect(self.cancel); self.cancel_button.setEnabled(False)
        self.log_toggle=QCheckBox('Show diagnostics'); row.addWidget(self.progress,1); row.addWidget(self.cancel_button); row.addWidget(self.log_toggle); logs.addLayout(row)
        self.log=QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120); self.log.hide(); self.log_toggle.toggled.connect(self.log.setVisible); logs.addWidget(self.log); layout.addWidget(footer)
        self.telemetry=QTimer(self); self.telemetry.timeout.connect(self.update_telemetry); self.telemetry.start(2000)

    def button(self,title,slot,primary=False):
        button=QPushButton(title); button.clicked.connect(slot)
        if primary: button.setObjectName('primary')
        self.actions.append(button); return button

    def fresh_folder(self,prefix='job-'):
        return Path(tempfile.mkdtemp(prefix=prefix,dir=self.workspace))

    def run_job(self,operation,completed):
        if self.busy: return
        if self.job is not None and not self.job.isRunning(): self.job.deleteLater()
        self.busy=True; self.progress.setRange(0,0); self.cancel_button.setEnabled(True); self.refresh_actions()
        self.job=Job(operation,self); self.job.progress.connect(self.log.append); self.job.completed.connect(completed); self.job.error.connect(self.failed); self.job.finished.connect(self.stopped); self.job.start()

    def failed(self,message):
        self.log.append('ERROR: '+message)
        if 'cancel' not in message.lower() and not self.closing: QMessageBox.critical(self,'Processing error',message)

    def stopped(self):
        if self.job is not None: self.job.operation=None
        self.busy=False; self.progress.setRange(0,1); self.progress.setValue(1); self.cancel_button.setEnabled(False); self.refresh_actions()
        if self.closing: self.close()

    def cancel(self):
        if self.job and self.job.isRunning(): self.job.cancel(); self.cancel_button.setEnabled(False); self.log.append('Cancellation requested; finishing the current bounded operation.')

    def refresh_actions(self,*args):
        for button in self.actions: button.setEnabled(not self.busy)
        self.process_button.setEnabled(not self.busy and self.data is not None and self.units.isChecked())
        self.save_button.setEnabled(not self.busy and self.result is not None)
        self.export_button.setEnabled(not self.busy and self.result is not None)
        self.units.setEnabled(not self.busy)

    def import_data(self):
        path,_=QFileDialog.getOpenFileName(self,'Import point cloud','','Point clouds (*.ply *.las *.laz *.e57 *.xyz *.xyzrgb)')
        if path:
            from av_common.data import load_dataset
            self.run_job(lambda cancel,progress:load_dataset(path,self.workspace/'imports',cancel,progress),self.accept_data)

    def accept_data(self,data):
        self.data=data; self.result=None; self.units.setChecked(False)
        self.source_label.setText(f"{Path(data.metadata.get('source',{}).get('path','Dataset')).name}\n{len(data.points):,} original points\n{data.metadata.get('units','Confirm units')}\nCRS: {'present' if data.metadata.get('crs') else 'not supplied'}")
        self.metrics.setText(f'{len(data.points):,} source points imported. Confirm units before geometric processing.')
        self.update_view(); self.refresh_actions()

    def preview_ids(self):
        return np.linspace(0,len(self.data.points)-1,min(100_000,len(self.data.points)),dtype='int64')

    def update_view(self,*args): pass
    def process(self): pass
    def open_project(self): pass
    def save_project(self): pass
    def export(self): pass

    def update_telemetry(self):
        import psutil
        process=psutil.Process()
        self.statusBar().showMessage(f'Local CPU processing  ·  RSS {process.memory_info().rss/1024**2:.0f} MiB  ·  Preview ≤100k points')

    def closeEvent(self,event):
        if self.job and self.job.isRunning(): self.closing=True; self.cancel(); event.ignore()
        else: self.telemetry.stop(); event.accept()
