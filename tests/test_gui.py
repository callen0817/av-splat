import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); os.environ.setdefault('AV_HEADLESS','1')
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtWidgets import QApplication
from av_common.data import Dataset
from av_common.studio import Job
from av_common.theme import BrandSplash
from core.rapid import rapid_splat,RapidConfig
from core.gpu import capabilities,train
from gui.app import SplatStudio

app=QApplication.instance() or QApplication([])

class GuiTests(unittest.TestCase):
    def data(self):
        points=np.random.default_rng(6).uniform(size=(1000,3)); return Dataset(points,np.full_like(points,.5))

    def test_gui_modes_startup_splash_and_missing_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            w=SplatStudio(); w.show(); app.processEvents()
            self.assertFalse(w.process_button.isEnabled()); self.assertFalse(w.export_button.isEnabled())
            data=self.data(); w.accept_data(data); w.units.setChecked(True)
            result=rapid_splat(data,d,RapidConfig(max_gaussians=5000)); w.accept_result(result)
            for index in range(w.view_mode.count()): w.view_mode.setCurrentIndex(index); app.processEvents()
            self.assertTrue(w.fit_button.isEnabled()); self.assertTrue(w.export_button.isEnabled())
            splash=BrandSplash(); splash.status('Ready'); self.assertFalse(splash.pixmap().isNull()); splash.close(); w.close()
            state=capabilities()
            if not state['available']:
                with self.assertRaises(RuntimeError): train(result,[],'identity',Path(d)/'checkpoint.pt')

    def test_cancellable_background_job(self):
        with tempfile.TemporaryDirectory() as d:
            job=Job(lambda cancel,progress:rapid_splat(self.data(),d,cancel=cancel,progress=progress))
            loop=QEventLoop(); timer=QTimer(); timer.setInterval(1); ticks=[]; results=[]
            timer.timeout.connect(lambda:ticks.append(1)); job.completed.connect(results.append); job.finished.connect(loop.quit)
            timeout=QTimer(); timeout.setSingleShot(True); timeout.timeout.connect(loop.quit); timeout.start(10000)
            timer.start(); job.start(); loop.exec(); timer.stop(); timeout.stop()
            if job.isRunning(): job.cancel(); job.wait(10000)
            self.assertTrue(results); self.assertTrue(ticks)

    def test_bidirectional_controls_and_inverse_worker(self):
        from gui.conversion import ConversionStudio
        with tempfile.TemporaryDirectory() as d:
            w=ConversionStudio(); w.accept_data(self.data()); w.units.setChecked(True)
            scene=rapid_splat(w.data,d,RapidConfig(max_gaussians=5000)); w.accept_result(scene)
            w.direction.setCurrentIndex(1); app.processEvents()
            self.assertEqual(w.process_button.text(),'Extract derived points'); self.assertTrue(w.process_button.isEnabled())
            w.process(); loop=QEventLoop(); w.job.finished.connect(loop.quit)
            timeout=QTimer(); timeout.setSingleShot(True); timeout.timeout.connect(loop.quit); timeout.start(10000)
            if w.job.isRunning(): loop.exec()
            else: app.processEvents()
            timeout.stop()
            if w.job.isRunning(): w.job.cancel(); w.job.wait(10000)
            self.assertIsNotNone(w.point_result); self.assertTrue(w.export_button.isEnabled()); self.assertFalse(w.save_button.isEnabled())
            self.assertEqual(len(w.point_result.points),len(scene.means))
            w.point_method.setCurrentIndex(1); self.assertIsNone(w.point_result); self.assertFalse(w.export_button.isEnabled())
            w.close()
