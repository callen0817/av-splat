"""Sampled process RSS; not an exact OS/native peak or a VRAM measurement."""
import threading
import time
import psutil

class ResourceMeter:
    def __init__(self):
        self.process=psutil.Process(); self.stop=threading.Event(); self.interval=.02
    def __enter__(self):
        self.started=time.perf_counter(); self.initial=self.process.memory_info().rss; self.peak=self.initial
        def sample():
            while not self.stop.wait(self.interval): self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss); self.stop.set(); self.thread.join(); self.seconds=time.perf_counter()-self.started
    def report(self):
        return {'sampled_peak_process_rss_bytes':self.peak,'initial_process_rss_bytes':self.initial,'rss_sample_interval_seconds':self.interval,'monitored_seconds':self.seconds,'gpu_memory':'unmeasured'}
