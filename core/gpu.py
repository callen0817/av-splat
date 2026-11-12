"""Optional CUDA/gsplat training. CPU-only installations never import this stack."""
from dataclasses import dataclass,asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from av_common.data import check_cancel
from core.rapid import GaussianScene


def capabilities():
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('gsplat') is None:
        return {'available':False,'reason':'Optional torch/gsplat packages are absent. Install requirements-gpu.txt only on a supported CUDA system.'}
    import torch
    if not torch.cuda.is_available(): return {'available':False,'reason':'CUDA is unavailable. Rapid Splat and the small-scene CPU appearance mode remain available.'}
    free,total=torch.cuda.mem_get_info()
    return {'available':True,'device':torch.cuda.get_device_name(),'free_vram_bytes':free,'total_vram_bytes':total}

@dataclass
class GpuConfig:
    iterations: int=300
    learning_rate: float=.005
    checkpoint_every: int=25
    max_image_side: int=1024


def train(scene,cameras,identity,checkpoint,config=None,cancel=lambda:False,progress=lambda message:None,resume=False):
    state=capabilities()
    if not state['available']: raise RuntimeError(state['reason'])
    import torch
    from gsplat import rasterization
    from scipy.special import expit,logit
    from formats.gaussian import protect
    protect(scene,checkpoint)
    if any(Path(checkpoint).resolve()==Path(c.image_path).resolve() for c in cameras): raise ValueError('Checkpoint must not overwrite an input image.')
    scene.validate(); config=config or GpuConfig()
    if len(cameras)<2 or not 1<=config.iterations<=100_000 or not 0<config.learning_rate<=.1 or config.checkpoint_every<1: raise ValueError('Invalid GPU training budget or calibration.')
    for camera in cameras:
        camera.validate()
        if max(camera.image.shape[:2])>config.max_image_side: raise MemoryError('Resize calibrated images within the configured GPU budget.')
    estimated=len(scene.means)*256+max(c.image.shape[0]*c.image.shape[1] for c in cameras)*96
    if estimated>state['free_vram_bytes']*.5: raise MemoryError('Insufficient free VRAM for a safe batch. Reduce Gaussian/image resolution.')
    identity=hashlib.sha256((identity+json.dumps(scene.metadata.get('config',{}),sort_keys=True)).encode()+scene.means.tobytes()).hexdigest()
    torch.cuda.reset_peak_memory_stats()
    origin=scene.means.mean(0)
    means=torch.tensor(scene.means-origin,device='cuda',dtype=torch.float32)
    params=[torch.nn.Parameter(torch.tensor(logit(np.clip(scene.colors,1e-4,1-1e-4)),device='cuda',dtype=torch.float32)),torch.nn.Parameter(torch.tensor(scene.opacity_logits,device='cuda',dtype=torch.float32)),torch.nn.Parameter(torch.tensor(scene.log_scales,device='cuda',dtype=torch.float32)),torch.nn.Parameter(torch.tensor(scene.quaternions,device='cuda',dtype=torch.float32))]
    optimizer=torch.optim.Adam(params,lr=config.learning_rate); start=0; history=[]
    if resume:
        saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
        if saved['identity']!=identity or saved['config']!=asdict(config): raise ValueError('GPU checkpoint data, calibration or configuration differs.')
        for parameter,value in zip(params,saved['parameters']): parameter.data.copy_(value)
        optimizer.load_state_dict(saved['optimizer']); start=saved['iteration']; history=saved['history']
    def predict(camera):
        transform=camera.world_to_camera.copy(); transform[:3,3]+=transform[:3,:3]@origin
        result,_,_=rasterization(means=means,quats=torch.nn.functional.normalize(params[3],dim=1),scales=torch.exp(params[2].clamp(-10,3)),opacities=torch.sigmoid(params[1]),colors=torch.sigmoid(params[0]),
            viewmats=torch.tensor(transform[None],device='cuda',dtype=torch.float32),Ks=torch.tensor(camera.K[None],device='cuda',dtype=torch.float32),width=camera.image.shape[1],height=camera.image.shape[0],sh_degree=None,packed=True,render_mode='RGB')
        return result[0]
    training=cameras[:-1]
    for iteration in range(start,config.iterations):
        check_cancel(cancel); camera=training[iteration%len(training)]
        optimizer.zero_grad(); prediction=predict(camera)
        target=torch.tensor(camera.image,device='cuda',dtype=torch.float32)
        loss=torch.mean(torch.abs(prediction-target))
        if not torch.isfinite(loss): raise FloatingPointError('CUDA renderer produced a nonfinite loss; last valid checkpoint is preserved.')
        loss.backward(); optimizer.step(); history.append(float(loss.detach().cpu()))
        with torch.no_grad():
            params[1].clamp_(-8,8); params[2].clamp_(-10,3)
            if not all(torch.isfinite(p).all() for p in params): raise FloatingPointError('Nonfinite CUDA parameters; last valid checkpoint is preserved.')
        if (iteration+1)%config.checkpoint_every==0 or iteration+1==config.iterations:
            path=Path(checkpoint); path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix='.cuda-',suffix='.pt',dir=path.parent); os.close(fd)
            try:
                torch.save({'identity':identity,'config':asdict(config),'parameters':[p.detach().cpu() for p in params],'optimizer':optimizer.state_dict(),'iteration':iteration+1,'history':history},temp); os.replace(temp,path)
            finally:
                if os.path.exists(temp): os.unlink(temp)
            progress(f'CUDA iteration {iteration+1}/{config.iterations} · training L1 {history[-1]:.6f}')
    with torch.no_grad():
        heldout=float(torch.mean((predict(cameras[-1])-torch.tensor(cameras[-1].image,device='cuda',dtype=torch.float32))**2).cpu())
        colors=torch.sigmoid(params[0]).cpu().numpy(); logits=params[1].cpu().numpy(); scales=params[2].cpu().numpy(); quats=torch.nn.functional.normalize(params[3],dim=1).cpu().numpy()
    from core.cameras import verify_camera_sources
    verify_camera_sources(cameras)
    metadata={**scene.metadata,'mode':'CUDA calibrated gsplat','optimized_with_images':True,'geometry_positions_fixed':True,'gpu_training':{'config':asdict(config),'history':history,'heldout_mse':heldout,'device':state['device'],'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint':str(Path(checkpoint).resolve())}}
    return GaussianScene(scene.means.copy(),colors,scales,quats,logits,metadata).validate()
