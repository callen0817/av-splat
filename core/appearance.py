"""Small-scene CPU reference optimization against actual calibrated RGB views.

Geometry is fixed. This is an explicit geometry-constrained appearance mode,
not a scalable replacement for CUDA 3DGS training.
"""
from dataclasses import dataclass,asdict
import copy
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import expit
from av_common.data import check_cancel,atomic_json
from core.rapid import GaussianScene

@dataclass
class AppearanceConfig:
    iterations: int=50
    learning_rate: float=.5
    max_gaussians: int=5000
    max_pixels: int=128*128
    checkpoint_every: int=5

    def validate(self):
        if not 1<=self.iterations<=5000 or not np.isfinite(self.learning_rate) or not 0<self.learning_rate<=2:
            raise ValueError('Invalid iteration budget or learning rate.')
        if not 1<=self.max_gaussians<=10_000 or not 1<=self.max_pixels<=512*512 or self.checkpoint_every<1:
            raise ValueError('Reference-renderer resource budget is invalid.')
        return self


def footprints(scene,camera,cancel=lambda:False):
    height,width=camera.image.shape[:2]
    rotation=camera.world_to_camera[:3,:3]; translation=camera.world_to_camera[:3,3]
    camera_points=scene.means@rotation.T+translation
    axes=Rotation.from_quat(scene.quaternions[:,[1,2,3,0]]).as_matrix()
    scales=np.exp(scene.log_scales)
    opacity=expit(scene.opacity_logits)
    transmission=np.ones((height,width),dtype='float64'); records=[]; pixels=0
    # Front-to-back alpha compositing in known camera depth order.
    for i in np.argsort(camera_points[:,2]):
        check_cancel(cancel); x,y,z=camera_points[i]
        if z<=1e-4: continue
        center=(camera.K@camera_points[i])[:2]/z
        jacobian=np.array([[camera.K[0,0]/z,camera.K[0,1]/z,-(camera.K[0,0]*x+camera.K[0,1]*y)/z**2],[0,camera.K[1,1]/z,-camera.K[1,1]*y/z**2]])
        covariance=axes[i]@np.diag(scales[i]**2)@axes[i].T
        covariance2=jacobian@rotation@covariance@rotation.T@jacobian.T+np.eye(2)*.3
        radius=min(3*np.sqrt(np.linalg.eigvalsh(covariance2).max()),max(height,width)*2)
        xmin,xmax=max(0,int(np.floor(center[0]-radius))),min(width,int(np.ceil(center[0]+radius))+1)
        ymin,ymax=max(0,int(np.floor(center[1]-radius))),min(height,int(np.ceil(center[1]+radius))+1)
        if xmin>=xmax or ymin>=ymax: continue
        pixels+=(ymax-ymin)*(xmax-xmin)
        if pixels>2_000_000: raise MemoryError('Reference renderer exceeded its two-million footprint-pixel budget. Reduce the scene/image size.')
        yy,xx=np.mgrid[ymin:ymax,xmin:xmax]; delta=np.stack([xx+.5-center[0],yy+.5-center[1]],axis=-1)
        distance=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(covariance2),delta)
        gaussian=np.exp(-.5*distance); gaussian[distance>9]=0
        alpha=np.clip(opacity[i]*gaussian,0,.995)
        region=(slice(ymin,ymax),slice(xmin,xmax))
        before=transmission[region].copy(); weight=before*alpha
        records.append((int(i),region,weight,alpha,gaussian,before))
        transmission[region]*=1-alpha
    return records


def render(scene,camera,cancel=lambda:False):
    records=footprints(scene,camera,cancel)
    image=np.zeros_like(camera.image,dtype='float64')
    for i,region,weight,_,_,_ in records: image[region]+=weight[:,:,None]*scene.colors[i]
    return image,records


def _checkpoint(path,scene,iteration,identity,history,config,initial_holdout):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.appearance-',suffix='.npz',dir=path.parent); os.close(fd)
    try:
        with open(temp,'wb') as stream:
            np.savez_compressed(stream,colors=scene.colors,opacity_logits=scene.opacity_logits,iteration=np.array(iteration),identity=np.array(identity),history=np.asarray(history),config=np.array(json.dumps(asdict(config))),initial_holdout=np.array(initial_holdout))
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def optimize(scene,cameras,identity,checkpoint,config=None,cancel=lambda:False,progress=lambda message:None,resume=False):
    import hashlib
    from formats.gaussian import protect
    protect(scene,checkpoint)
    if any(Path(checkpoint).resolve()==Path(c.image_path).resolve() for c in cameras): raise ValueError('Checkpoint must not overwrite an input image.')
    scene.validate(); config=(config or AppearanceConfig()).validate()
    if len(cameras)<2: raise ValueError('At least two calibrated views are required.')
    if len(scene.means)>config.max_gaussians or any(c.image.shape[0]*c.image.shape[1]>config.max_pixels for c in cameras):
        raise MemoryError('Scene exceeds CPU reference-mode limits. Reduce the Gaussian/image budget or use a validated CUDA backend.')
    for camera in cameras: camera.validate()
    geometry=hashlib.sha256(scene.means.tobytes()+scene.log_scales.tobytes()+scene.quaternions.tobytes()).hexdigest()
    identity=hashlib.sha256((identity+geometry).encode()).hexdigest()
    result=copy.deepcopy(scene); history=[]; start=0; baseline=None
    if resume:
        with np.load(checkpoint,allow_pickle=False) as state:
            if str(state['identity'])!=identity or state['colors'].shape!=scene.colors.shape: raise ValueError('Checkpoint calibration or geometry differs.')
            if str(state['config'])!=json.dumps(asdict(config)): raise ValueError('Checkpoint optimization settings differ.')
            result.colors=np.array(state['colors']); result.opacity_logits=np.array(state['opacity_logits']); start=int(state['iteration']); history=state['history'].tolist(); baseline=float(state['initial_holdout']) if 'initial_holdout' in state else None
    # Reserve the last camera as an untouched held-out view for reporting.
    training=cameras[:-1]
    initial_holdout=baseline if baseline is not None else float(np.mean((render(result,cameras[-1],cancel)[0]-cameras[-1].image)**2))
    for iteration in range(start,config.iterations):
        check_cancel(cancel)
        gradients=np.zeros_like(result.colors,dtype='float64'); opacity_gradient=np.zeros(len(result.means)); loss=0
        contributions=np.zeros(len(result.means))
        for camera in training:
            prediction,records=render(result,camera,cancel)
            if not records: raise ValueError('No Gaussians are visible in a training camera; verify calibration, units and poses.')
            residual=prediction-camera.image
            loss+=float(np.mean(residual**2))
            # Per-splat RGB gradients use exact front-to-back contribution weights.
            # Normalize by support to avoid scene-size-dependent tiny updates.
            behind=np.zeros_like(prediction)
            for i,region,weight,alpha,gaussian,before in reversed(records):
                local=residual[region]
                gradients[i]+=2*np.sum(local*weight[:,:,None],axis=(0,1))
                contributions[i]+=float(weight.sum())
                derivative=before[:,:,None]*(result.colors[i]-behind[region])*gaussian[:,:,None]
                derivative*=((expit(result.opacity_logits[i])*gaussian)<.995)[:,:,None]
                opacity_gradient[i]+=2*float(np.sum(local*derivative))*float(expit(result.opacity_logits[i])*(1-expit(result.opacity_logits[i])))
                behind[region]=alpha[:,:,None]*result.colors[i]+(1-alpha[:,:,None])*behind[region]
        normalizer=np.maximum(contributions,1)[:,None]
        result.colors=np.clip(result.colors-config.learning_rate*gradients/normalizer,0,1).astype('float32')
        result.opacity_logits=np.clip(result.opacity_logits-config.learning_rate*opacity_gradient/np.maximum(contributions,1),-8,8).astype('float32')
        history.append(loss/len(training))
        if (iteration+1)%config.checkpoint_every==0 or iteration+1==config.iterations:
            _checkpoint(checkpoint,result,iteration+1,identity,history,config,initial_holdout)
            progress(f'Appearance iteration {iteration+1}/{config.iterations} · train MSE {history[-1]:.6f}')
    from core.cameras import verify_camera_sources
    verify_camera_sources(cameras)
    final_holdout=float(np.mean((render(result,cameras[-1],cancel)[0]-cameras[-1].image)**2))
    result.metadata={**result.metadata,'mode':'calibrated CPU appearance reference','optimized_with_images':True,'geometry_fixed':True,'appearance':{'cameras':[{'K':c.K.tolist(),'world_to_camera':c.world_to_camera.tolist(),'image':c.image_path,'shape':list(c.image.shape),'source':c.source} for c in cameras],'config':asdict(config),'iterations_completed':config.iterations,'training_mse':history,'initial_heldout_mse':initial_holdout,'heldout_mse':final_holdout,'calibration_geometry_identity':identity,'checkpoint':str(Path(checkpoint).resolve())},
                     'warnings':['Reference CPU image fitting optimizes RGB and opacity, with positions, scales and orientations fixed. It is not full scalable photorealistic CUDA reconstruction.','Synthetic held-out image tests are not production photometric validation.']}
    return result.validate()
