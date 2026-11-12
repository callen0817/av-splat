"""Geometry-initialized Gaussians; no photorealistic optimization is implied."""
from dataclasses import dataclass,asdict,field
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from av_common.data import check_cancel,atomic_json
from av_common.geometry import voxelize,neighbourhood_features


@dataclass
class RapidConfig:
    voxel_size: float=.02
    max_gaussians: int=250_000
    neighbours: int=20
    opacity: float=.85
    min_scale: float=.001
    max_scale: float=1.
    allow_uncolored: bool=False

    def validate(self):
        if not all(np.isfinite(x) for x in [self.voxel_size,self.opacity,self.min_scale,self.max_scale]) or self.voxel_size<=0 or not 0<self.opacity<1:
            raise ValueError('Voxel size and opacity must be finite and valid.')
        if not 100<=self.max_gaussians<=1_000_000 or not 4<=self.neighbours<=64 or not 0<self.min_scale<=self.max_scale:
            raise ValueError('Gaussian count, neighbourhood or scale bounds are invalid.')
        return self


@dataclass
class GaussianScene:
    means: np.ndarray
    colors: np.ndarray
    log_scales: np.ndarray
    quaternions: np.ndarray # w,x,y,z
    opacity_logits: np.ndarray
    metadata: dict=field(default_factory=dict)

    def validate(self):
        n=len(self.means)
        for array,shape in [(self.means,(n,3)),(self.colors,(n,3)),(self.log_scales,(n,3)),(self.quaternions,(n,4)),(self.opacity_logits,(n,))]:
            if np.asarray(array).shape!=shape or not np.isfinite(array).all(): raise ValueError('Invalid finite Gaussian scene arrays.')
        if not n or np.any((self.colors<0)|(self.colors>1)) or not np.allclose(np.linalg.norm(self.quaternions,axis=1),1,atol=1e-5):
            raise ValueError('Invalid colors or non-unit Gaussian orientations.')
        if np.any((self.log_scales<-20)|(self.log_scales>10)) or np.any(np.abs(self.opacity_logits)>30):
            raise ValueError('Gaussian scale or opacity is outside supported numerical bounds.')
        return self


def rapid_splat(data,folder,config=None,cancel=lambda:False,progress=lambda message:None):
    config=(config or RapidConfig()).validate(); data.validate(); started=time.perf_counter(); folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if (data.colors is None or data.metadata.get('colors_complete') is False) and not config.allow_uncolored: raise ValueError('Rapid Splat needs source RGB. Enable neutral-color mode explicitly for uncolored geometry.')
    progress('Indexing geometry within the Gaussian budget')
    voxels=voxelize(data.points,folder,config.voxel_size,config.max_gaussians,cancel,progress)
    if len(voxels.points)<4: raise ValueError('At least four distinct points are required for covariance estimation.')
    progress('Estimating local anisotropic covariance and orientation')
    eigenvalues,eigenvectors=neighbourhood_features(voxels.points,config.neighbours,cancel)
    scales=np.sqrt(np.maximum(eigenvalues,config.min_scale**2))
    # Neighbour covariance describes the support of several samples. Reduce
    # tangent scale to avoid turning sparse observations into enormous blobs.
    scales=np.clip(scales/np.sqrt(config.neighbours),config.min_scale,config.max_scale)
    q=Rotation.from_matrix(eigenvectors).as_quat()[:,[3,0,1,2]]
    q[q[:,0]<0]*=-1
    colors=np.asarray(data.colors[voxels.representative_ids],dtype='float32') if data.colors is not None else np.full((len(scales),3),.5,dtype='float32')
    logits=np.full(len(scales),np.log(config.opacity/(1-config.opacity)),dtype='float32')
    metadata={'schema_version':1,'mode':'rapid','optimized_with_images':False,'source':data.metadata,'config':asdict(config),'effective_voxel_size_m':voxels.size,'source_points':len(data.points),'gaussians':len(scales),'seconds':time.perf_counter()-started,
              'warnings':['Direct geometric initialization, not optimized photorealistic reconstruction. Original LiDAR remains authoritative.','View preview displays Gaussian centres; exported files contain full anisotropic Gaussian parameters.']}
    scene=GaussianScene(voxels.points,colors,np.log(scales).astype('float32'),q.astype('float32'),logits,metadata).validate()
    check_cancel(cancel); progress('Rapid Splat complete')
    return scene
