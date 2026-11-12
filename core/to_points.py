"""Splat-to-point extraction and explicitly synthetic covariance sampling."""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation
from av_common.data import Dataset,allocate,check_cancel,verify_source

@dataclass
class PointConfig:
    method: str='centres'
    samples_per_gaussian: int=4
    opacity_threshold: float=.05
    max_points: int=5_000_000
    seed: int=42

    def validate(self):
        if self.method not in {'centres','sample'} or not 1<=self.samples_per_gaussian<=100:
            raise ValueError('Choose centres or covariance sampling, with 1–100 samples per Gaussian.')
        if not np.isfinite(self.opacity_threshold) or not 0<=self.opacity_threshold<=1 or not 1<=self.max_points<=20_000_000:
            raise ValueError('Opacity threshold or point budget is invalid.')
        return self


def splat_to_points(scene,folder,config=None,cancel=lambda:False,progress=lambda message:None):
    scene.validate(); config=(config or PointConfig()).validate()
    opacity=expit(scene.opacity_logits); selected=np.flatnonzero(opacity>=config.opacity_threshold)
    count=len(selected)*(1 if config.method=='centres' else config.samples_per_gaussian)
    if count==0: raise ValueError('No Gaussians pass the opacity threshold.')
    if count>config.max_points: raise MemoryError('Requested conversion exceeds the point budget. Lower sampling density or split the scene.')
    points=allocate(folder,'converted_points',(count,3)); colors=allocate(folder,'converted_colors',(count,3),'float32')
    ids=allocate(folder,'gaussian_id',(count,),'uint32'); alphas=allocate(folder,'gaussian_opacity',(count,),'float32')
    rng=np.random.default_rng(config.seed); offset=0
    for start in range(0,len(selected),4096):
        check_cancel(cancel); block=selected[start:start+4096]
        if config.method=='centres':
            positions=scene.means[block]; repeated=block
        else:
            n=config.samples_per_gaussian
            standardized=rng.normal(size=(len(block),n,3))
            invalid=np.linalg.norm(standardized,axis=2)>3
            while invalid.any():
                check_cancel(cancel); standardized[invalid]=rng.normal(size=(int(invalid.sum()),3)); invalid=np.linalg.norm(standardized,axis=2)>3
            rotation=Rotation.from_quat(scene.quaternions[block][:,[1,2,3,0]]).as_matrix()
            delta=np.einsum('bij,bnj->bni',rotation,standardized*np.exp(scene.log_scales[block])[:,None,:])
            positions=(scene.means[block,None,:]+delta).reshape(-1,3); repeated=np.repeat(block,n)
        end=offset+len(positions); points[offset:end]=positions; colors[offset:end]=scene.colors[repeated]
        ids[offset:end]=repeated.astype('uint32'); alphas[offset:end]=opacity[repeated]
        offset=end; progress(f'Extracted {offset:,} / {count:,} derived points')
    for array in [points,colors,ids,alphas]: array.flush()
    source=scene.metadata.get('source',{}).get('source')
    metadata={'schema_version':1,'derivation':'Gaussian centres' if config.method=='centres' else 'synthetic samples inside anisotropic Gaussian covariance, truncated at 3 sigma',
              'not_original_survey':True,'config':asdict(config),'source':source,'gaussian_source_metadata':scene.metadata,'count':count,
              'units':'input coordinate units; metres only when verified','crs':scene.metadata.get('crs') or scene.metadata.get('source',{}).get('crs'),
              'coordinate_frame':scene.metadata.get('coordinate_frame','source Gaussian coordinate frame'),
              'warnings':['This conversion cannot recover original LiDAR points, intensity, scan stations or survey accuracy. Colors use direction-neutral DC RGB.']}
    return Dataset(points,colors,{'gaussian_id':ids,'gaussian_opacity':alphas},metadata).validate()
