"""Global voxel indexing and bounded neighbourhood features, independent of tiles."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from av_common.data import allocate,check_cancel


@dataclass
class VoxelCloud:
    points: np.ndarray
    keys: np.ndarray
    representative_ids: np.ndarray
    inverse: np.ndarray
    size: float


def voxelize(points,folder,size=.05,max_voxels=200_000,cancel=lambda:False,progress=lambda message:None,chunk_size=100_000):
    if not np.isfinite(size) or size<=0 or max_voxels<10: raise ValueError('Positive voxel size and at least ten solver voxels are required.')
    for attempt in range(12):
        lookup={}; reps=[]; keys=[]; over_budget=False
        inverse=allocate(folder,'voxel_index',(len(points),),'int32')
        for start in range(0,len(points),chunk_size):
            check_cancel(cancel)
            part=np.asarray(points[start:start+chunk_size])
            scaled=np.floor(part/size)
            if not np.isfinite(scaled).all() or np.max(np.abs(scaled))>9e18:
                raise ValueError('Coordinates exceed stable voxel-index range; use an explicit local frame.')
            unique,first,back=np.unique(scaled.astype('int64'),axis=0,return_index=True,return_inverse=True)
            ids=np.empty(len(unique),dtype='int32')
            for i,key in enumerate(unique):
                key=tuple(key)
                if key not in lookup:
                    if len(lookup)>=max_voxels: over_budget=True; break
                    lookup[key]=len(lookup); keys.append(key); reps.append(start+int(first[i]))
                ids[i]=lookup[key]
            if over_budget: break
            inverse[start:start+len(part)]=ids[back]
        if not over_budget:
            inverse.flush(); rep_ids=np.asarray(reps,dtype='int64')
            return VoxelCloud(np.asarray(points[rep_ids]),np.asarray(keys,dtype='int64'),rep_ids,inverse,size)
        size*=1.6; progress(f'Adapting solver resolution to {size:.4f} m for the voxel budget')
    raise MemoryError('Scene exceeds the bounded voxel budget. Increase voxel size or partition the source.')


def neighbourhood_features(points,k=24,cancel=lambda:False):
    if len(points)<4: raise ValueError('At least four distinct geometric samples are required.')
    tree=cKDTree(points); k=min(k,len(points))
    eigenvalues=np.empty((len(points),3)); rotations=np.empty((len(points),3,3))
    for start in range(0,len(points),4096):
        check_cancel(cancel); end=min(start+4096,len(points))
        _,ids=tree.query(points[start:end],k=k,workers=1)
        neighbours=points[ids]; delta=neighbours-neighbours.mean(axis=1,keepdims=True)
        covariance=np.einsum('nki,nkj->nij',delta,delta)/max(k-1,1)
        values,vectors=np.linalg.eigh(covariance)
        reflected=np.linalg.det(vectors)<0
        vectors[reflected,:,2]*=-1
        eigenvalues[start:end]=np.maximum(values,0); rotations[start:end]=vectors
    return eigenvalues,rotations
