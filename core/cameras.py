"""Explicit pinhole calibration: world-to-camera OpenCV RDF, metres."""
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from av_common.data import fingerprint

@dataclass
class Camera:
    K: np.ndarray
    world_to_camera: np.ndarray
    image: np.ndarray
    image_path: str
    source: dict=field(default_factory=dict)

    def validate(self):
        if self.K.shape!=(3,3) or self.world_to_camera.shape!=(4,4) or not np.isfinite(self.K).all() or not np.isfinite(self.world_to_camera).all():
            raise ValueError('Camera matrices must be finite 3 × 3 K and 4 × 4 world_to_camera.')
        if self.K[0,0]<=0 or self.K[1,1]<=0 or not np.allclose(self.K[2],[0,0,1]) or not np.allclose(self.world_to_camera[3],[0,0,0,1]):
            raise ValueError('Invalid pinhole intrinsics or camera transform.')
        rotation=self.world_to_camera[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):
            raise ValueError('Camera pose must be a proper rigid world-to-camera transform.')
        if self.image.ndim!=3 or self.image.shape[2]!=3 or not np.isfinite(self.image).all() or self.image.min()<0 or self.image.max()>1:
            raise ValueError('Expected finite normalized RGB images.')
        return self


def load_cameras(path,max_side=256):
    path=Path(path).resolve(); manifest=json.loads(path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('convention')!='opencv_world_to_camera' or manifest.get('units')!='metres':
        raise ValueError('Calibration requires schema_version 1, convention opencv_world_to_camera and units metres.')
    if not 16<=max_side<=4096: raise ValueError('Image-side budget must be 16–4096 pixels.')
    if len(manifest.get('cameras',[]))>256: raise MemoryError('Use at most 256 calibrated views per job.')
    cameras=[]; stamps=[]; decoded_bytes=0
    for record in manifest.get('cameras',[]):
        if any(float(value)!=0 for value in record.get('distortion',[])): raise ValueError('Undistort and recalibrate images before pinhole reconstruction.')
        image_path=(path.parent/record['image']).resolve()
        with Image.open(image_path) as opened:
            image=opened.convert('RGB'); old_width,old_height=image.size
            scale=min(1,max_side/max(image.size))
            if scale<1: image=image.resize((max(1,round(old_width*scale)),max(1,round(old_height*scale))),Image.Resampling.LANCZOS)
            K=np.array(record['K'],dtype=float)
            if K.shape!=(3,3): raise ValueError('Invalid intrinsics shape.')
            K[0]*=image.width/old_width; K[1]*=image.height/old_height
            camera=Camera(K,np.array(record['world_to_camera'],float),np.asarray(image,dtype='float32')/255,str(image_path),{'image':fingerprint(image_path),'calibration':fingerprint(path)}).validate()
            decoded_bytes+=camera.image.nbytes
            if decoded_bytes>512*1024**2: raise MemoryError('Decoded images exceed the 512 MiB job budget. Reduce image resolution or the view set.')
            cameras.append(camera); stamps.append(fingerprint(image_path))
    if len(cameras)<2: raise ValueError('At least two calibrated RGB views are required; do not infer poses from unregistered images.')
    identity=hashlib.sha256(json.dumps({'manifest':fingerprint(path),'images':stamps},sort_keys=True).encode()).hexdigest()
    return cameras,identity


def verify_camera_sources(cameras):
    for camera in cameras:
        for stamp in camera.source.values():
            if fingerprint(stamp['path'])!=stamp:
                raise ValueError('Calibration or RGB input changed during optimization. Restore the original inputs or start a new job.')
