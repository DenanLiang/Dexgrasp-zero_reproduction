"""Object geometry and the official filtered-CMAP split ONLY. No DRO grasp labels/model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from .paths import DATA, CACHE


def split_objects(split='train'):
    if split not in ('train','test'): raise ValueError('split must be train or test')
    path=DATA/'CMapDataset_filtered/split_train_validate_objects.json'
    splits=json.loads(path.read_text())
    assert not set(splits['train']) & set(splits['validate']), 'Leaking object split'
    assert len(splits['train'])==len(set(splits['train']))
    assert len(splits['validate'])==len(set(splits['validate']))
    return splits['train' if split=='train' else 'validate']


def object_mesh_path(name):
    dataset,obj=name.split('+')
    path=DATA/'data_urdf/object'/dataset/obj/f'{obj}.stl'
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def geometry(name, seed=1, num_points=200):
    path=object_mesh_path(name)
    mesh=trimesh.load(path,force='mesh',process=False)
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)==0: raise ValueError(f'Invalid mesh: {name}')
    center=mesh.centroid.copy();mesh.apply_translation(-center)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rng_seed=int.from_bytes(hashlib.sha256(f'{name}:{seed}'.encode()).digest()[:4],'little')
    points,_=trimesh.sample.sample_surface(mesh,num_points,seed=rng_seed)
    mass=float(np.clip(mesh.convex_hull.volume*500.,.01,2.))
    return {'name':name,'mesh':mesh,'points':points.astype(np.float32),'center_offset':center,
            'mass':mass,'min_z':float(mesh.bounds[0,2]),'radius':float(np.linalg.norm(mesh.extents)/2),
            'source':str(path),'sha256':digest}


def manifest(split):
    path=DATA/'CMapDataset_filtered/split_train_validate_objects.json'
    return {'split':split,'source_key':'train' if split=='train' else 'validate',
            'split_file':str(path),'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'objects':split_objects(split),'data_usage':'object STL meshes and object split JSON only; no CMAP tensors, demonstrations or DRO algorithms'}
