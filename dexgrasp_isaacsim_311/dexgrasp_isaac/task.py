"""Simulator-independent policy, geometry and evaluation protocol.

The original package is reused for graph construction, network and initial poses.
No RaiSim binary is imported. Angles use the original intrinsic XYZ convention.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
DEX = Path(os.environ.get('DEX_ROOT', PROJECT.parent / 'dexgrasp_310/GraspXL/raisimGymTorch/dexgrasp_zero'))
sys.path.insert(0, str(DEX.parent))
YCB = Path(os.environ.get('YCB_ROOT', DEX / 'rsc/ycb_crossDex'))
HAND = Path(os.environ.get('LEAP_URDF', DEX / 'rsc/leap_hand/leap_hand_right_envusing.urdf'))
WEIGHT = Path(os.environ.get('DEX_MODEL', DEX / 'model/multi_hand/full_2700_r.pt'))
CONFIG = Path(os.environ.get('DEX_MODEL_CONFIG', DEX / 'model/multi_hand/model_gnn_cfg_L10_v4.yaml'))
HANDS = json.loads((PROJECT / 'dexgrasp_isaac/hands.json').read_text())


def hand_spec(name):
    spec = HANDS[name].copy()
    spec['name'] = name
    spec['path'] = HAND if name == 'leap' else DEX / spec['urdf']
    spec['act_dim'] = len(spec['joints'])
    spec['obs_dim'] = 12 + 2*spec['act_dim'] + 4*len(spec['contacts']) + 6*len(spec['frames'])
    return spec


def usd_name(name):
    import re
    return ('joint_' if name[0].isdigit() else '') + re.sub(r'[^a-zA-Z0-9_]', '_', name)


DT = 0.0025
CONTROL_DT = 0.01


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(seed, name, update=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{update}'.encode()).digest()[:4], 'little')


def mat(quat):
    """Scalar-first quaternion to matrix, including batches."""
    return Rotation.from_quat(np.asarray(quat)[..., [1,2,3,0]]).as_matrix()


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[..., [3,0,1,2]]


def mv(matrix, vector):
    return np.einsum('...ij,...j->...i', matrix, vector)


def transpose(matrix):
    return np.swapaxes(matrix, -1, -2)


class Kinematics:
    """URDF forward kinematics in policy joint order (not importer DOF order)."""
    def __init__(self, spec=None):
        self.spec = spec or hand_spec('leap')
        self.order = self.spec['joints']
        root = ET.parse(self.spec['path']).getroot()
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.mass = sum(float(link.findall('inertial/mass')[-1].get('value')) for link in root.findall('link') if link.find('inertial/mass') is not None)
        self.lower = np.array([float(self.joints[n].find('limit').get('lower')) for n in self.order])
        self.upper = np.array([float(self.joints[n].find('limit').get('upper')) for n in self.order])
        self.chain = []
        children = {}
        for joint in self.joints.values():
            children.setdefault(joint.find('parent').get('link'), []).append(joint)
        def visit(parent):
            for j in children.get(parent, []):
                origin = j.find('origin')
                xyz = np.fromstring(origin.get('xyz','0 0 0'), sep=' ') if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get('rpy','0 0 0'), sep=' ') if origin is not None else np.zeros(3)
                axis = j.find('axis')
                axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1.,0,0])
                self.chain.append((j.get('name'), parent, j.find('child').get('link'), j.get('type'),
                                   xyz, Rotation.from_euler('xyz', rpy).as_matrix(), axis))
                visit(j.find('child').get('link'))
        visit(self.spec['base_link'])
        assert [x[0] for x in self.chain if x[3] != 'fixed'] == self.order

    def forward(self, q, base_pos, base_rot):
        poses = {self.spec['base_link']: (base_pos, base_rot)}
        frames = {}
        for name, parent, child, kind, xyz, rot, axis in self.chain:
            pp, pr = poses[parent]
            p, r = pp + mv(pr, xyz), pr @ rot
            if kind != 'fixed':
                value = q[:, self.order.index(name)]
                if kind == 'prismatic':
                    p = p + mv(r, value[:, None] * axis)
                else:
                    r = r @ Rotation.from_rotvec(value[:, None] * axis).as_matrix()
            poses[child] = frames[name] = (p, r)
        return np.stack([frames[n][0] for n in self.spec['frames']], axis=1), np.stack([frames[n][1] for n in self.spec['frames']], axis=1)


_REFERENCE = None


def reference_poses():
    global _REFERENCE
    if _REFERENCE is None:
        path = PROJECT/'reference/initial_poses.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != metadata['archive_sha256']:
            raise ValueError('Reference pose archive checksum mismatch; regenerate in dexgrasp310')
        if digest(DEX/'helper/initial_pose_final.py') != metadata['helper_sha256']:
            raise ValueError('Initial pose helper changed; regenerate reference poses in dexgrasp310')
        with np.load(path) as data:
            arrays = {h:data[h].copy() for h in metadata['hands']}
        for h,a in arrays.items():
            if a.shape != (25,len(metadata['objects']),18) or not np.isfinite(a).all():
                raise ValueError(f'Invalid reference pose array: {h}')
        _REFERENCE = (arrays,metadata)
    return _REFERENCE


class ObjectData:
    def __init__(self, name, seed=1):
        import joblib
        import trimesh
        self.name = name
        self.path = YCB / name
        for f in [f'{name}.urdf', 'top_watertight_tiny.obj', 'bottom_watertight_tiny.obj', 'lowest_point_new.txt']:
            if not (self.path / f).is_file():
                raise FileNotFoundError(self.path / f)
        self.aff = trimesh.load_mesh(self.path / 'top_watertight_tiny.obj')
        self.nonaff = trimesh.load_mesh(self.path / 'bottom_watertight_tiny.obj')
        self.has_nonaff_contact = float(ET.parse(self.path/f'{name}.urdf').find("link[@name='bottom']/inertial/mass").get('value')) > .001
        self.has_nonaff = np.linalg.norm(self.nonaff.centroid - [0.346408]*3) >= 0.01
        self.labels = joblib.load(YCB / f'{name}_label.pkl')
        self.lowest = float((self.path / 'lowest_point_new.txt').read_text())
        # Fixed per-object samples independent of batch size, order and resume.
        np.random.seed(stable_seed(seed, name))
        self.points = np.stack([trimesh.sample.sample_surface(m, 200)[0] for m in [self.aff, self.nonaff]])
        self.seed = seed
        if os.environ.get('DEX_RAW_INITIAL_POSES') != '1':
            _, metadata = reference_poses()
            expected = metadata['inputs'][name]
            actual = {'label':digest(YCB/f'{name}_label.pkl'),
                      'mesh':digest(self.path/'top_watertight_tiny.obj'),
                      'lowest':digest(self.path/'lowest_point_new.txt')}
            if actual != expected:
                raise ValueError(f'{name}: source assets/labels changed; regenerate reference poses in dexgrasp310')

    def initial(self, update, hand='leap'):
        if os.environ.get('DEX_RAW_INITIAL_POSES') != '1':
            arrays, metadata = reference_poses()
            row = arrays[hand][update,metadata['objects'].index(self.name)].copy()
            return row[:3],row[3:12].reshape(3,3),row[12:15],row[15:18]
        from dexgrasp_zero.helper.initial_pose_final import get_initial_pose_faive_label
        label = self.labels[f'{self.name}_{update}']  # Fail on missing trials, never silently resample.
        np.random.seed(stable_seed(self.seed, self.name, update))
        if hand == 'barrett':
            obj = np.array([1.,0.,0.502-self.lowest])
            return obj+[0,0,.3], Rotation.from_euler('XYZ',[3.14,0,0]).as_matrix(), obj, np.asarray(label['target_center']).reshape(3)
        rot, pos, bias = get_initial_pose_faive_label(
            self.aff, self.nonaff if self.has_nonaff else None, hand,
            label['target_direction'], label['rand_offset'], label['target_center'])
        obj = np.array([1., 0., 0.502-self.lowest])
        return obj + pos.reshape(3), Rotation.from_rotvec(rot.reshape(3)).as_matrix(), obj, bias.reshape(3)


def load_policy(device='cuda:0', spec=None):
    spec = spec or hand_spec('leap')
    import torch
    import yaml
    from dexgrasp_zero.helper.constants import get_data_dict, HandType
    from dexgrasp_zero.helper.urdf_parser import URDFGraphParser
    from dexgrasp_zero.helper.graph_data_builder import GraphDataBuilder
    from dexgrasp_zero.algo.ppo.model_factory import build_model_from_config
    config = yaml.safe_load(CONFIG.read_text())
    data = get_data_dict(**{'TRAIN_'+spec['name'].upper():True})
    data[spec['hand_type']]['obs_dim'] = spec['obs_dim']
    parser = URDFGraphParser()
    parser.parse_urdf(str(spec['path']), spec['hand_type'])
    graph = GraphDataBuilder(parser, data, spec['act_dim'], version=config['version'])
    policy = build_model_from_config(config, graph, model_role='actor', data_dict=data).to(device)
    # These are the user's local, trusted training artifacts; optimizer state may contain NumPy objects.
    checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    policy.load_state_dict(checkpoint['actor_architecture_state_dict'], strict=True)
    policy.eval()
    return policy


def summarize(rows, expected):
    successful = sum(r['success'] for r in rows)
    by_object = {}
    for name in sorted({r['object'] for r in rows}):
        rs = [r for r in rows if r['object'] == name]
        by_object[name] = {'trials':len(rs), 'successes':sum(r['success'] for r in rs),
                           'success_rate':sum(r['success'] for r in rs)/len(rs)}
    metrics = ['lift_height_m','center_error_m','direction_error','angle_error_rad','contact_ratio']
    valid_rows = [r for r in rows if r['valid']]
    success_rows = [r for r in rows if r['success']]
    average = lambda rs: {k:float(np.mean([r[k] for r in rs])) if rs else None for k in metrics}
    return {'metrics_all_valid':average(valid_rows),'metrics_success':average(success_rows),'completed_trials':len(rows), 'expected_trials':expected, 'complete':len(rows)==expected,
            'successes':successful, 'success_rate':successful/len(rows) if rows else None,
            'macro_object_success_rate':np.mean([v['success_rate'] for v in by_object.values()]).item() if rows else None,
            'invalid_trials':sum(not r['valid'] for r in rows), 'per_object':by_object}
