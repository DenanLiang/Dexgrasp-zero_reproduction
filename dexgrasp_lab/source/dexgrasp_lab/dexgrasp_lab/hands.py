"""Canonical policy order; simulator joint names are mapped explicitly."""
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .paths import SOURCE
from .algorithm.constants import get_data_dict, get_cross_hand_spec, build_node_to_joint_sequence

TRAIN_HANDS=('shadow','allegro','barrett')
DATA=get_data_dict(TRAIN_SHADOW=True,TRAIN_ALLEGRO=True,TRAIN_BARRETT=True)


def usd_name(name):
    return ('joint_' if name[0].isdigit() else '')+re.sub('[^a-zA-Z0-9_]','_',name)


def hand_spec(name):
    if name not in TRAIN_HANDS: raise ValueError(f'Training hands: {TRAIN_HANDS}; evaluation hands are deferred')
    spec=get_cross_hand_spec(name);d=DATA[spec.hand_type]
    path=SOURCE/spec.urdf_relpath
    root=ET.parse(path).getroot();joints={j.get('name'):j for j in root.findall('joint')}
    sequence=build_node_to_joint_sequence()[spec.hand_type]
    order=[n for seq in sequence for n in seq]
    frames=d['body_parts'];links=[joints[f].find('child').get('link') for f in frames]
    contact_ids=[i for i in range(len(frames)) if i not in d['fingertips_id']]
    contact_links=[links[i] for i in contact_ids]
    dim=12+2*len(order)+4*len(contact_ids)+6*len(frames)
    assert dim==d['obs_dim'],(name,dim,d['obs_dim'])
    assert len(order)==d['act_dim']
    q=np.zeros(len(order))
    if name in ('shadow','allegro'):q[-4]=1.7
    lower=np.array([float(joints[n].find('limit').get('lower')) for n in order])
    upper=np.array([float(joints[n].find('limit').get('upper')) for n in order])
    q=np.clip(q,lower,upper)
    # Child-link FK used only to define the initial grasp center; no simulation dependency.
    poses={'world':np.eye(4)}
    remaining=list(root.findall('joint'))
    while remaining:
        progress=False
        for j in remaining[:]:
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent not in poses:continue
            t=np.eye(4);o=j.find('origin')
            if o is not None:
                t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            n=j.get('name')
            if n in order:
                a=np.fromstring(j.find('axis').get('xyz'),sep=' ');v=q[order.index(n)]
                u=np.eye(4)
                if j.get('type')=='prismatic':u[:3,3]=a*v
                else:u[:3,:3]=Rotation.from_rotvec(a*v).as_matrix()
                t=t@u
            poses[child]=poses[parent]@t;remaining.remove(j);progress=True
        if not progress:raise ValueError('Disconnected hand URDF')
    p=np.stack([poses[l][:3,3] for l in links])
    fingers={'shadow':[2,6,10,15],'allegro':[2,6,10],'barrett':[5,7]}[name]
    thumb={'shadow':21,'allegro':16,'barrett':3}[name]
    center=(p[fingers].mean(0)+p[thumb])*.5-p[0]
    return {'name':name,'hand_type':spec.hand_type,'path':path,'data':d,'order':order,
            'root_link':links[0],'links':links,'contact_links':contact_links,'contact_ids':contact_ids,
            'q0':q,'lower':lower,'upper':upper,'center':center,'act_dim':len(order),'obs_dim':dim}
