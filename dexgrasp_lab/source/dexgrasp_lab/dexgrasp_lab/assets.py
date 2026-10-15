"""Isaac Lab converters for native floating hands and single-rigid-body object meshes."""
import hashlib,json,fcntl
import xml.etree.ElementTree as ET
import numpy as np
from .paths import CACHE
from .hands import usd_name

VERSION=2

def hand_usd(spec):
    from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
    source=spec['path'];key=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    directory=CACHE/f'hand_{spec["name"]}_{key}_v{VERSION}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'convert.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'complete.json').exists():return directory/'hand.usd'
        root=ET.parse(source).getroot()
        root.set('name','hand')
        wrist=set(spec['order'][:6])
        keep={spec['root_link']}
        for _ in root.findall('link'):
            for joint in root.findall('joint'):
                if joint.find('parent').get('link') in keep:
                    keep.add(joint.find('child').get('link'))
        discard={link.get('name') for link in root.findall('link')}-keep
        for joint in root.findall('joint'):
            if joint.find('parent').get('link') not in keep or joint.find('child').get('link') not in keep:
                root.remove(joint)
        changes=[]
        for link in root.findall('link'):
            if link.get('name') in discard:root.remove(link);continue
            # Match RaiSim's last inertial block when there are duplicate blocks.
            for extra in link.findall('inertial')[:-1]:link.remove(extra)
            inertial=link.find('inertial')
            if inertial is None:
                inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='0.0001')
                ET.SubElement(inertial,'inertia',ixx='1e-7',iyy='1e-7',izz='1e-7',ixy='0',ixz='0',iyz='0')
                changes.append({'link':link.get('name'),'reason':'missing inertial, tiny fixed marker link'})
            mass=inertial.find('mass')
            if float(mass.get('value'))<=0:
                mass.set('value','.0001');changes.append({'link':link.get('name'),'reason':'nonpositive mass'})
            inertia=inertial.find('inertia')
            m=np.array([[float(inertia.get('i'+i+j,inertia.get('i'+j+i,'0'))) for j in 'xyz'] for i in 'xyz'])
            val,vec=np.linalg.eigh(m)
            if val.min()<1e-8:
                changes.append({'link':link.get('name'),'inertia_eigenvalues':val.tolist()})
                m=(vec*np.maximum(val,1e-8))@vec.T
                for i,a in enumerate('xyz'):
                    for j,b in enumerate('xyz'):
                        if j>=i:inertia.set('i'+a+b,str(m[i,j]))
            link.set('name',usd_name(link.get('name')))
        for joint in root.findall('joint'):
            joint.set('name',usd_name(joint.get('name')))
            for tag in ['parent','child']:
                node=joint.find(tag);node.set('link',usd_name(node.get('link')))
            mimic=joint.find('mimic')
            if mimic is not None:joint.remove(mimic)
            limit=joint.find('limit')
            if limit is not None and 'velocity' not in limit.attrib:limit.set('velocity','10')
        for mesh in root.findall('.//mesh'):
            name=mesh.get('filename')
            if name.startswith('package://'):name=name[len('package://'):].split('/',1)[1]
            path=(source.parent/name).resolve(strict=True)
            if usd_name(path.stem)!=path.stem:
                mesh_dir=directory/'meshes'/hashlib.sha256(path.read_bytes()).hexdigest()[:12]
                mesh_dir.mkdir(parents=True,exist_ok=True)
                alias=mesh_dir/(usd_name(path.stem)+path.suffix)
                if not alias.exists():alias.symlink_to(path)
                path=alias
            mesh.set('filename',str(path))
        urdf=directory/'hand.urdf';ET.ElementTree(root).write(urdf)
        cfg=UrdfConverterCfg(asset_path=str(urdf),usd_dir=str(directory),usd_file_name='hand.usd',
            fix_base=False,root_link_name=usd_name(spec['root_link']),merge_fixed_joints=False,
            make_instanceable=False,force_usd_conversion=True,collider_type='convex_hull',self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.,damping=.2)))
        converted=UrdfConverter(cfg)
        from pxr import Usd,UsdPhysics
        stage=Usd.Stage.Open(converted.usd_path)
        if not stage.GetDefaultPrim() or not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()):
            raise RuntimeError(f'Incomplete hand conversion: {source}')
        (directory/'complete.json').write_text(json.dumps({'source':str(source),'removed_wrist_joints':list(wrist),
            'native_free_base':True,'adjustments':changes},indent=2))
        return converted.usd_path


def object_usd(data):
    from isaaclab.sim.converters import MeshConverter,MeshConverterCfg
    from isaaclab.sim import RigidBodyPropertiesCfg,MassPropertiesCfg,CollisionPropertiesCfg,ConvexDecompositionPropertiesCfg
    directory=CACHE/f'object_{data["name"]}_{data["sha256"][:12]}_v{VERSION}';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'convert.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'complete.json').exists():return directory/'object.usd'
        mesh=directory/'object.obj';data['mesh'].export(mesh)
        cfg=MeshConverterCfg(asset_path=str(mesh),usd_dir=str(directory),usd_file_name='object.usd',
            make_instanceable=False,force_usd_conversion=True,
            mass_props=MassPropertiesCfg(mass=data['mass']),rigid_props=RigidBodyPropertiesCfg(),
            collision_props=CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            mesh_collision_props=ConvexDecompositionPropertiesCfg())
        converted=MeshConverter(cfg)
        (directory/'complete.json').write_text(json.dumps({k:v for k,v in data.items() if k not in ['mesh','points','center_offset']}|{'center_offset':data['center_offset'].tolist()},indent=2))
        return converted.usd_path
