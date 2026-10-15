"""Local URDF -> USD conversion; never edits the original assets."""
from pathlib import Path
import json
import fcntl
import numpy as np
import xml.etree.ElementTree as ET
from .task import PROJECT, digest, usd_name

CACHE_VERSION = 6


def convert(source, hand=False):
    # USD crate files must never be opened while another process is writing them.
    # Both GPUs share this cache, including the nested importer-generated layers.
    lock_dir = PROJECT/'cache'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir/'conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _convert_locked(source,hand)


def _convert_locked(source, hand=False):
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd
    enable_extension('isaacsim.asset.importer.urdf')
    source = Path(source)
    key = f'{source.stem}_{digest(source)[:12]}_v{CACHE_VERSION}'
    directory = PROJECT / 'cache' / key
    directory.mkdir(parents=True, exist_ok=True)
    usd = directory / 'asset.usd'
    if usd.is_file() and (directory/'conversion.json').is_file():
        stage = Usd.Stage.Open(str(usd))
        if stage and stage.GetDefaultPrim():
            return usd
    tree = ET.parse(source)
    tree.getroot().set('name', 'hand' if hand else 'object')
    for mesh in tree.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename.startswith('package://'):
            filename = filename[len('package://'):].split('/',1)[1]
        path = (source.parent / filename).resolve(strict=True)
        if usd_name(path.stem) != path.stem:
            mesh_dir = directory/'meshes'/digest(path)[:12]
            mesh_dir.mkdir(parents=True,exist_ok=True)
            alias = mesh_dir/(usd_name(path.stem)+path.suffix)
            if not alias.exists(): alias.symlink_to(path)
            for dependency in path.parent.iterdir():
                if dependency.is_file() and dependency.suffix.lower() in ['.mtl','.png','.jpg','.jpeg']:
                    link = mesh_dir/dependency.name
                    if not link.exists(): link.symlink_to(dependency.resolve())
            path = alias
        mesh.set('filename', str(path))
    for j in tree.findall('joint'):
        j.set('name', usd_name(j.get('name')))
        for tag in ['parent','child']:
            j.find(tag).set('link', usd_name(j.find(tag).get('link')))
        mimic = j.find('mimic')
        if mimic is not None:
            j.remove(mimic)  # Source RaiSim treats these as independently actuated DOFs.
        lim = j.find('limit')
        if lim is not None and 'velocity' not in lim.attrib:
            lim.set('velocity', '100')
    # PhysX cannot represent zero-mass intermediate articulated links.
    # Preserve original source URDF for the GNN, regularize only the import copy.
    regularized = []
    inertia_regularized = []
    for link in tree.findall('link'):
        link.set('name', usd_name(link.get('name')))
        # RaiSim reads the last inertial block; Schunk has duplicate blocks.
        for extra in link.findall('inertial')[:-1]: link.remove(extra)
        mass = link.find('inertial/mass')
        if mass is not None and float(mass.get('value')) == 0:
            mass.set('value', '0.00001')
            inertia = link.find('inertial/inertia')
            for axis in ['ixx','iyy','izz']:
                inertia.set(axis, '0.00000001')
            regularized.append(link.get('name'))
        for inertial in link.findall('inertial'):
            inertia = inertial.find('inertia')
            matrix = np.array([[float(inertia.get('i'+i+j,inertia.get('i'+j+i,'0'))) for j in 'xyz'] for i in 'xyz'])
            values, vectors = np.linalg.eigh(matrix)
            if values.min() <= 0:
                inertia_regularized.append({'link':link.get('name'),'original_eigenvalues':values.tolist()})
                matrix = (vectors*np.maximum(values,1e-8))@vectors.T
                for i,axis1 in enumerate('xyz'):
                    for j,axis2 in enumerate('xyz'):
                        if j>=i: inertia.set('i'+axis1+axis2,str(matrix[i,j]))
    temp = directory / 'source.urdf'
    tree.write(temp)
    _, config = omni.kit.commands.execute('URDFCreateImportConfig')
    config.merge_fixed_joints = True
    config.make_default_prim = True
    config.fix_base = hand
    config.self_collision = hand
    config.import_inertia_tensor = True
    config.convex_decomp = True
    config.parse_mimic = False
    config.default_drive_strength = 100 if hand else 0
    config.default_position_drive_damping = 0.2 if hand else 0
    ok, path = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(temp),
                                        import_config=config, dest_path=str(usd))
    if not ok:
        raise RuntimeError(f'URDF import failed: {source}')
    stage = Usd.Stage.Open(str(usd))
    if not stage.GetDefaultPrim():
        raise RuntimeError(f'Imported USD has no default prim: {usd}, returned {path}')
    (directory / 'conversion.json').write_text(json.dumps({
        'source':str(source), 'sha256':digest(source), 'regularized_links':regularized,
        'inertia_regularized':inertia_regularized,'min_inertia_eigenvalue':1e-8,
        'dummy_mass_kg':1e-5, 'dummy_inertia_kg_m2':1e-8,
        'convex_decomposition':True, 'numeric_joint_prefix':'joint_',
    }, indent=2))
    return usd
