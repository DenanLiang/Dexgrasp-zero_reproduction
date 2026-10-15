from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False,'extra_args': ['--/app/fastShutdown=true']})
from pathlib import Path
import xml.etree.ElementTree as ET
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.asset.importer.urdf')
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdPhysics
root=Path(__file__).resolve().parents[1]
src=root.parent/'dexgrasp_310/GraspXL/raisimGymTorch/dexgrasp_zero/rsc/leap_hand/leap_hand_right_envusing.urdf'
cache=root/'cache'; cache.mkdir(exist_ok=True)
tree=ET.parse(src)
for mesh in tree.findall('.//mesh'):
 mesh.set('filename',str((src.parent/mesh.get('filename')).resolve()))
for j in tree.findall('joint'):
 if j.get('name').isdigit(): j.set('name', 'joint_'+j.get('name'))
 lim=j.find('limit')
 if lim is not None: lim.set('velocity','100')
for link in tree.findall('link'):
 mass=link.find('inertial/mass')
 if mass is not None and float(mass.get('value'))==0:
  mass.set('value','0.00001')
  inertia=link.find('inertial/inertia')
  for axis in ['ixx','iyy','izz']: inertia.set(axis,'0.00000001')
tree.write(cache/'hand.urdf')
_,cfg=omni.kit.commands.execute('URDFCreateImportConfig')
cfg.make_default_prim=True; cfg.merge_fixed_joints=True; cfg.fix_base=True; cfg.import_inertia_tensor=True; cfg.self_collision=True
cfg.default_drive_strength=100; cfg.default_position_drive_damping=0.2
ok,path=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=str(cache/'hand.urdf'),import_config=cfg,dest_path=str(cache/'hand.usd'))
print('IMPORTED',ok,path,flush=True)
world=World(physics_dt=0.0025,rendering_dt=0.01,backend='numpy',device='cpu')
prim=add_reference_to_stage(str(cache/'hand.usd'),'/World/hand')
for p in Usd.PrimRange(prim):
 if p.HasAPI(UsdPhysics.ArticulationRootAPI): print('ROOT',p.GetPath(),flush=True)
hand=world.scene.add(Articulation('/World/hand',name='hand'))
world.reset()
print('DOFS',hand.dof_names,flush=True)
print('BODIES',hand.body_names,flush=True)
print('Q',hand.get_joint_positions(),flush=True)
for _ in range(4): world.step(render=False)
print('Q_AFTER',hand.get_joint_positions(),flush=True)
app.close()
