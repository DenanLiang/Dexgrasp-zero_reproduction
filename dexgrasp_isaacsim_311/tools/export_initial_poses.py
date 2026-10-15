"""Export all saved-label initial poses for comparing the two Python environments."""
import sys,os,json,importlib.metadata
os.environ['DEX_RAW_INITIAL_POSES']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dexgrasp_isaac.task import YCB,HANDS,ObjectData,DEX,digest
objects=[ObjectData(p.name) for p in sorted(YCB.iterdir()) if p.is_dir()]
poses={}
for h in HANDS:
 rows=[]
 for u in range(25):
  rows.append([np.concatenate([p.ravel() for p in o.initial(u,h)]) for o in objects])
 poses[h]=np.asarray(rows)
 print('EXPORTED',h,poses[h].shape,flush=True)
np.savez_compressed(sys.argv[1],**poses)

metadata={'objects':[o.name for o in objects],'hands':list(HANDS),
          'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','trimesh','torch']},
          'python':sys.version,'helper_sha256':digest(DEX/'helper/initial_pose_final.py'),
          'inputs':{o.name:{'label':digest(YCB/f'{o.name}_label.pkl'),
                           'mesh':digest(o.path/'top_watertight_tiny.obj'),
                           'lowest':digest(o.path/'lowest_point_new.txt')} for o in objects},
          'layout':'wrist_position(3), wrist_rotation_matrix(9), object_position(3), target_center(3)',
          'archive_sha256':digest(sys.argv[1])}
Path(sys.argv[1]).with_suffix('.json').write_text(json.dumps(metadata,indent=2))
