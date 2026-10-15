"""Time scene construction, reset and the first ten physics/control steps."""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90,repeat=True)
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':1,'multi_gpu':False,'sync_loads':False})
try:
 import numpy as np
 import torch
 torch.set_num_threads(4)
 from dexgrasp_isaac.task import YCB,ObjectData
 from dexgrasp_isaac.simulation import CrossHandYCB
 objects=[ObjectData(p.name) for p in sorted(YCB.iterdir()) if p.is_dir()]
 t=time.monotonic();print('PROFILE start',flush=True)
 env=CrossHandYCB(objects,'cuda:1',hand='leap')
 env.world.get_physics_context().enable_fabric(True)
 print('PROFILE scene',time.monotonic()-t,flush=True);t=time.monotonic()
 env.reset(0)
 print('PROFILE reset',time.monotonic()-t,flush=True)
 for i in range(10):
  t=time.monotonic();env.step(np.zeros((len(objects),22)),i);print('PROFILE step',i,time.monotonic()-t,flush=True)
 env.close()
except BaseException:
 import traceback;traceback.print_exc()
finally:app.close()
