"""Check all poses read by the adapter equal the archive exported in Python 3.10."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dexgrasp_isaac.task import YCB,HANDS,ObjectData,reference_poses,PROJECT
arrays,metadata=reference_poses()
objects=[ObjectData(n) for n in metadata['objects']]
report={}
for hand in HANDS:
    actual=np.array([[np.concatenate([x.ravel() for x in o.initial(u,hand)]) for o in objects] for u in range(25)])
    error=float(np.max(np.abs(actual-arrays[hand])))
    assert error==0,(hand,error)
    report[hand]={'poses':1950,'max_abs_difference':error}
(PROJECT/'results/initial_pose_validation.json').write_text(json.dumps(report,indent=2))
print('ALL_15600_INITIAL_POSES_PASS (bitwise identical to source export)')
