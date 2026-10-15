from pathlib import Path
import os
PROJECT = Path(__file__).resolve().parents[3]
WORKSPACE = PROJECT.parent
SOURCE = Path(os.environ.get('DEXGRASP_SOURCE', WORKSPACE/'dexgrasp_310/GraspXL/raisimGymTorch/dexgrasp_zero'))
DATA = Path(os.environ.get('DRO_OBJECT_ROOT', WORKSPACE/'DRO_data'))
CACHE = PROJECT/'cache'
