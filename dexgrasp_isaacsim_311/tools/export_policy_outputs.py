"""Compare the same checkpoint and inputs across the two Torch/PyG environments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
torch.set_num_threads(4)
from dexgrasp_isaac.task import HANDS,hand_spec,load_policy
rng=np.random.default_rng(321)
outputs={}
for h in HANDS:
 s=hand_spec(h); policy=load_policy('cpu',s)
 obs=rng.normal(0,.1,size=(8,s['obs_dim'])).astype(np.float32)
 with torch.inference_mode():outputs[h]=policy(torch.from_numpy(obs),s['hand_type']).numpy()
np.savez_compressed(sys.argv[1],**outputs)
print('POLICY_OUTPUTS_SAVED',sys.argv[1])
