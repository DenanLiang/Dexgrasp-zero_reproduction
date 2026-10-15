"""CPU checks of data separation, dimensions, actor/critic and each training hand's gradients."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/dexgrasp_lab'))
import torch
from dexgrasp_lab.hands import TRAIN_HANDS,hand_spec
from dexgrasp_lab.dataset import split_objects,geometry,manifest
from dexgrasp_lab.policy import ActorCritic
torch.set_num_threads(4);torch.manual_seed(1)
model=ActorCritic();report={'splits':{},'hands':{}}
for split in ['train','test']:
 names=split_objects(split)
 for name in names:
  g=geometry(name);assert g['points'].shape==(200,3)
 report['splits'][split]={'count':len(names),'geometry_validated':len(names),**manifest(split)}
for name in TRAIN_HANDS:
 s=hand_spec(name);obs=torch.randn(4,model.obs_dim)*.05;ht=torch.full((4,),s['hand_type'])
 a,lp,en,v=model.act(obs,ht)
 assert a.shape==(4,model.action_dim) and torch.isfinite(a).all()
 assert not a[:,s['act_dim']:].count_nonzero()
 loss=lp.mean()+v.square().mean()
 model.zero_grad();loss.backward()
 grad=sum(float(p.grad.square().sum()) for p in model.actor.parameters() if p.grad is not None)**.5
 assert grad>0 and torch.isfinite(lp).all() and torch.isfinite(v).all()
 report['hands'][name]={'obs_dim':s['obs_dim'],'action_dim':s['act_dim'],'contact_channels':len(s['contact_links']),'keypoints':len(s['links']),'actor_gradient_norm':grad}
assert report['hands']['barrett']['obs_dim']==130
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/contracts.json').write_text(json.dumps(report,indent=2))
print('CONTRACTS_PASS',report['hands'])
