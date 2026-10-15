import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/dexgrasp_lab'))
import torch
from dexgrasp_lab.ppo import gae
from dexgrasp_lab.dataset import split_objects
from dexgrasp_lab.hands import hand_spec

class ContractTests(unittest.TestCase):
    def test_timeout_bootstrap_and_terminal_boundary(self):
        r=torch.tensor([[1.],[2.],[3.]]);v=torch.tensor([[10.],[20.],[30.]])
        nxt=torch.tensor([[20.],[30.],[40.]])
        terminated=torch.tensor([[False],[True],[False]])
        done=torch.tensor([[True],[True],[False]])
        a,returns=gae(r,v,nxt,terminated,done,.9,1.)
        torch.testing.assert_close(a,torch.tensor([[9.],[-18.],[9.]]))
        torch.testing.assert_close(returns,torch.tensor([[19.],[2.],[39.]]))
    def test_barrett_contract_includes_palm_contact(self):
        s=hand_spec('barrett')
        self.assertEqual((s['act_dim'],len(s['contact_ids']),len(s['links']),s['obs_dim']),(14,9,9,130))
        self.assertEqual(s['contact_links'][0],s['root_link'])
    def test_held_out_objects_never_enter_training_split(self):
        train,test=split_objects('train'),split_objects('test')
        self.assertEqual((len(train),len(test)),(48,10))
        self.assertFalse(set(train)&set(test))
        with self.assertRaises(ValueError):split_objects('validate')

if __name__=='__main__':unittest.main()
