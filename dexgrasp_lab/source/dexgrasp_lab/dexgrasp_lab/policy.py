"""Original DexGrasp-Zero GNN actor/critic with masked heterogeneous action distributions."""
import torch
from torch import nn
from torch.distributions import Normal
import yaml
from pathlib import Path
from .hands import hand_spec,TRAIN_HANDS
from .algorithm.urdf_parser import URDFGraphParser
from .algorithm.graph_data_builder import GraphDataBuilder
from .algorithm.networks import GNN,GNNCritic

class ActorCritic(nn.Module):
    def __init__(self,hands=TRAIN_HANDS,config=None):
        super().__init__()
        self.hands={hand_spec(h)['hand_type']:hand_spec(h) for h in hands}
        data={i:s['data'].copy() for i,s in self.hands.items()}
        self.action_dim=max(s['act_dim'] for s in self.hands.values())
        self.obs_dim=max(s['obs_dim'] for s in self.hands.values())
        parser=URDFGraphParser()
        for i,s in self.hands.items():parser.parse_urdf(str(s['path']),i)
        graph=GraphDataBuilder(parser,data,self.action_dim,'v4')
        for d in data.values():d['adj_matrix'].requires_grad_(False)
        self.config=config or yaml.safe_load((Path(__file__).parent/'configs/model_gnn.yaml').read_text())
        self.actor=GNN.from_config(self.config,graph,data)
        self.critic=GNNCritic.from_config(self.config,graph,data)
        self.log_std=nn.Parameter(torch.zeros(self.action_dim))

    def forward(self,obs,hand_types):
        means=obs.new_zeros((len(obs),self.action_dim));values=obs.new_zeros(len(obs));mask=torch.zeros_like(means)
        for ht,s in self.hands.items():
            idx=(hand_types==ht).nonzero().flatten()
            if not len(idx):continue
            means[idx]=self.actor(obs[idx,:s['obs_dim']],ht)
            values[idx]=self.critic(obs[idx,:s['obs_dim']],ht).flatten()
            mask[idx,:s['act_dim']]=1.
        return means,values,mask

    def act(self,obs,hand_types,actions=None,deterministic=False):
        means,values,mask=self(obs,hand_types)
        distribution=Normal(means,self.log_std.exp().expand_as(means))
        if actions is None:actions=(means if deterministic else distribution.sample())*mask
        log_prob=(distribution.log_prob(actions)*mask).sum(-1)
        entropy=(distribution.entropy()*mask).sum(-1)
        return actions,log_prob,entropy,values
