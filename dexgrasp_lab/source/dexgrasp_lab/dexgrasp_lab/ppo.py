"""Torch PPO transport for the original GNN: clipped surrogate/value loss and GAE.

No RaiSim sampler/binary dependency. Padded action dimensions do not contribute to
probabilities, entropy or KL. Timeouts bootstrap final pre-reset observations.
"""
from dataclasses import dataclass
import torch

@dataclass
class PPOCfg:
    learning_rate:float=5e-4
    gamma:float=.996
    lam:float=.95
    clip:float=.2
    value_coef:float=.5
    entropy_coef:float=0.
    epochs:int=4
    minibatches:int=4
    max_grad_norm:float=.5
    desired_kl:float=.01


def gae(rewards,values,next_values,terminated,done,gamma,lam):
    advantages=torch.zeros_like(rewards);last=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        last=delta+gamma*lam*(~done[t])*last
        advantages[t]=last
    return advantages,advantages+values


class PPO:
    def __init__(self,model,cfg):
        self.model=model;self.cfg=cfg
        self.optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)

    def collect(self,env,observation,steps):
        data={k:[] for k in ['obs','actions','log_prob','values','next_values','reward','terminated','done','mean','std','hand_type']}
        obs=observation
        with torch.no_grad():
            for _ in range(steps):
                mean,value,mask=self.model(obs,env.hand_types)
                std=self.model.log_std.exp().expand_as(mean)
                dist=torch.distributions.Normal(mean,std)
                action=dist.sample()*mask
                log_prob=(dist.log_prob(action)*mask).sum(-1)
                next_obs,reward,terminated,truncated,info=env.step(action)
                next_obs=next_obs['policy']
                # DirectRLEnv resets before returning. Use the final pre-reset observation to bootstrap.
                _,next_value,_=self.model(info['terminal_observation'],env.hand_types)
                batch={'obs':obs,'actions':action,'log_prob':log_prob,'values':value,'next_values':next_value,
                       'reward':reward,'terminated':terminated,'done':terminated|truncated,
                       'mean':mean,'std':std,'hand_type':env.hand_types}
                for k,v in batch.items():data[k].append(v.clone())
                obs=next_obs
        data={k:torch.stack(v) for k,v in data.items()}
        advantage,returns=gae(data['reward'],data['values'],data['next_values'],data['terminated'],data['done'],self.cfg.gamma,self.cfg.lam)
        data['advantages']=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        data['returns']=returns
        return data,obs

    def update(self,data):
        cfg=self.cfg;b={k:v.flatten(0,1) for k,v in data.items()};count=len(b['obs'])
        if cfg.minibatches>count:raise ValueError('Too many minibatches')
        stats=[]
        before=torch.cat([p.detach().flatten().clone() for p in self.model.actor.parameters()])
        for _ in range(cfg.epochs):
            for indices in torch.randperm(count,device=b['obs'].device).tensor_split(cfg.minibatches):
                mean,values,mask=self.model(b['obs'][indices],b['hand_type'][indices])
                std=self.model.log_std.exp().expand_as(mean)
                dist=torch.distributions.Normal(mean,std)
                lp=(dist.log_prob(b['actions'][indices])*mask).sum(-1)
                entropy=(dist.entropy()*mask).sum(-1).mean()
                with torch.no_grad():
                    kl=((torch.log(std/b['std'][indices]+1e-5)+(b['std'][indices].square()+(b['mean'][indices]-mean).square())/(2*std.square())-.5)*mask).sum(-1).mean()
                    lr=self.optimizer.param_groups[0]['lr']
                    if kl>cfg.desired_kl*2:lr=max(1e-5,lr/1.2)
                    elif 0<kl<cfg.desired_kl/2:lr=min(1e-2,lr*1.2)
                    for group in self.optimizer.param_groups:group['lr']=lr
                ratio=(lp-b['log_prob'][indices]).exp()
                advantage=b['advantages'][indices]
                surrogate=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-cfg.clip,1+cfg.clip)).mean()
                clipped=b['values'][indices]+(values-b['values'][indices]).clamp(-cfg.clip,cfg.clip)
                value_loss=torch.maximum((values-b['returns'][indices]).square(),(clipped-b['returns'][indices]).square()).mean()
                loss=surrogate+cfg.value_coef*value_loss-cfg.entropy_coef*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg.max_grad_norm,error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():self.model.log_std.clamp_(min=__import__('math').log(.2),max=2.)
                stats.append([float(loss),float(surrogate),float(value_loss),float(entropy),float(kl),float(norm)])
        after=torch.cat([p.detach().flatten() for p in self.model.actor.parameters()])
        delta=float((after-before).norm())
        if not delta>0:raise RuntimeError('Actor weights did not change')
        means=torch.tensor(stats).mean(0).tolist()
        result=dict(zip(['loss','policy_loss','value_loss','entropy','kl','gradient_norm'],means))
        result.update(actor_parameter_delta=delta,learning_rate=self.optimizer.param_groups[0]['lr'])
        return result
