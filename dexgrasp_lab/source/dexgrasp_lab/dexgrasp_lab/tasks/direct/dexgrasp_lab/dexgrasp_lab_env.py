"""A heterogeneous, shared-policy DexGrasp-Zero DirectRLEnv.

Each environment owns one training hand and one object. Views are homogeneous per
hand; global environment indices are explicitly mapped to each local asset view.
"""
import xml.etree.ElementTree as ET
import re
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.utils.math import (quat_apply,quat_apply_inverse,quat_mul,quat_conjugate,
    quat_from_angle_axis,quat_from_matrix,matrix_from_euler,matrix_from_quat,axis_angle_from_quat,normalize)
from ....hands import hand_spec,usd_name
from ....dataset import split_objects,geometry
from ....assets import hand_usd,object_usd
from .dexgrasp_lab_env_cfg import DexgraspLabEnvCfg


def intrinsic_xyz(q):
    m=matrix_from_quat(q)
    return torch.stack((torch.atan2(-m[:,1,2],m[:,2,2]),torch.asin(m[:,0,2].clamp(-1.,1.)),torch.atan2(-m[:,0,1],m[:,0,0])),dim=-1)


def squash_velocity_penalty(x):
    return torch.where(x<-.75,(x+.75)/4.-.75,x).clamp(min=-1.)


class DexgraspLabEnv(DirectRLEnv):
    cfg:DexgraspLabEnvCfg
    def __init__(self,cfg,render_mode=None,**kwargs):
        if cfg.object_split!='train':raise ValueError('This training task only admits train objects; held-out evaluation is deferred')
        if cfg.scene.num_envs%len(cfg.hands):raise ValueError('num_envs must be divisible by the number of hands')
        self.specs={h:hand_spec(h) for h in cfg.hands}
        cfg.action_space=max(s['act_dim'] for s in self.specs.values())
        cfg.observation_space=max(s['obs_dim'] for s in self.specs.values())
        super().__init__(cfg,render_mode,**kwargs)
        env_number=lambda path:int(re.search(r'/env_(\d+)/',path).group(1))
        object_envs=[env_number(p) for p in self.object.root_physx_view.prim_paths]
        assert sorted(object_envs)==list(range(self.num_envs))
        self.object_order=torch.tensor([object_envs.index(i) for i in range(self.num_envs)],device=self.device)
        self.hand_types=torch.empty(self.num_envs,dtype=torch.long,device=self.device)
        self.action_mask=torch.zeros((self.num_envs,self.cfg.action_space),device=self.device)
        self.base_pos=torch.zeros((self.num_envs,3),device=self.device)
        self.base_quat=torch.zeros((self.num_envs,4),device=self.device);self.base_quat[:,0]=1.
        self.initial_relative_quat=self.base_quat.clone()
        self.target_pos=self.base_pos.clone();self.target_quat=self.base_quat.clone()
        self.targets=torch.zeros((self.num_envs,self.cfg.action_space),device=self.device)
        self.obs=torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device)
        self.rewards=torch.zeros(self.num_envs,device=self.device)
        self.bad=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.numerical_failures=0;self.completed_episodes=0
        self.workspace_failures=0
        self.failed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.initial_object_z=torch.zeros(self.num_envs,device=self.device)
        self.points=torch.tensor(__import__('numpy').stack([d['points'] for d in self.object_data]),device=self.device)
        self.object_min_z=torch.tensor([d['min_z'] for d in self.object_data],device=self.device)
        self.object_radius=torch.tensor([d['radius'] for d in self.object_data],device=self.device)
        self.centers={};self.orders={};self.node_ids={};self.root_ids={};self.masses={}
        self.dist_weights={};self.contact_weights={};self.impulse_limits={}
        for name,s in self.specs.items():
            r=self.robots[name]
            ids=torch.tensor([env_number(p) for p in r.root_physx_view.prim_paths],device=self.device)
            assert set(ids.tolist())==set(self.global_ids[name].tolist())
            self.global_ids[name]=ids
            for sensor in self.sensors[name]:
                assert [env_number(p) for p in sensor.body_physx_view.prim_paths]==ids.tolist()
            assert r.num_instances==len(ids),(name,r.num_instances,len(ids))
            assert r.num_joints==s['act_dim']-6,(name,r.joint_names)
            self.orders[name]=r.find_joints([usd_name(n) for n in s['order'][6:]],preserve_order=True)[0]
            self.node_ids[name]=r.find_bodies([usd_name(n) for n in s['links']],preserve_order=True)[0]
            self.root_ids[name]=r.body_names.index(usd_name(s['root_link']))
            self.centers[name]=torch.tensor(s['center'],device=self.device,dtype=torch.float32)
            self.hand_types[ids]=s['hand_type'];self.action_mask[ids,:s['act_dim']]=1.
            self.masses[name]=r.root_physx_view.get_masses().sum(-1).to(self.device)
            weights=torch.ones(len(s['links']),device=self.device);weights[s['data']['fingertips_id']]=4.
            self.dist_weights[name]=weights/weights.sum()*s['data']['finger_weights_scale']
            cw=torch.ones(len(s['contact_ids']),device=self.device);limits=torch.full_like(cw,.1)
            for ci,ni in enumerate(s['contact_ids']):
                if s['data']['node_semantic_mapping'][ni].value==3:cw[ci]*=3.
                if s['data']['finger_id_mapping'][ni].value==1:cw[ci]*=2.;limits[ci]=.2
            self.contact_weights[name]=cw/cw.sum()*len(cw);self.impulse_limits[name]=limits
        print('DEXGRASP_SCENE_READY',{'hands':list(self.specs),'num_envs':self.num_envs,'objects':len(set(self.object_names))},flush=True)

    def _setup_scene(self):
        from pxr import Usd,UsdPhysics,UsdShade
        # Heterogeneous environments must not inherit subsequent env_0 edits.
        self.scene.clone_environments(copy_from_source=True)
        material_cfg=sim_utils.RigidBodyMaterialCfg(static_friction=3.,dynamic_friction=3.,restitution=0.)
        material_cfg.func('/World/GraspMaterial',material_cfg)
        material=UsdShade.Material(self.sim.stage.GetPrimAtPath('/World/GraspMaterial'))
        self.robots={};self.sensors={};self.global_ids={}
        names=split_objects(self.cfg.object_split)
        per_hand=self.num_envs//len(self.specs)
        # Each hand sees the same object sequence; no validation object can enter this task.
        self.object_names=[names[i%len(names)] for _ in self.specs for i in range(per_hand)]
        unique={n:geometry(n,self.cfg.seed,self.cfg.num_surface_points) for n in set(self.object_names)}
        self.object_data=[unique[n] for n in self.object_names]
        self.object_usds={n:object_usd(d) for n,d in unique.items()}
        for group,(name,s) in enumerate(self.specs.items()):
            indices=torch.arange(group*per_hand,(group+1)*per_hand,device=self.device)
            self.global_ids[name]=indices
            usd=hand_usd(s)
            urdf=ET.parse(s['path']);joint_xml={j.get('name'):j for j in urdf.findall('joint')}
            damping={usd_name(n):.2+float(joint_xml[n].find('dynamics').get('damping','0')) if joint_xml[n].find('dynamics') is not None else .2 for n in s['order'][6:]}
            effort={usd_name(n):float(joint_xml[n].find('limit').get('effort','100')) for n in s['order'][6:]}
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd),activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=2.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=16,solver_velocity_iteration_count=4))
            for i in indices.tolist():
                env_path=self.scene.env_prim_paths[i]
                spawn.func(f'{env_path}/{name}',spawn,translation=(0.,0.,1.))
            cfg=ArticulationCfg(prim_path=f'/World/envs/env_.*/{name}',spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,1.),
                    joint_pos={usd_name(n):float(q) for n,q in zip(s['order'][6:],s['q0'][6:])}),
                actuators={'fingers':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=100.,damping=damping,effort_limit_sim=effort)})
            robot=Articulation(cfg);self.robots[name]=robot;self.scene.articulations[name]=robot
            self.sensors[name]=[]
            for ci,body in enumerate(s['contact_links']):
                sensor=ContactSensor(ContactSensorCfg(prim_path=f'/World/envs/env_.*/{name}/{usd_name(body)}',
                    update_period=0.,history_length=0,filter_prim_paths_expr=[f'/World/envs/env_.*/Object_{name}'],max_contact_data_count_per_prim=64))
                self.sensors[name].append(sensor);self.scene.sensors[f'{name}_contact_{ci}']=sensor
        for i,d in enumerate(self.object_data):
            env_path=self.scene.env_prim_paths[i]
            hand_name=list(self.specs)[i//per_hand]
            cfg=sim_utils.UsdFileCfg(usd_path=str(self.object_usds[d['name']]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=self.cfg.stage=='fixed',
                    max_depenetration_velocity=2.,solver_position_iteration_count=16,solver_velocity_iteration_count=4))
            cfg.func(f'{env_path}/Object_{hand_name}',cfg,translation=(0.,0.,.7))
            table=sim_utils.CuboidCfg(size=(1.,1.,.5),collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.3,.3,.3)),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.,dynamic_friction=3.,restitution=0.))
            table.func(f'{env_path}/Table',table,translation=(0.,0.,.25))
            # Original mask excludes hand/table collisions.
            root=self.sim.stage.GetPrimAtPath(f'{env_path}/{hand_name}')
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(f'{env_path}/Table')
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
            for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(f'{env_path}/Object_{hand_name}')):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
        self.object=RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/Object_.*',spawn=None))
        self.scene.rigid_objects['object']=self.object
        self.scene.filter_collisions(global_prim_paths=[])
        light=sim_utils.DomeLightCfg(intensity=2000.);light.func('/World/Light',light)

    def _object(self,attribute):
        return getattr(self.object.data,attribute)[self.object_order]

    def _reset_idx(self,env_ids):
        # Asset views contain subsets of the global environments: map reset indices explicitly.
        if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
        self.episode_length_buf[env_ids]=0
        self.completed_episodes+=len(env_ids)
        self.bad[env_ids]=False
        obj_state=torch.zeros((len(env_ids),13),device=self.device)
        obj_state[:,:3]=self.scene.env_origins[env_ids]
        obj_state[:,2]+=.502-self.object_min_z[env_ids]
        yaw=torch.rand(len(env_ids),device=self.device)*2*torch.pi
        axis=torch.zeros((len(env_ids),3),device=self.device);axis[:,2]=1.
        obj_state[:,3:7]=quat_from_angle_axis(yaw,axis)
        self.object.write_root_state_to_sim(obj_state,env_ids=self.object_order[env_ids]);self.object.reset(self.object_order[env_ids])
        self.initial_object_z[env_ids]=obj_state[:,2]
        for name,s in self.specs.items():
            gids=self.global_ids[name];local=torch.isin(gids,env_ids).nonzero().flatten();ids=gids[local]
            if not len(ids):continue
            r=self.robots[name];n=len(ids)
            center=self.centers[name].expand(n,-1);local_axis=normalize(center)
            direction=normalize(torch.randn((n,3),device=self.device));direction[:,2]=-direction[:,2].abs()-.2;direction=normalize(direction)
            align=normalize(torch.cat((1+(local_axis*direction).sum(-1,keepdim=True),torch.linalg.cross(local_axis,direction)),dim=-1))
            spin=quat_from_angle_axis(torch.rand(n,device=self.device)*2*torch.pi,direction)
            orientation=quat_mul(spin,align)
            # Place the grasp center outside the object's bounding sphere; the
            # old constant center gap could initialize fingers deep inside larger objects.
            distance=center.norm(dim=-1)+self.object_radius[ids]+self.cfg.pregrasp_clearance
            pos=self._object('root_pos_w')[ids]-direction*distance[:,None]
            state=torch.zeros((n,13),device=self.device);state[:,:3]=pos;state[:,3:7]=orientation
            r.write_root_state_to_sim(state,env_ids=local)
            ordered=torch.tensor(s['q0'][6:],device=self.device,dtype=torch.float32).expand(n,-1)
            q=r.data.default_joint_pos[local].clone();q[:,self.orders[name]]=ordered
            r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=local)
            r.set_joint_position_target(q,env_ids=local);r.reset(local)
            for sensor in self.sensors[name]:sensor.reset(local)
            self.base_pos[ids]=pos;self.base_quat[ids]=orientation
            self.initial_relative_quat[ids]=quat_mul(quat_conjugate(self._object('root_quat_w')[ids]),orientation)
            self.target_pos[ids]=pos;self.target_quat[ids]=orientation
            self.targets[ids]=0.;self.targets[ids,6:s['act_dim']]=ordered
        self._state_fresh=False

    def _pre_physics_step(self,actions):
        if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite policy actions')
        self.actions=actions.detach()*self.action_mask
        for name,s in self.specs.items():
            ids=self.global_ids[name];r=self.robots[name];a=self.actions[ids,:s['act_dim']]
            q=r.data.joint_pos[:,self.orders[name]]
            finger_target=q+a[:,6:]*self.cfg.finger_action_scale
            limits=r.data.joint_pos_limits[:,self.orders[name]]
            finger_target=torch.max(torch.min(finger_target,limits[:,:,1]),limits[:,:,0])
            self.targets[ids,6:s['act_dim']]=finger_target
            r.set_joint_position_target(finger_target,joint_ids=self.orders[name])
            wrist=r.data.root_link_pos_w;orientation=r.data.root_link_quat_w
            bias=torch.zeros_like(wrist)
            goal=orientation
            if self.cfg.root_guided:
                bias=self._object('root_pos_w')[ids]-wrist-quat_apply(orientation,self.centers[name].expand(len(ids),-1))
                goal=quat_mul(self._object('root_quat_w')[ids],self.initial_relative_quat[ids])
            self.target_pos[ids]=wrist+bias+quat_apply(self.base_quat[ids],a[:,:3]*self.cfg.wrist_position_scale)
            delta=quat_from_matrix(matrix_from_euler(a[:,3:6]*self.cfg.wrist_rotation_scale,'XYZ'))
            self.target_quat[ids]=quat_mul(goal,delta)
            self.targets[ids,:3]=quat_apply_inverse(self.base_quat[ids],self.target_pos[ids]-self.base_pos[ids])
            self.targets[ids,3:6]=intrinsic_xyz(quat_mul(quat_conjugate(self.base_quat[ids]),self.target_quat[ids]))
        self._state_fresh=False

    def _apply_action(self):
        for name,r in self.robots.items():
            ids=self.global_ids[name]
            force=self.cfg.wrist_position_gain*(self.target_pos[ids]-r.data.root_link_pos_w)-self.cfg.wrist_velocity_gain*r.data.root_link_lin_vel_w
            force[:,2]+=9.81*self.masses[name]
            error=quat_mul(self.target_quat[ids],quat_conjugate(r.data.root_link_quat_w))
            torque=self.cfg.wrist_rotation_gain*axis_angle_from_quat(error)-self.cfg.wrist_angular_velocity_gain*r.data.root_link_ang_vel_w
            r.set_external_force_and_torque(force.clamp(-100.,100.)[:,None],torque.clamp(-10.,10.)[:,None],body_ids=[self.root_ids[name]],is_global=True)

    def _compute_state(self):
        self.obs.zero_();self.rewards.zero_();self.bad.zero_()
        obj_pos=self._object('root_pos_w');obj_quat=self._object('root_quat_w')
        for name,s in self.specs.items():
            ids=self.global_ids[name];r=self.robots[name];n=len(ids)
            wp=r.data.root_link_pos_w;wq=r.data.root_link_quat_w
            center=quat_apply_inverse(wq,obj_pos[ids]-wp)-self.centers[name]
            qr=intrinsic_xyz(quat_mul(quat_conjugate(self.base_quat[ids]),wq))
            q=torch.cat((quat_apply_inverse(self.base_quat[ids],wp-self.base_pos[ids]),qr,r.data.joint_pos[:,self.orders[name]]),dim=-1)
            goal=quat_mul(quat_conjugate(self.base_quat[ids]),quat_mul(obj_quat[ids],self.initial_relative_quat[ids]))
            orientation_error=qr-intrinsic_xyz(goal)
            wlin=quat_apply_inverse(wq,r.data.root_link_lin_vel_w);wang=quat_apply_inverse(wq,r.data.root_link_ang_vel_w)
            olin=quat_apply_inverse(wq,self._object('root_lin_vel_w')[ids]);oang=quat_apply_inverse(wq,self._object('root_ang_vel_w')[ids])
            impulses=torch.stack([sensor.data.force_matrix_w.reshape(n,-1,3).sum(1).norm(dim=-1)*self.physics_dt for sensor in self.sensors[name]],dim=-1)
            contacts=(impulses>1e-10).float();nonaff=torch.zeros_like(contacts)
            nodes=r.data.body_link_pos_w[:,self.node_ids[name]]
            oq=obj_quat[ids,None,:].expand(-1,len(s['links']),-1)
            nodes_local=quat_apply_inverse(oq,nodes-obj_pos[ids,None,:])
            distances=torch.cdist(nodes_local,self.points[ids])
            nearest=distances.argmin(-1)
            vectors=torch.gather(self.points[ids],1,nearest[:,:,None].expand(-1,-1,3))-nodes_local
            relative=quat_mul(quat_conjugate(wq),obj_quat[ids])[:,None,:].expand(-1,len(s['links']),-1)
            vectors=quat_apply(relative,vectors)
            obs=torch.cat((center,orientation_error,q[:,6:],self.targets[ids,:s['act_dim']]-q,wlin,wang,
                contacts,impulses,olin-wlin,oang-wang,nonaff,nonaff,vectors.flatten(1),torch.zeros_like(vectors).flatten(1)),dim=-1)
            assert obs.shape[1]==s['obs_dim']
            self.obs[ids,:s['obs_dim']]=obs
            cw=self.contact_weights[name]
            reward=(contacts*cw).sum(-1)/len(cw)+.5*(impulses.clamp(min=0.).minimum(self.impulse_limits[name])*cw).sum(-1)
            reward-=.3*(vectors.norm(dim=-1)*self.dist_weights[name]).sum(-1)
            reward-=10.*center.square().sum(-1)+.01*orientation_error.square().sum(-1)
            for coeff,velocity in [(.001,wlin),(.0001,wang),(1.5,self._object('root_lin_vel_w')[ids]),(.01,self._object('root_ang_vel_w')[ids])]:
                penalty=-coeff*velocity.square().sum(-1)
                reward+=squash_velocity_penalty(penalty) if self.cfg.stage=='floating' else penalty
            self.rewards[ids]=reward
            finite=torch.isfinite(obs).all(-1)&torch.isfinite(reward)
            self.bad[ids]=~finite
        self._state_fresh=True

    def _get_observations(self):
        if not self._state_fresh:self._compute_state()
        return {'policy':torch.nan_to_num(self.obs).clone()}

    def _get_dones(self):
        self._compute_state()
        # Invalid dynamics terminate immediately; time limits bootstrap the actual final state.
        out=((self._object('root_pos_w')-self.scene.env_origins).norm(dim=-1)>3.) | (self.obs[:,:3].norm(dim=-1)>self.cfg.max_grasp_center_distance)
        terminated=self.bad|out
        self.failed=terminated.clone()
        self.workspace_failures+=int((out & ~self.bad).sum())
        truncated=self.episode_length_buf>=self.max_episode_length
        self.numerical_failures+=int(self.bad.sum())
        self.extras['terminal_observation']=torch.nan_to_num(self.obs).clone()
        self.extras['terminal_observation'][terminated]=0.
        self.extras['numerical_failures']=self.numerical_failures
        self.extras['workspace_failures']=self.workspace_failures
        return terminated,truncated

    def _get_rewards(self):
        reward=torch.where(self.failed,torch.full_like(self.rewards,self.cfg.failure_reward),self.rewards)
        self.extras['log']={f'reward/{h}':reward[ids].mean() for h,ids in self.global_ids.items()}
        return reward.clone()
