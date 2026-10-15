"""PhysX implementation of the LEAP_TEST task in Environment.hpp.

Import this module only after SimulationApp has started.
"""
import numpy as np
import torch
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from .task import (Kinematics, hand_spec, usd_name, DT, CONTROL_DT,
                   mv, mat, quat, transpose)
from .assets import convert


class CrossHandYCB:
    def __init__(self, objects, device='cuda:0', render=False, root_guided=True, hand='leap'):
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf
        self.objects, self.n = objects, len(objects)
        self.device, self.render, self.root_guided = device, render, root_guided
        self.spec = hand_spec(hand)
        self.kin = Kinematics(self.spec)
        self.a = self.spec['act_dim']
        self.c = len(self.spec['contacts'])
        self.world = World(physics_dt=DT, rendering_dt=CONTROL_DT, backend='numpy', device='cpu')
        self.world.get_physics_context().set_solver_type('TGS')
        self.world.get_physics_context().set_broadphase_type('MBP')
        stage = self.world.stage
        material = PhysicsMaterial('/World/material', static_friction=3., dynamic_friction=3., restitution=0.)
        self.world.scene.add(FixedCuboid('/World/ground',name='ground',position=np.array([0.,0.,-.05]),scale=np.array([200.,200.,.1]),physics_material=material))
        physmat = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath('/World/material'))
        physmat.CreateFrictionCombineModeAttr('average')
        hand_usd = convert(self.spec['path'], hand=True)
        self.offsets = np.array([[4.*(i % 10), 4.*(i // 10), 0.] for i in range(self.n)])
        roots, obj_roots, bodies, contact_channels, filters = [], [], [], [], []
        for i, data in enumerate(objects):
            prefix = f'/World/env_{i}'
            env_prim = UsdGeom.Xform.Define(stage, prefix)
            env_prim.AddTranslateOp().Set(Gf.Vec3d(*self.offsets[i]))
            group = UsdPhysics.CollisionGroup.Define(stage,f'/World/collision_group_{i}')
            group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(prefix)
            group.CreateFilteredGroupsRel().SetTargets([f'/World/collision_group_{j}' for j in range(self.n) if j!=i])
            hand = add_reference_to_stage(str(hand_usd), prefix+'/hand')
            obj = add_reference_to_stage(str(convert(data.path/f'{data.name}.urdf')), prefix+'/object')
            table_path = prefix+'/table'
            self.world.scene.add(FixedCuboid(table_path, name=f'table_{i}',
                position=self.offsets[i]+[1.25,0,.25], scale=np.array([2.,1.,.5]),
                color=np.array([.3,.25,.2]), physics_material=material))
            for root in [hand, obj]:
                art_roots = []
                for prim in Usd.PrimRange(root):
                    if prim.IsInstance(): prim.SetInstanceable(False)
                for prim in Usd.PrimRange(root):
                    for axis in ['linear','angular']:
                        if prim.HasAPI(UsdPhysics.DriveAPI,axis):
                            UsdPhysics.DriveAPI.Get(prim,axis).CreateTypeAttr('force')
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        art_roots.append(str(prim.GetPath()))
                        api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                        api.CreateSolverPositionIterationCountAttr(16)
                        api.CreateSolverVelocityIterationCountAttr(4)
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose='physics')
                        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        collision.CreateContactOffsetAttr(.001)
                        collision.CreateRestOffsetAttr(0.)
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose='physics')
                        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        rb.CreateLinearDampingAttr(0.)
                        rb.CreateAngularDampingAttr(0.)
                        rb.CreateMaxDepenetrationVelocityAttr(2.)
                        if root == hand:
                            bodies.append(str(prim.GetPath()))
                            contact_channels.append(next((k for k,n in enumerate(self.spec['contacts']) if usd_name(n)==prim.GetName()),0))
                            filters.append([prefix+'/object/top',prefix+'/object/bottom'])
                            # RaiSim masks exclude hand/table collisions but retain hand/object and self contact.
                            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(table_path)
                if len(art_roots) != 1:
                    raise RuntimeError(f'Expected one articulation: {root.GetPath()}: {art_roots}')
                (roots if root == hand else obj_roots).append(art_roots[0])
        self.sensor_channels = np.array(contact_channels)
        self.sensors_per_env = len(bodies)//self.n
        self.hand = self.world.scene.add(Articulation(roots, name='hands', reset_xform_properties=False))
        self.obj = self.world.scene.add(Articulation(obj_roots, name='objects', reset_xform_properties=False))
        self.top = self.world.scene.add(RigidPrim([f'/World/env_{i}/object/top' for i in range(self.n)],
                                                 name='tops', reset_xform_properties=False))
        self.palm = self.world.scene.add(RigidPrim([f'/World/env_{i}/hand/{usd_name(self.kin.joints[self.spec["frames"][0]].find("child").get("link"))}' for i in range(self.n)],
                                                  name='palms', reset_xform_properties=False))
        self.contact = self.world.scene.add(RigidPrim(bodies, name='contacts', reset_xform_properties=False,
            track_contact_forces=True, contact_filter_prim_paths_expr=filters,
            max_contact_count=max(16384, self.n*2048)))
        self.world.reset()
        # Read physics tensors directly. Avoid writing thousands of transforms
        # back to USD on every 2.5 ms substep (about 30x overhead at 78 envs).
        self.world.get_physics_context().enable_fabric(True)
        if self.hand.num_dof != self.a or self.obj.num_dof != 1:
            raise RuntimeError(f'Unexpected DOF count: hand={self.hand.num_dof}, object={self.obj.num_dof}')
        self.order = np.array([self.hand.dof_names.index(usd_name(j)) for j in self.spec['joints']])
        self.inverse_order = np.argsort(self.order)
        g = self.spec['gains']
        gains = np.tile([g['wrist_Pgain']]*3+[g['rot_Pgain']]*(self.a-3),(self.n,1)).astype(np.float32)
        damping = np.tile([g['wrist_Dgain']]*3+[g['rot_Dgain']]*(self.a-3),(self.n,1)).astype(np.float32)
        passive_damping = np.array([float(self.kin.joints[j].find('dynamics').get('damping','0')) if self.kin.joints[j].find('dynamics') is not None else 0. for j in self.spec['joints']])
        damping += passive_damping
        self.hand.set_gains(gains[:,self.inverse_order], damping[:,self.inverse_order])
        object_damping = np.array([[float(ET.parse(o.path/f'{o.name}.urdf').find('joint/dynamics').get('damping','0'))] for o in objects],np.float32)
        self.obj.set_gains(np.zeros_like(object_damping),object_damping)
        # This is passive damping, not an actuated position controller.
        self.obj.set_max_efforts(np.full_like(object_damping,1e6))
        effort = np.array([float(self.kin.joints[j].find('limit').get('effort','1e6')) for j in self.spec['joints']],np.float32)
        self.hand.set_max_efforts(np.tile(effort[self.inverse_order],(self.n,1)))
        self.points = torch.tensor(np.stack([o.points for o in objects]), dtype=torch.float32, device=device)
        self.has_nonaff = np.array([o.has_nonaff for o in objects])
        self.scale = np.array([.005]*6+[.015]*(self.a-6))
        self.target = np.zeros((self.n,self.a))
        self.last_q = np.zeros_like(self.target)
        self.obs = np.zeros((self.n,self.spec['obs_dim']),np.float32)
        print(f'SCENE_READY environments={self.n} dofs={self.hand.dof_names}', flush=True)

    def reset(self, update):
        self.failed = np.zeros(self.n,dtype=bool)
        self.failure_reason = [None]*self.n
        self.terminated_step = [None]*self.n
        self.current_step = -1
        initial = [o.initial(update,self.spec['name']) for o in self.objects]
        self.base_pos = np.stack([r[0] for r in initial])+self.offsets
        self.base_rot = np.stack([r[1] for r in initial])
        self.obj_initial = np.stack([r[2] for r in initial])+self.offsets
        self.aff_center = np.stack([r[3] for r in initial])
        self.hand.set_world_poses(self.base_pos.astype(np.float32), quat(self.base_rot).astype(np.float32))
        self.obj.set_world_poses(self.obj_initial.astype(np.float32), np.tile([1.,0,0,0],(self.n,1)).astype(np.float32))
        self.obj.set_velocities(np.zeros((self.n,6),np.float32))
        self.obj.set_joint_positions(np.zeros((self.n,1),np.float32))
        self.obj.set_joint_velocities(np.zeros((self.n,1),np.float32))
        q = np.zeros((self.n,self.a),np.float32)
        name = self.spec['name']
        if name in ['leap','allegro','shadow']: q[:,-4]=1.7
        elif name in ['schunk','inspire','ability','brainco']:
            q[:,6]={'schunk':1.,'inspire':1.3,'ability':-1.9,'brainco':.9}[name]
        self.hand.set_joint_positions(q[:,self.inverse_order])
        self.hand.set_joint_velocities(np.zeros_like(q))
        self.hand.set_joint_position_targets(q[:,self.inverse_order])
        self.hand.set_joint_velocity_targets(np.zeros_like(q))
        self.hand.set_joint_efforts(np.zeros_like(q))
        self.target = np.zeros_like(q)  # Original first reset uses zero pTarget_clipped.
        self.last_q = q.copy()
        pos, rot = self.kin.forward(q,self.base_pos,self.base_rot)
        fingers = pos[:,self.spec['finger_center_ids']].mean(axis=1)
        thumb = pos[:,self.spec['thumb_id']]
        if name == 'inspire': fingers = (fingers+thumb)*.5
        center = (fingers+thumb)*.5
        across = fingers-center
        across /= np.linalg.norm(across,axis=1,keepdims=True)
        self.hand_center = mv(transpose(rot[:,0]), center-pos[:,0])
        self.grasp_axis = center-pos[:,0]
        self.grasp_axis /= np.linalg.norm(self.grasp_axis,axis=1,keepdims=True)
        self.across_axis = across
        self.across_wrist = mv(transpose(rot[:,0]), across)
        self.wrist_in_obj_initial = rot[:,0].copy()
        self.lift = False
        # No physics settle steps: source starts policy inference immediately after reset.
        return self.observe(reset=True)

    def fail(self, mask, reason):
        """End only numerically invalid trials; park their bodies out of contact."""
        ids=np.flatnonzero(mask & ~self.failed)
        if not len(ids): return
        self.failed[ids]=True
        for i in ids:
            self.failure_reason[i]=reason
            self.terminated_step[i]=self.current_step
            print(f'NUMERICAL_FAILURE object={self.objects[i].name} step={self.current_step} reason={reason}',flush=True)
        q=np.zeros((len(ids),self.a),np.float32)
        parking=self.offsets[ids]+[0.,0.,-100.]
        identity=np.tile([1.,0.,0.,0.],(len(ids),1)).astype(np.float32)
        self.hand.set_world_poses(parking.astype(np.float32),identity,indices=ids)
        self.hand.set_joint_positions(q,indices=ids)
        self.hand.set_joint_velocities(q,indices=ids)
        self.hand.set_joint_position_targets(q,indices=ids)
        self.obj.set_world_poses((parking+[0.,0.,-10.]).astype(np.float32),identity,indices=ids)
        self.obj.set_velocities(np.zeros((len(ids),6),np.float32),indices=ids)
        self.obj.set_joint_positions(np.zeros((len(ids),1),np.float32),indices=ids)
        self.obj.set_joint_velocities(np.zeros((len(ids),1),np.float32),indices=ids)

    def observe(self, reset=False):
        q = self.hand.get_joint_positions()[:,self.order].astype(np.float64)
        self.fail(~np.isfinite(q).all(axis=1),'nonfinite_joint_positions')
        q[self.failed]=0.
        pos, rot = self.kin.forward(q,self.base_pos,self.base_rot)
        wrist_pos, wrist_rot = pos[:,0],rot[:,0]
        iw = transpose(wrist_rot)
        if reset:
            obj_pos, obj_rot = self.obj_initial, np.tile(np.eye(3),(self.n,1,1))
            wrist_vel = obj_vel = np.zeros((self.n,6))
            impulses = np.zeros((self.n,self.c,2))
        else:
            obj_pos, obj_quat = self.top.get_world_poses()
            bad=(~np.isfinite(obj_pos).all(axis=1) | ~np.isfinite(obj_quat).all(axis=1)
                 | (np.linalg.norm(obj_quat,axis=1)<1e-8))
            self.fail(bad,'invalid_object_pose')
            obj_pos[self.failed]=self.obj_initial[self.failed]
            obj_quat[self.failed]=[1.,0.,0.,0.]
            obj_rot = mat(obj_quat)
            wrist_vel, obj_vel = self.palm.get_velocities(), self.top.get_velocities()
            forces = self.contact.get_contact_force_matrix(dt=1.)  # impulses in N s, NOT N
            if forces.shape != (self.n*self.sensors_per_env,2,3):
                raise RuntimeError(f'Unexpected contact matrix shape: {forces.shape}')
            sensor_impulses = np.linalg.norm(forces,axis=-1)
            impulses = np.zeros((self.n,self.c,2))
            np.add.at(impulses,(np.repeat(np.arange(self.n),self.sensors_per_env),self.sensor_channels),sensor_impulses)
        contacts = (impulses>1e-10).astype(float)
        # Source suppresses non-affordance contacts on a body also contacting affordance.
        impulses[:,:,1] -= impulses[:,:,0]*contacts[:,:,1]
        contacts[:,:,1] -= contacts[:,:,0]*contacts[:,:,1]
        contact_mask = np.array([o.has_nonaff_contact for o in self.objects])
        impulses[~contact_mask,:,1] = 0
        contacts[~contact_mask,:,1] = 0
        center_error = mv(iw, mv(obj_rot,self.aff_center)+obj_pos-wrist_pos)-self.hand_center
        target_rot = transpose(self.base_rot) @ obj_rot @ self.wrist_in_obj_initial
        target_euler = Rotation.from_matrix(target_rot).as_euler('XYZ')
        wlin, wang = mv(iw,wrist_vel[:,:3]),mv(iw,wrist_vel[:,3:])
        raw = np.concatenate([center_error, q[:,3:6]-target_euler, q[:,6:], self.target-q,
                              wlin, wang, contacts[:,:,0],impulses[:,:,0],
                              mv(iw,obj_vel[:,:3])-wlin,mv(iw,obj_vel[:,3:])-wang,
                              contacts[:,:,1],impulses[:,:,1]],axis=1)
        joints_obj = np.einsum('nij,nkj->nki',transpose(obj_rot),pos-obj_pos[:,None])
        joints = torch.tensor(joints_obj,dtype=torch.float32,device=self.device)
        vectors = []
        for part in range(2):
            distance = torch.cdist(joints,self.points[:,part])
            ids = distance.argmin(dim=-1)
            nearest = torch.gather(self.points[:,part],1,ids[:,:,None].expand(-1,-1,3))
            vec = (nearest-joints).cpu().numpy()
            vec = np.einsum('nij,nkj->nki',iw@obj_rot,vec)
            if part == 1: vec[~self.has_nonaff] = 0
            vectors.append(vec.reshape(self.n,-1))
        self.obs = np.concatenate([raw,*vectors],axis=1).astype(np.float32)
        assert self.obs.shape == (self.n,self.spec['obs_dim'])
        self.fail(~np.isfinite(self.obs).all(axis=1),'nonfinite_observation')
        self.obs[self.failed]=0.
        self.last_q = q
        self.obj_pos, self.obj_rot = obj_pos,obj_rot
        self.wrist_pos, self.wrist_rot = wrist_pos,wrist_rot
        self.contacts = contacts
        if not reset:
            native_pos, native_quat = self.palm.get_world_poses()
            bad=(~np.isfinite(native_pos).all(axis=1) | ~np.isfinite(native_quat).all(axis=1)
                 | (np.linalg.norm(native_quat,axis=1)<1e-8))
            self.fail(bad,'invalid_hand_pose')
            active=~self.failed
            if not active.any(): return self.obs
            error = np.max(np.linalg.norm(native_pos[active]-wrist_pos[active],axis=1))
            rot_error = np.max(np.abs(mat(native_quat[active])-wrist_rot[active]))
            if error > 1e-3 or rot_error > 1e-3:
                raise RuntimeError(f'FK/PhysX mismatch: position={error} rotation={rot_error}')
        return self.obs

    def step(self, actions, step):
        self.current_step=step
        self.fail(~np.isfinite(actions).all(axis=1),'nonfinite_policy_action')
        actions=actions.copy()
        actions[self.failed]=0.
        if step == 130: self.lift = True
        mean = self.last_q.copy()
        bias = np.zeros((self.n,3))
        if self.root_guided:
            bias = mv(self.obj_rot,self.aff_center)+self.obj_pos-self.wrist_pos-mv(self.wrist_rot,self.hand_center)
            relative = transpose(self.wrist_rot)@self.obj_rot@self.wrist_in_obj_initial
            mean[:,3:6] = Rotation.from_matrix(relative).as_euler('XYZ')
        bias[:,2] += self.kin.mass*9.81*.01 + ((.5 if self.spec['name']=='barrett' and self.root_guided else .2) if self.lift else 0.)
        mean[:,:3] += mv(transpose(self.base_rot),bias)
        self.target = np.clip(mean+actions*self.scale,self.kin.lower,self.kin.upper)
        self.target[self.failed]=0.
        self.hand.set_joint_position_targets(self.target[:,self.inverse_order].astype(np.float32))
        for _ in range(round(CONTROL_DT/DT)):
            self.world.step(render=False)
        if self.render: self.world.render()
        return self.observe()

    def results(self, update, valid):
        center = np.linalg.norm(self.obs[:,:3],axis=1)
        height = self.obj_pos[:,2]-self.obj_initial[:,2]
        grasp = mv(self.wrist_rot,self.hand_center)
        grasp /= np.linalg.norm(grasp,axis=1,keepdims=True)
        direction = np.linalg.norm(mv(transpose(self.obj_rot),grasp)-self.grasp_axis,axis=1)
        across = mv(transpose(self.obj_rot),mv(self.wrist_rot,self.across_wrist))
        angle = np.arccos(np.clip(np.sum(across*self.across_axis,axis=1),-1,1))
        ratio = self.contacts[:,:,0].sum(1)/(self.contacts.sum((1,2))+1e-7)
        rows=[]
        for i,o in enumerate(self.objects):
            good = bool(valid[i] and not self.failed[i] and np.all(np.isfinite([center[i],height[i],direction[i],angle[i],ratio[i]])))
            rows.append({'object':o.name,'update':update,'valid':good,
                         'success':bool(good and height[i]>.1 and center[i]<.1),
                         'failure_reason':self.failure_reason[i], 'terminated_step':self.terminated_step[i],
                         'lift_height_m':float(height[i]) if good else None,
                         'center_error_m':float(center[i]) if good else None,
                         'direction_error':float(direction[i]) if good else None,
                         'angle_error_rad':float(angle[i]) if good else None,
                         'contact_ratio':float(ratio[i]) if good else None})
        return rows

    def close(self):
        from isaacsim.core.api import World
        self.world.stop()
        self.world.clear()
        World.clear_instance()
