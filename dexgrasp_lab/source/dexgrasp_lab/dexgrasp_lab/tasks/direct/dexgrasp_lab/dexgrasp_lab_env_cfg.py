from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg,PhysxCfg,RigidBodyMaterialCfg
from isaaclab.utils import configclass

@configclass
class DexgraspLabEnvCfg(DirectRLEnvCfg):
    decimation=4
    episode_length_s=1.3  # Source training rollout: 130 control steps, no evaluation lift schedule.
    action_space=28
    observation_space=268
    state_space=0
    seed=1
    hands:tuple[str,...]=('shadow','allegro','barrett')
    object_split:str='train'
    stage:str='floating'
    num_surface_points:int=200
    # Native free-base Cartesian servo; finger gains retain the source force-PD values.
    wrist_position_gain:float=100.
    wrist_velocity_gain:float=10.
    wrist_rotation_gain:float=10.
    wrist_angular_velocity_gain:float=1.
    wrist_position_scale:float=.005
    wrist_rotation_scale:float=.01
    finger_action_scale:float=.015
    root_guided:bool=True
    pregrasp_clearance:float=.04
    max_grasp_center_distance:float=.5
    failure_reward:float=-5.
    sim:SimulationCfg=SimulationCfg(dt=.0025,render_interval=4,
        physics_material=RigidBodyMaterialCfg(static_friction=3.,dynamic_friction=3.,restitution=0.),
        physx=PhysxCfg(solver_type=1,min_position_iteration_count=16,min_velocity_iteration_count=4))
    scene:InteractiveSceneCfg=InteractiveSceneCfg(num_envs=144,env_spacing=2.5,replicate_physics=False)
