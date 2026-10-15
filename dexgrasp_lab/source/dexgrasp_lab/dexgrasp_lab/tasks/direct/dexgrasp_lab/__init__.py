import gymnasium as gym
from .dexgrasp_lab_env_cfg import DexgraspLabEnvCfg

gym.register(id='DexGraspZero-CMAP-Direct-v0',entry_point=f'{__name__}.dexgrasp_lab_env:DexgraspLabEnv',
             disable_env_checker=True,kwargs={'env_cfg_entry_point':DexgraspLabEnvCfg})
