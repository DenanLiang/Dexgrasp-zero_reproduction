#!/usr/bin/env bash

# ==============================
# DexGrasp-Zero + Isaac Sim
# ==============================


export ISAAC_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GRASPXL_ROOT="$(dirname -- "$ISAAC_PROJECT_ROOT")/dexgrasp_310/GraspXL"

export DEX_ROOT="$GRASPXL_ROOT/raisimGymTorch/dexgrasp_zero"

export PYTHONPATH="$GRASPXL_ROOT/raisimGymTorch${PYTHONPATH:+:$PYTHONPATH}"

export YCB_ROOT="$DEX_ROOT/rsc/ycb_crossDex"

export LEAP_URDF="$DEX_ROOT/rsc/leap_hand/leap_hand_right_envusing.urdf"

export DEX_MODEL="$DEX_ROOT/model/multi_hand/full_2700_r.pt"

export DEX_MODEL_CONFIG="$DEX_ROOT/model/multi_hand/model_gnn_cfg_L10_v4.yaml"
