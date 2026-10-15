# DexGrasp-Zero on Isaac Lab

This project ports DexGrasp-Zero to **Isaac Lab 2.3.2 / Isaac Sim 5.1**, using an external project structure and a **Direct-based workflow** to train a shared GNN policy from scratch across Shadow, Allegro, and Barrett hands.

Only object meshes and splits from DRO's filtered CMAP dataset are used: 48 `train` objects for training and 10 `validate` objects reserved for future testing. DRO algorithms and grasp annotations are not used. The Barrett observation dimension has been corrected to **130**. Short training runs and checkpoint resumption have been verified; evaluation with Robotiq 3-finger and Sharpa Wave is planned for later.

## Environment Versions

The following main versions have been verified locally in the `dexgrasp_isaacsim` Conda environment:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| PyTorch Geometric | 2.8.1 |

Linux, a working NVIDIA GPU, and a compatible driver are required. See the [official installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) for Isaac Sim installation and system requirements. The steps below configure these versions; skip the base environment installation if you already have a working environment.

## 1. Install the Base Environment

Run from the project root:

```bash
export DEXGRASP_ROOT="$(pwd -P)"
conda create -n dexgrasp_isaacsim python=3.11 -y
conda activate dexgrasp_isaacsim
python -m pip install --upgrade pip

python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' \
  --index-url https://download.pytorch.org/whl/cu128

# Skip cloning if IsaacLab v2.3.2 is already present in this directory.
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
./isaaclab.sh --install none
cd "$DEXGRASP_ROOT"
```

This project uses its own PPO implementation, so `--install none` skips additional RL backends. Follow the license acceptance prompt when launching Isaac Sim for the first time.

## 2. Install the Training Project

Run from the project root with the `dexgrasp_isaacsim` environment activated:

```bash
python -m pip install \
  'numpy==1.26.0' 'scipy==1.15.3' 'trimesh==4.5.1' \
  'PyYAML==6.0.2' 'urdf-parser-py==0.0.4' 'torch-geometric==2.8.1'
python -m pip install -e dexgrasp_lab/source/dexgrasp_lab \
  --no-deps --no-build-isolation
```

There is no need to compile RaiSim or source the old `dexgrasp_isaacsim_311/env.sh`. Keep the original project's Python 3.10 environment, `dexgrasp310`, separate from this environment.

## 3. Data and Model Paths

Default directory structure:

```text
Dexgrasp-zero/
├── IsaacLab/                 # v2.3.2
├── dexgrasp_lab/             # Current training project
├── dexgrasp_310/GraspXL/     # Original project; provides hand URDFs and meshes
└── DRO_data/
    ├── CMapDataset_filtered/split_train_validate_objects.json
    └── data_urdf/object/     # contactdb and ycb object STL files
```

Hand assets come from the original DexGrasp-Zero project; object assets come from [DRO-Grasp](https://github.com/zhenyuwei2003/DRO-Grasp). The data is already available locally. No path configuration is needed with the layout above. For assets stored elsewhere, set:

```bash
export DEXGRASP_SOURCE=/path/to/GraspXL/raisimGymTorch/dexgrasp_zero
export DRO_OBJECT_ROOT=/path/to/DRO_data
```

Assets are converted automatically on the first run and cached in `dexgrasp_lab/cache/`.

## 4. Verify the Environment

Run from the project root:

```bash
conda activate dexgrasp_isaacsim
cd dexgrasp_lab
python scripts/validate_contracts.py

# Four environments per hand across three hands; only two PPO updates.
python scripts/train.py --headless --num_envs 12 \
  --iterations 2 --rollout 130 --epochs 1 --minibatches 2 \
  --output logs/my_trainability_check
```

Use a new output directory name for each run. Training logs and checkpoints are saved there. Asset conversion may take longer on the first run.
