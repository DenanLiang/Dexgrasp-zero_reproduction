# DexGrasp-Zero 在 Isaac Lab 中重新训练

这是由 **Isaac Lab v2.3.2 官方模板生成器**创建的外部 Git 项目，使用 `DirectRLEnv`（Direct / single-agent workflow）。它在 PhysX 中从头训练一套跨手型共享的 DexGrasp-Zero GNN actor/critic，训练手为 **Shadow、Allegro、Barrett**。不加载上一轮 sim2sim 的策略权重。

数据仅使用本地 `DRO_data` 中的**对象 STL 网格**和 `CMapDataset_filtered/split_train_validate_objects.json`。`train` 的 48 个对象用于训练；经确认将 `validate` 的 10 个对象留作后续 test。不会读取 CMAP 接触图、DRO 抓取示范、DRO 网络或损失。Robotiq 3-finger 和 Sharpawave 的评估接入按当前约定留待后续。

## 安装与短训练

已在现有 `dexgrasp_isaacsim` 环境中完成 editable 安装；未替换 Isaac Lab、Isaac Sim、PyTorch 或 CUDA。

```bash
conda activate dexgrasp_isaacsim
cd /home/vlak/Dexgrasp-zero/dexgrasp_lab
# 若迁移到其他机器，先安装该外部包：
python -m pip install -e source/dexgrasp_lab --no-deps --no-build-isolation

# 默认 L6/v4 GNN；三种手各四个环境，只运行两次 PPO 更新
python scripts/train.py --headless --num_envs 12 \
  --iterations 2 --rollout 130 --epochs 1 --minibatches 2 \
  --output logs/my_trainability_check
```

每种手在这个短测试中只使用前 4 个训练对象。`--num_envs 144` 对应每手 48 个环境，覆盖全部 48 个训练对象；`--num_envs 288` 对应每对象、每手 2 个实例。环境数必须能被手型数 3 整除。对象几何固定分配到环境，回合重置时随机物体 yaw 和接近方向。

以下是后续全对象训练的启动方式，**本次没有执行长期训练**：

```bash
python scripts/train.py --headless --num_envs 144 \
  --iterations 10000 --output logs/full_training
```

恢复训练时，使用新的日志目录，`--iterations` 表示额外执行的更新次数：

```bash
python scripts/train.py --headless --num_envs 12 \
  --iterations 1 --rollout 16 --epochs 1 --minibatches 2 \
  --resume logs/my_trainability_check/checkpoint.pt \
  --output logs/my_resume_check
```

恢复会读取共享 actor/critic、动作分布、Adam 状态、随机数状态及迭代编号，并检查训练对象划分、手模型、网格、模型配置和实现代码哈希。仿真场景重新重置，不恢复上一个进程的 PhysX 内部状态，因此不承诺与不中断运行的轨迹逐位相同。更改代码或数据时，应明确开始新实验。

默认资产位置相对本仓库的父目录定位；换位置时可设置：

```bash
export DEXGRASP_SOURCE=/path/to/GraspXL/raisimGymTorch/dexgrasp_zero
export DRO_OBJECT_ROOT=/path/to/DRO_data
```

## 算法与环境

- 保留原工程 `GNN`、`GNNCritic`、URDF 22 维特征、图构建器、节点语义和 FLEX/ABD/ROT 动作映射。纯 Python 网络代码保存在 `algorithm/`，带来源哈希和原 MIT 许可证。没有引入 RaiSim 二进制依赖。
- 三种手共享同一套 actor 和 critic，按照手型构图。观测右侧补零到 268 维，动作补零到 28 维；对数概率、熵和 KL 都排除未使用的动作维度。
- Torch PPO 使用原训练入口的 `gamma=0.996`、`lambda=0.95`、clip 0.2、value coefficient 0.5、entropy coefficient 0、初始学习率 `5e-4`、KL 目标 0.01、梯度范数上限 0.5。默认 4 epochs / 4 minibatches。超时用**重置前**的终态 bootstrap，失败终止不 bootstrap。
- 物理步长 0.0025 s，动作每 4 步更新一次（0.01 s），训练回合 130 个控制步。训练不启用评估阶段的第 130 步抬升安排。
- 默认 `--stage floating`：动态物体放在桌上，复用原训练奖励项与系数，包括接触、冲量、关键点距离、中心/方向误差和速度惩罚。`fixed` 选项将物体设为运动学刚体，未作为本次主要验证阶段。
- 本次只使用对象集合，不含 affordance 标注，因此整个物体表面视为可抓取区域，非 affordance 通道为零。每对象固定采样 200 个表面点；目标为对象几何中心。这一设定与原工程带分区标注的数据不同。

### Barrett 的维度修复

| 手型 | 观测 | 动作（含 6 维手腕） | 接触通道 | 关键点 |
|---|---:|---:|---:|---:|
| Shadow | 268 | 28 | 17 | 22 |
| Allegro | 210 | 22 | 13 | 17 |
| Barrett | **130** | 14 | **9** | 9 |

统一公式为 `12 + 2*A + 4*C + 6*K`。Barrett 有 8 个手指自由度和 6 维手腕动作，接触通道包含掌部，因此 `12 + 2*14 + 4*9 + 6*9 = 130`。原 `constants.py` 的 186 和注释中的 8 个接触体均不正确；原文件和外部项目内的对应副本都已修正。训练环境直接实现 Barrett 的初始化和控制，不再依赖原 runner 中缺失的 Barrett 分支。

### PhysX 适配

手指关节按照原语义顺序显式对应 PhysX DOF，保留力驱动和原关节限位；初始关节角裁剪到有效范围（原 Allegro 的 1.7 rad 初始值超过部分 URDF 限位）。手腕 6 维动作改为**原生自由基座的 Cartesian 力/力矩控制**，不创建零质量虚拟手腕连杆。位置 PD 为 100/10，旋转 PD 为 10/1，包含重力补偿，力/力矩分别限制为 100 N / 10 Nm。手指采用 stiffness 100、damping 0.2 加 URDF 被动阻尼。参数位于环境配置和控制实现中。

物体使用单刚体凸分解碰撞，移除几何中心偏移。数据没有质量标注，质量按凸包体积 × 500 kg/m³ 估算，并限制在 0.01–2 kg；这一假设会写入转换缓存。摩擦系数 3，恢复系数 0，桌面高 0.5 m。手与桌子碰撞被排除，手与物体、手部自碰撞保留；不同环境彼此隔离。

初始化按对象包围球和额外 4 cm 间距放置手的抓取中心，减少初始穿透。中心误差超过 0.5 m 或对象离开工作区时，回合以 -5 奖励终止并重置。非有限状态单独计数并终止。原 URDF 中的缺失/非正质量与惯量仅在导入副本中修正，记录在 `cache/hand_*/complete.json`。

接触观测为 PhysX 对指定对象的法向冲量，每个手型的传感器只匹配其对象。环境、对象和关节均显式映射实际 PhysX 实例顺序，支持不同手型的异构环境和部分重置。这些物理/初始化调整是本次迁移的一部分，不能将新训练轨迹解释为 RaiSim 的逐步复刻。

## 检查与输出

```bash
python scripts/validate_contracts.py
python -m unittest discover -s tests -v
python scripts/smoke_env.py --headless --num_envs 6 --steps 140
python scripts/list_envs.py
```

任务 ID：`DexGraspZero-CMAP-Direct-v0`。模型配置在 `source/dexgrasp_lab/dexgrasp_lab/configs/model_gnn.yaml`；环境实现位于 `tasks/direct/dexgrasp_lab/`。

每个训练目录保存：

- `manifest.json`：训练划分、实际对象分配、版本、物理参数、代码与数据哈希。
- `metrics.jsonl`：每次 PPO 更新的损失、梯度范数、权重变化、分手型奖励/接触比例、失败计数。
- `checkpoint.pt`：网络、分布、优化器、随机数状态与迭代编号。
- `summary.json`：本次实际完成的更新数、transition 数和检查结果。

短测试证明训练流程可执行、梯度和优化器可更新，不证明已经学会抓取或取得跨手型泛化效果。大规模训练及保留对象/评估手型的成功率评估留待后续。

本次实际验证记录见 [artifacts/validation_summary.json](artifacts/validation_summary.json)。最终短训练 `logs/trainability_v2` 使用 12 个环境（三种手各 4 个）、4 个训练对象，完成 2 次 PPO 更新、3,120 个 transition，损失有限且 actor 权重均有更新；数值异常为 0，另有 6 次正常的越界失败终止。`logs/resume_check` 成功恢复到第 3 次更新，新增 192 个 transition，Adam 步数从 4 增至 6。全部 58 个对象通过网格与划分检查，但这次物理短训练只覆盖其中 4 个训练对象。早期调试目录 `logs/trainability_check` 不作为最终验证结果。

外部项目结构依据 [Isaac Lab v2.3.2 官方模板文档](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/own-project/template.html)，生成参数见 `template_spec.json`。原网络代码许可证见 `algorithm/LICENSE`；手模型和数据保留各自来源许可。
