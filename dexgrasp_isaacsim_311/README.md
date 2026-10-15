# DexGrasp-Zero：Isaac Sim / PhysX 的 YCB 跨引擎评估

使用已有 `full_2700_r.pt`，在 Isaac Sim 5.1 中执行 YCB 抓取。支持 Allegro、Shadow、LEAP、Inspire、Ability、Schunk、BrainCo、Barrett 八种手型。此目录是新仿真后端；原 `dexgrasp_310` 源码、权重、标签和模型资产不作修改。

这是现有策略的 **RaiSim → PhysX 迁移评估**，不是在 PhysX 中重新训练。成功率应标注仿真后端，不能直接称为原论文在 RaiSim 上的复现成功率。

## 环境与运行

```bash
conda activate dexgrasp_isaacsim
cd /home/vlak/Dexgrasp-zero/dexgrasp_isaacsim_311
source env.sh
```

`env.sh` 根据脚本位置定位项目，直接使用 `dexgrasp_zero/rsc`，避开 GraspXL 下指向旧安装位置的软链接。评估使用原项目的图构建器、URDF 语义解析器、GNN、checkpoint，以及在原 Python 3.10 环境中从现有标签导出的固定初始姿态；不导入 Python 3.10 的 RaiSim 扩展。

本次补装了 `joblib` 和 `scikit-learn` 用于标签和姿态兼容性检查；正式评估直接读取 `reference/initial_poses.npz`。完整依赖列表见 `requirements-eval.txt`；不要为运行本目录而替换已经安装的 Isaac Sim、PyTorch 或 CUDA。

输入和模型检查（不启动仿真，允许 CPU）：

```bash
python evaluate.py --hand leap --preflight
```

小样本完整抓取：

```bash
python evaluate.py --hand leap \
  --objects 003_cracker_box 025_mug --updates 2 \
  --output results/my_smoke
```

全部八种手型、两张 GPU 分工：

```bash
python -u run_all.py --gpus 0 1 --updates 25 --batch-size 78 \
  --output results/my_full_ycb
```

仅一张 GPU 时使用 `--gpus 0`。每个工作进程使用一张 GPU 进行策略推理，PhysX 使用 CPU TGS 求解器，并通过 Fabric/物理张量读取状态，关闭逐物理子步的 USD 写回；每个手型在独立 Isaac Sim 进程中运行，78 个对象放在独立碰撞组中并行模拟。

单手完整评估：

```bash
python evaluate.py --hand shadow --updates 25 --batch-size 78 \
  --output results/my_shadow_ycb
```

中断后继续运行时，重复相同命令并添加 `--resume`。模型、代码、数据标签、对象清单、种子或评估参数发生变化会拒绝续跑，应使用新的输出目录。只补充缺失的 `(object, update)`，不会重复计数。请勿让两个评估进程同时写同一结果目录。

固定种子和初始姿态不等于 PhysX 轨迹逐位确定。进程重建、求解器状态和浮点计算可能造成轨迹差异；`--resume` 保证记录覆盖与去重，不承诺与不中断的轨迹完全一致。

`--visualize` 可显示单手评估场景，推荐只选择一个对象。`--dump-reset` 保存初始观测与关键点构建输入；`--trace` 保存逐步观测、动作和关节状态，用于排查跨引擎差异。默认无界面、无实时限速。

## 评估协议

- 数据是本项目保存的全部 **78 个 YCB 对象**；没有只取常见的 21 个子集。
- 每对象使用已有 `*_label.pkl` 的 25 个姿态（由原 `dexgrasp310` 环境导出，避免 sklearn PCA 版本的轴符号差异），共每手 1,950 次、八手 15,600 次。
- 每次 200 个控制步；物理步长 0.0025 s、控制步长 0.01 s，每个动作推进四个物理步。
- 在零起始索引的第 130 步启用抬升引导。
- 成功要求同时满足：最终物体高度相对初始高度 **严格大于 0.1 m**，抓取中心误差 **严格小于 0.1 m**。
- 策略使用确定性均值动作，未增加动作噪声；维度和顺序按原手型约定映射，不能使用 USD 导入器的默认关节排列。
- 每个物体、每个部件采样 200 个表面点；通过对象名和种子固定采样，独立于批次排列，实际采样点另存文件。
- 所有试验都纳入分母；姿态、观测或动作数值无效时，该试验标记 `valid=false`，记录 `failure_reason` 和 `terminated_step`，将其刚体移出接触区域并计为失败，不重试该试验。其他对象继续运行；资产或未识别的程序异常仍终止当前手型，汇总不会把未完成手型当作完成。

## 输出与核验

`run_all.py` 为每种手型保存一个目录和一个 `.log`，最终生成 `all_hands_summary.json`。每个手型目录包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 参数、依赖版本、权重/代码/标签/URDF 的 SHA-256 与仿真假设 |
| `surface_points.npz` | 实际用于观测的固定表面采样 |
| `trials.jsonl` / `trials.csv` | 每次试验的对象、姿态编号、成功与否、高度变化、中心/方向/角度误差和接触比例 |
| `summary.json` | 已完成/应完成试验数、总成功率、每对象成功率、全部有效试验和成功子集的指标均值 |

每完成一个姿态批次就写入并同步试验记录，更新汇总。只有 `complete=true` 才表示该目录覆盖了清单中的全部试验。

独立核验覆盖率、重复记录、成功判据和汇总计数：

```bash
python tools/check_results.py results/my_full_ycb
# 生成八手型汇总 CSV、15,600 条试验 CSV、每对象结果和成功率图
python tools/report_results.py results/my_full_ycb
```

## 已完成的接口对照

`tools/validate_contracts.py` 在原 `dexgrasp310` 环境中调用真实 RaiSim C++ 环境，核对八种手型的关节顺序、关键点正运动学、抓取中心和观测维度。验证使用同一初始姿态，刷新原环境缓存后比较；它不声称两种物理求解器的轨迹相同。

```bash
# 在项目根目录执行
source dexgrasp_310/env.sh
/home/vlak/miniconda3/envs/dexgrasp310/bin/python \
  dexgrasp_isaacsim_311/tools/validate_contracts.py
```

本次八手型最大几何误差低于 `4e-8 m`，关节顺序误差为零。质量也逐一对照 RaiSim `getTotalMass()`，验证记录在 `results/contract_validation.json` 和 `results/mass_validation.json`。`tools/source_baseline.py leap` 可执行同标签、同表面采样的原版小样本轨迹对照。原环境日志本身不作为 Isaac Sim 结果。

### 固定初始姿态的跨环境核验

仅复用标签还不足以保证相同姿态：本次发现 Python 3.10 与 3.11 环境中的 PCA 实现在部分对象上会翻转轴方向。`reference/initial_poses.npz` 保存了原环境生成的全部 15,600 个腕部位置、旋转矩阵、物体位置和目标中心；`.json` 保存标签、网格、高度文件及生成器哈希。正式评估验证这些哈希后直接读取姿态，不在新环境中重新计算 PCA。

```bash
# 必须使用原 dexgrasp310 解释器重新生成；仅在原标签/网格改变时需要
/home/vlak/miniconda3/envs/dexgrasp310/bin/python tools/export_initial_poses.py reference/initial_poses.npz
# 使用新环境逐元素验证读取结果
python tools/verify_reference_poses.py
```

本次核验覆盖全部 15,600 个姿态，读取误差为零。独立记录位于 `results/initial_pose_validation.json`。另外，八种手型在两套 Torch/PyG 环境中以 CPU 推理同一组观测的策略输出也逐元素一致，记录位于 `results/policy_validation.json`。

## 仿真适配与明确差异

- 动作缩放保持原值：腕部平移/旋转 0.005，手指 0.015；保持各手型 PD 增益、手部 URDF 力矩/速度限制、腕部重力补偿和抬升引导。PhysX 使用力驱动；PD 阻尼叠加原 URDF 的被动阻尼，避免导入后覆盖被动阻尼。物体关节使用零刚度和 URDF 阻尼；该被动阻尼驱动的力上限设为 `1e6`，避免被作为主动电机力矩裁剪。
- 摩擦系数 3.0、恢复系数 0；排除手与桌子的碰撞，保留手与物体及手的自碰撞，并隔离不同环境。
- PhysX 不能使用零质量的虚拟腕部连杆，导入副本使用 `1e-5 kg` 和 `1e-8 kg m²`。Shadow 原 URDF 的非正惯量特征值下限设为 `1e-8 kg m²`；保留其余值。所有转换记录保存在 `cache/*/conversion.json`。
- Schunk 的重复 `<inertial>` 使用最后一个，已实测与 RaiSim 的质量一致。原图网络仍读取未经修改的原 URDF。
- 数字、句点、连字符等 USD 非法名称在导入副本中统一转换；ROS `package://` URI 解析到原资产目录。USD 生成使用跨进程锁，避免双 GPU 首次转换时的缓存写入竞争。
- 物体碰撞网格使用 PhysX 凸分解。接触观测取 PhysX 每连杆/部件的法向冲量；RaiSim 取最后一个接触的冲量范数，两者并不完全等价。原 C++ 未列出的接触刚体映射到通道 0 的行为也被保留。
- 桌面高度/尺寸保持原值，使用静态桌子；原版桌子是 100 kg 动态刚体。地面使用本地静态几何体，无需联网下载场景。
- 重置时使用当前试验的新目标和清零的“上一 PD 目标”缓冲。原版首次观测可能残留上一试验目标/PD 缓冲；这里消除了试验顺序带来的耦合，并在 manifest 中注明。
- Inspire 保留原实现对拇指的重复中心加权；Barrett 使用原评估器写明的固定腕部姿态覆盖，并直接保留标签目标中心，绕过原初始姿态辅助函数未实现 Barrett 的分支。Barrett 实际观测为 130 维，而非旧配置中的 186 维。

这些差异以及 PhysX/RaiSim 的接触求解差异可能降低零样本迁移成功率。没有为了提高数字而放宽判据、排除困难物体或更改策略权重。若后续需要接近原论文的成功率，应另开实验进行控制参数校准或 PhysX 上的策略训练，并保留本次迁移基线。

`tools/validate_numerical_isolation.py` 在真实 PhysX 场景中注入非法四元数和 NaN 动作，检查单试验异常隔离、失败计数和下一次重置。注意：`valid=true` 仅代表数值检查通过，并不保证跨引擎动力学一致；很大的有限位移仍可能反映动力学发散，应结合逐试验指标分析。

异常隔离保障大规模评估的统计完整性，PhysX 发散本身仍作为失败保留。若要研究稳定动力学或提高迁移成功率，还需要单独验证接触几何、虚拟连杆惯量和控制参数；这些后续实验应使用独立结果目录。

参考：[NVIDIA Isaac Sim 5.1 URDF 导入器](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.asset.importer.urdf/docs/index.html)。
