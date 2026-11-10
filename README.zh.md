# MB-ADR

中文 | [English](README.md)

**M**odel-**B**ased **A**ny**D**ex**R**etargeting。

面向多种灵巧手的端到端、**model-based** 神经网络 retargeting 方案。

用一个学习出来的网络替代逐帧 NLopt 求解，同时保持优化器的目标函数和机器人运动学作为唯一的真值来源。

本仓库完全自包含：自带各手型定义、URDF 与 MuJoCo 资产、retargeting 配置、NLopt 参考求解器、可微 FK，以及 Pico 4 输入驱动。不依赖仓库之外的任何代码，唯一的外部依赖都是常规的库（numpy、torch、nlopt、mujoco、pinocchio、scipy）。

![六只手对同一段关键点录像做 retargeting](docs/demo_grid.gif)

*l20、Shadow、Wuji、Sharpa、Gaia、Inspire —— 同一份输入，每只手各自的网络，同一套场景（背景与掌心朝向在 `mbadr/scene.py` 中统一处理）。*

## "model-based" 在这里指什么

监督信号来自一个**解析的、可微的正运动学（FK）模型**，而不是学出来的 FK 代理，也不是关节角标签：

```
人手关键点 ──► 自适应目标 ──► loss ◄── 解析 FK ◄── 网络输出
                  (纯 numpy)     │
                                 └─── 梯度来自 autograd
```

这个选择带来三个直接后果，每一个都关系到指尖精度：

| | 学习 FK 代理的方案 | 本项目 |
|---|---|---|
| 训练时用的 FK | 对采样姿态拟合出的 MLP 代理 | 精确的 URDF 运动学链，与 Pinocchio 对齐到 1e-16 m |
| mimic 关节 | 当作独立自由度预测，之后再做修正 | 在 FK 内部展开，始终保持一致 |
| 网络预测的关节 | 全部 nq 个，包括从属关节 | 仅独立自由度 |

这个差异不是表面上的。如果网络预测了远端关节、之后又被 PIP 关节的值覆盖，指尖就会落到网络从未优化过的位置。只预测独立自由度从结构上消除了这个失败模式，因为关节耦合关系被固化在 FK 里，loss 看到的就是硬件实际会到达的位置。

## 支持的手型

| `--robot` | nq | 独立自由度 | mimic | 备注 |
|---|---|---|---|---|
| `l20` | 21 | 16 | 5 | DexForce Linker L20 |
| `shadow` | 22 | 22 | — | Shadow Hand |
| `wuji` | 20 | 20 | — | Wuji Hand |
| `sharpa` | 22 | 22 | — | Sharpa Wave |
| `gaia` | 20 | 16 | 4 | Gaia Hand20 |
| `inspire` | 12 | 6 | 6 | Inspire Hand，欠驱动程度最高 |

手型定义在 `mbadr/robot_configs.py`，资产放在 `assets/`，配置放在 `configs/pico4/`。新增一只手只需要在那里加一条记录、补上对应的 URDF/MJCF；FK、loss、训练和评估代码都是手型无关的。左右手的 link 命名约定按厂商各自镜像处理（`rh_`→`lh_`、`right_`→`left_`）。

## 目录结构

```
mbadr/
  chain.py             URDF -> 扁平化运动学链描述，mimic 关节记录
  config.py            HandSpec：每只手的 link/偏移/URDF/MJCF 解析
  robot_configs.py     手型表（link 名、偏移、资产路径）
  mediapipe.py         关键点坐标系转换（手腕坐标系 -> MANO）
  fk_numpy.py          NumPy FK + 解析 Jacobian
  fk_torch.py          批量化可微 FK，mimic 展开实现为线性映射
  reference.py         NumPy + NLopt 参考求解器（"标准答案"）
  loss_torch.py         同一个目标函数的 Torch 版本，用于训练
  data.py              数据加载、去重、带解剖学检查的数据增强
  model.py             做摊销推理的 MLP，关节限位作为硬约束
  mjcf.py              URDF 关节顺序 <-> MuJoCo qpos 的映射
  scene.py             统一的可视化场景：背景、手指朝上掌心朝前的姿态
  pico4.py             Pico 4 输入驱动（relay / direct）
  train.py             自监督训练循环
  evaluate.py          在相同关键点上对比网络与 NLopt
scripts/
  verify_fk.py         NumPy FK/Jacobian 对照 Pinocchio
  verify_fk_torch.py   Torch FK 对照 Pinocchio（前向、autograd Jacobian、CUDA）
  verify_loss.py       Torch loss 对照 NumPy 参考实现（数值与梯度）
  demo_mujoco.py       用网络回放一段录像，在 MuJoCo 中展示
  demo_pico4.py        Pico 4 实时输入 -> 网络 -> MuJoCo
  check_nlopt_budget.py  网络的优势是真实的，还是 NLopt 没收敛到位？
  summarise.py         汇总各手型的评估结果
assets/                六只手的 URDF / MJCF / mesh
configs/pico4/         每只手的 retargeting YAML
data/                  关键点数据集（.npy）以及 recordings/（.pkl，供 demo 使用）
out/                   checkpoint 与评估结果 JSON
```

## 正确性验证

目标函数有两份独立实现（Torch 和 NumPy），互相校验；运动学部分对照 Pinocchio 校验。这些检查不是摆设——靠它们真的发现了四个 bug：

1. mimic 关节的限位裁剪语义问题（`index_pip` 上限 × 0.89 会超出 `index_dip` 的上限）；
2. Torch FK 里的矩阵乘法顺序写反了（URDF 的关节旋转应作用在右侧）；
3. URDF 关节顺序与 Pinocchio 关节顺序之间的转换问题；
4. `AdaptiveParams` 的默认值当初是从 L20 的 YAML 里抄来的，而不是参考实现自己的默认值——这个问题在只支持 L20 时完全不可见，因为它的配置文件把每个键都显式写出来了。

```bash
python scripts/verify_fk.py --robot all       # 位置误差 1e-16 m，Jacobian 误差 1e-10
python scripts/verify_fk_torch.py             # 前向误差 3.7e-8 m（float32），Jacobian 误差 9.7e-17
python scripts/verify_loss.py --robot all     # loss 误差 2.3e-13，梯度误差 5.5e-8
```

## 训练

```bash
python -m mbadr.train --robot shadow --data-npy data/qsq_pico4_merged_v1.npy \
    --steps 12000 --out out/net_shadow.pt
```

训练只需要关键点数据，不收集也不使用任何关节角标签。使用 `--data-npy` 时，会按时间顺序留出采集数据最后 15% 作为验证集，因此验证集与训练集在时间上相邻但互不重叠。

## 评估

```bash
python -m mbadr.evaluate --robot shadow --checkpoint out/net_shadow.pt \
    --data-npy data/qsq_pico4_merged_v1.npy
python scripts/summarise.py
```

评估帧按 pinch 覆盖率做分层抽样：真实数据里大约三分之一的帧存在 pinch，但往往连续成段出现，直接取前 N 帧很可能一个 pinch 帧都抽不到，从而悄悄漏掉最关键的那项指标。

## 可视化

```bash
# 用录像回放，不需要硬件
python scripts/demo_mujoco.py --robot shadow --recording pico4_check.pkl

# Pico 4 实时输入（relay 守护进程监听 127.0.0.1:63902，或使用 --mode direct）
python scripts/demo_pico4.py --robot l20 --hand right
```

## 环境搭建

```bash
conda env create -f environment.yml      # python 3.10 + numpy/scipy/pinocchio/mujoco
conda activate mbadr
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按你的驱动版本选择
pip install nlopt
```

项目所需的一切都在这个仓库里：手型定义、URDF 与 MuJoCo 资产、retargeting 配置、关键点录像。
