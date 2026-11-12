# 新增灵巧手

[English](ADDING_HAND.md) · [返回 README](../README.zh.md)

本教程以虚构的五指手 `my_hand` 为例。示例中的 link 名称和路径需要替换成模型中的真实名称；它不是项目已支持的第七种手型。请在完整源码目录中操作，先完成训练与仿真，再接真机。

## 1. 准备机器人模型

将 URDF、mesh 和可选的 MuJoCo 模型放在同一个资产目录中，例如：

```text
assets/my_hand/
  right.urdf
  right.xml
  meshes/
```

URDF 是训练与 FK 的依据，MJCF 是 MuJoCo 可视化的依据。仅训练不要求 MJCF；运行仿真示例需要可加载的 MJCF，当前项目没有通用的一键 URDF 转 MJCF 命令。

- 使用米和弧度，检查关节轴、原点及限位；缺失限位的普通活动关节会被读成 `[0, 0]`。
- 使用固定根的运动学树。当前 FK 面向固定、旋转及移动关节，不处理自由根、球关节或闭环机构。
- URDF 和 MJCF 中由模型驱动的关节应使用相同名称，且零位、方向和限位一致。MuJoCo 映射按名称匹配，不按文件中的排列位置匹配。
- 耦合关节使用 URDF `<mimic joint="..." multiplier="..." offset="..."/>`。源关节必须是独立关节；当前不支持 mimic 驱动另一个 mimic。

## 2. 注册手型和任务点

在 [robot_configs.py](../mbadr/robot_configs.py) 的 `ROBOT_CONFIGS` 中加入一项。下面的命名假设每根手指有 proximal、middle、distal 和 tip frame：

```python
"my_hand": {
    "origin_link": "palm",
    "tip_links": [
        "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
    ],
    "link1_names": [
        f"{f}_proximal" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "link3_names": [
        f"{f}_middle" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "link4_names": [
        f"{f}_distal" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "num_fingers": 5,
    "urdf_subdir": "assets/my_hand",
    "urdf_file": {"right": "right.urdf"},
    "mjcf_subdir": "assets/my_hand",
    "mjcf_file": {"right": "right.xml"},
},
```

所有手指数组都按 **拇指、食指、中指、无名指、小指** 排列。这里的 link 必须是 URDF 中存在的 link；不能填关节名。

| 字段 | 用途 |
|---|---|
| `origin_link` | 掌部参考 frame |
| `tip_links` | 指尖任务点所在的 link |
| `link1_names` | 近节 frame；也用于仿真场景的朝向估计 |
| `link3_names` / `link4_names` | 与人手 PIP / DIP 任务点对应的 link |
| `tip_offsets` / `link3_offsets` / `link4_offsets` | 可选的局部坐标偏移，每指一个 `[x, y, z]`，单位米；默认全零 |

先查看各 link 原点是否真在关节或接触位置。若 tip link 已位于接触点，不要再添加一次同样的 `tip_offsets`。偏移在所属 link 的局部坐标系中定义。

规范名称 `my_hand` 会自动被 `resolve_robot_name` 接受；短名称需另外添加到 [config.py](../mbadr/config.py) 的 `ROBOT_ALIASES`。示例只注册右手。要支持左手，补上左手文件，并检查 `_mirror_name`；新厂商不会自动套用 `right_` → `left_` 规则。

四指模型可设置 `num_fingers: 4`，默认映射拇指到无名指并丢弃小指。其他手指选择、顺序或拓扑需要同步调整关键点索引与目标构造。

## 3. 添加 retargeting YAML

创建 `configs/pico4/pico4_my_hand.yaml`。默认路径使用**规范名称**，不是短别名。以下配置可作为结构模板，数值仍需按手型标定：

```yaml
optimizer:
  type: AdaptiveOptimizerAnalytical
robot:
  type: my_hand
retarget:
  huber_delta: 2.0
  huber_delta_dir: 0.5
  norm_delta: 0.04
  w_pos: 5.0
  w_dir: 1.0
  w_full_hand: 1.0
  w_contact: 6.0
  alpha: 1.0
  pinch_scaling: 1.0
  lateral_scaling: 1.0
  segment_scaling:
    thumb: [1.0, 1.0, 1.0, 1.0]
    index: [1.0, 1.0, 1.0, 1.0]
    middle: [1.0, 1.0, 1.0, 1.0]
    ring: [1.0, 1.0, 1.0, 1.0]
    pinky: [1.0, 1.0, 1.0, 1.0]
  pinch_thresholds:
    index: {d1: 1.0, d2: 1.5}
    middle: {d1: 1.0, d2: 1.5}
    ring: {d1: 2.0, d2: 4.0}
    pinky: {d1: 2.0, d2: 4.0}
  mediapipe_rotation:
    right: {x: 0.0, y: 0.0, z: 0.0}
    left: {x: 0.0, y: 0.0, z: 0.0}
```

`segment_scaling` 的四个值依次缩放手腕 → MCP、MCP → PIP、PIP → DIP、DIP → TIP 的段向量，再累加得到目标点，不是四个电机增益。先用张掌、握拳和逐指捏合检查模型，再调整尺度。`mediapipe_rotation` 是归一化手腕坐标系之后的 XYZ 旋转，单位度；`d1/d2` 是捏合距离阈值，单位厘米，且应满足 `d1 < d2`。这些参数不能直接照搬 L20 的标定值。

## 4. 检查模型并训练

新增定义和 YAML 后，先打印解析结果，核对关节名、顺序与限位：

```bash
python - <<'PY'
from mbadr.chain import KinematicChain
from mbadr.config import HandSpec, default_config_path

spec = HandSpec.from_name("my_hand", "right")
chain = KinematicChain.from_urdf(spec.urdf_path)
print(default_config_path("my_hand"))
print("full joints:", chain.joint_names)
print("independent joints:", chain.independent_names)
print("limits:", chain.independent_limits())
PY

python scripts/verify_fk.py --robot my_hand --hand right
python scripts/verify_loss.py --robot my_hand --hand right --frames 40

python -m mbadr.train --robot my_hand --hand right \
    --data-npy data/qsq_pico4_merged_v1.npy \
    --steps 12000 --out out/net_my_hand.pt

python -m mbadr.evaluate --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --data-npy data/qsq_pico4_merged_v1.npy \
    --limit 100 --out out/eval_my_hand.json
```

训练只需要人手关键点，不需要机器人关节角标签。可以先用自带数据打通流程，再收集覆盖目标动作的数据。新手型要训练自己的权重；关节数相同不代表可以复用其他手型的 checkpoint。改变任务点、限位或标定后也应重新训练。

检查命令需要安装 `verification` 依赖。`verify_fk.py` 和 `verify_loss.py` 的 `--robot all` 当前使用固定名单，纳入持续检查时需补上新手型；`verify_fk_torch.py` 当前专门检查 L20，不能通过 `--robot` 切换。

## 5. 回放、实时仿真和真机

准备好 MJCF 后，先运行录像回放，再接 Pico：

```bash
python scripts/demo_mujoco.py --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --recording pico4_check.pkl

# 另一个终端先运行 Pico relay；头显连接见 Pico 教程
python scripts/demo_pico4.py --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --device cpu
```

检查张掌、握拳、逐指弯曲和捏合时的方向、指尖位置以及所有目标关节是否被映射。[Pico 连接步骤](PICO_SETUP.md) 可直接复用。`KeypointRetargeter` 返回的是 URDF 活动关节顺序；仿真需要经过 [MJCF 名称映射](../mbadr/mjcf.py)。

这只完成模型与仿真接入。其他手型的硬件输出需要各自的通信协议、关节到执行器映射、零位/方向/限位及失联处理；[L20 驱动](../mbadr/l20.py) 可作为结构参考，但其寄存器和标定仅适用于 L20。

提交新增手型时，同步更新支持列表、模型来源记录与 `assets/manifest.json`；新增公开数据/权重时更新 `artifacts/manifest.json`。在 [贡献指南](../CONTRIBUTING.md) 中查看全仓库检查命令。
