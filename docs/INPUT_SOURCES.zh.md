# 接入其他输入源

[English](INPUT_SOURCES.md) · [返回 README](../README.zh.md)

相机、其他 VR 设备或自定义网络接收器都可以通过 Python API 提供人手关键点。现有 `demo_pico4.py` 和 `mbadr-l20` 仍直接创建 `Pico4`，没有通用的 `--input` 参数；接新设备时编写自己的采集入口，并复用下面的推理接口。

## 1. 统一关键点格式

送入 [KeypointRetargeter](../mbadr/live.py) 的单帧数据为数值型 `numpy.ndarray`，形状 `(21, 3)`，单位**米**。索引沿用 MediaPipe 21 点顺序：

| 索引 | 位置，按指根到指尖排列 |
|---|---|
| 0 | 手腕 |
| 1–4 | 拇指 CMC、MCP、IP、TIP |
| 5–8 | 食指 MCP、PIP、DIP、TIP |
| 9–12 | 中指 MCP、PIP、DIP、TIP |
| 13–16 | 无名指 MCP、PIP、DIP、TIP |
| 17–20 | 小指 MCP、PIP、DIP、TIP |

在采集适配层中完成以下转换：

1. 根据 SDK 的点位定义重新排列索引。不要把任意 21 个点直接截取过来，也不要将关节旋转四元数当作位置。
2. 将毫米除以 1000、厘米除以 100。图像像素坐标或归一化的二维坐标不能直接作为米制三维点使用。
3. 明确真实的左右手、相机镜像及坐标系反射约定。预处理会去掉整体平移和刚体旋转，但不能据此忽略镜像差异。用张掌、拇指外展和捏合动作检查映射。
4. 可先减去第 0 点使手腕居中；预处理内部也会居中，因此不要求输入一开始位于原点。
5. 在 SDK 报告丢失、低置信度或缺点时丢弃该帧，不用全零或上一次姿态冒充新采样。所有值必须有限，掌部不能退化成一点或一条直线。

Pico 的 26 → 21 映射仅适用于其 SDK 点序，参见 [JOINT_21_INDICES](../mbadr/pico4.py)。其他设备需建立自己的映射表。

[preprocess_keypoints](../mbadr/data.py) 负责手腕坐标系 → MANO、YAML 中的旋转标定及掌长归一化（手腕到中指 MCP 为 0.092 m）。`KeypointRetargeter` 会自动调用它；**传入原始适配后的点，不要先预处理再调用推理接口**。训练数据也应保存同一阶段的原始点。

## 2. 先运行一次与设备无关的推理

在仓库根目录运行以下例子。它使用自带录像数组证明输入接口可用，不建立 Pico 或串口连接：

```python
from pathlib import Path

import numpy as np
import torch

from mbadr.live import KeypointRetargeter

rt = KeypointRetargeter(
    checkpoint=Path("out/net_l20.pt"),
    hand="right",
    device=torch.device("cpu"),
    robot="l20",
    smooth=0.3,
)
frames = np.load("data/qsq_pico4_merged_v1.npy", allow_pickle=False)
for raw in frames:
    q = rt(raw)
    if q is not None:
        print(dict(zip(rt.chain.joint_names, q.tolist())))
        break
else:
    raise RuntimeError("No valid hand frame")
```

接 SDK 后，将 `raw` 换成按上节约定转换的单帧数组。输出 `q` 包含 mimic 展开后的全部活动关节，顺序为 `rt.chain.joint_names`；L20 为 21 个弧度值。其他模型若含移动关节，对应位置量使用米。无效帧返回 `None`。

模型、手型、左右手和标定需要匹配。自带 checkpoint 训练于右手；切换 `hand="left"` 不会自动把右手权重变成左手训练结果。

## 3. 实时采样与新鲜度

建议设备适配层在关键点之外携带以下信息。这是采集入口应维护的约定，不是已有 SDK 基类：

| 信息 | 处理方式 |
|---|---|
| `hand` | 与推理器选择的左右手一致 |
| `active` / 置信度 | 用于在推理前拒绝丢失或不可靠的采样 |
| `sequence` | 每次真正新采样的标识；重连后可包含会话编号 |
| `received_at` | 收到新帧时的本机 `time.monotonic()` 时间 |

下面是采集回调可复用的示例，使用上节已创建的 `rt`。`points` 是完成索引、单位及坐标约定转换的原始点：

```python
import time

last_sequence = None

def on_sample(points, *, hand, active, sequence, received_at):
    global last_sequence
    age = time.monotonic() - received_at
    if not active or hand != rt.hand or not 0.0 <= age <= 0.5:
        return None
    if sequence == last_sequence:
        return None
    last_sequence = sequence
    q = rt(points)
    if time.monotonic() - received_at > 0.5:
        return None
    return q
```

仅在新采样到来时调用，不要在轮询缓存时更新 `received_at`。推理完成后再次检查超时；返回 `None` 时不发新目标。平滑系数 `smooth=0.3` 按有效新帧更新，采样率改变也会改变平滑的时间响应。输入中断后若需要清空平滑状态，可以为新会话重新创建推理器。

仿真输出要通过 [build_mapping](../mbadr/mjcf.py) 将 URDF 顺序转换为 MuJoCo qpos。真机输出要使用对应设备的驱动；停止发送新目标的行为见 [硬件说明](HARDWARE_VALIDATION.md)，它不等于设备已经制动或卸力。

## 4. 保存自己的训练数据

将适配后、预处理前的有效原始点按时间顺序保存为 `(N, 21, 3)` 的浮点 `.npy`。每个文件只存一种左右手；过滤缺点、非有限值和退化掌部，保留采集阶段的索引/单位转换规则。训练会再次调用统一预处理，不要保存已归一化的 MANO 点作为原始输入。

例如，`raw_frames` 是采集入口积累的有效数组列表：

```python
raw_frames = np.stack(raw_frames).astype(np.float32)
assert raw_frames.shape[1:] == (21, 3)
assert np.isfinite(raw_frames).all()
np.save("data/my_source_right.npy", raw_frames, allow_pickle=False)
```

若沿用现有标定配置：

```bash
python -m mbadr.train --robot l20 --hand right \
    --data-npy data/my_source_right.npy --steps 12000 \
    --out out/net_l20_my_source.pt

python -m mbadr.evaluate --robot l20 --hand right \
    --checkpoint out/net_l20_my_source.pt --data-npy data/my_source_right.npy \
    --limit 100 --out out/eval_l20_my_source.json
```

加载器会去除连续重复帧，随后默认按时间留出最后 15% 作为验证集；训练段至少需要 100 帧。收集张掌、握拳、逐指屈伸和捏合动作，并覆盖不同采集会话。数据格式相同只说明接口可接入，跨设备的精度仍需评估，必要时用新设备数据重新训练。

## 5. 使用输入源专属标定

若设备适配后仍需调整尺度或 `mediapipe_rotation`，可以复制配置：

```bash
mkdir -p configs/my_source
cp configs/pico4/pico4_linker_l20.yaml configs/my_source/my_source_linker_l20.yaml

# 修改新 YAML 中的标定值后训练
python -m mbadr.train --robot l20 --hand right \
    --config configs/my_source/my_source_linker_l20.yaml \
    --data-npy data/my_source_right.npy --steps 12000 \
    --out out/net_l20_my_source.pt
```

`default_config_path("l20", "my_source")` 能解析该路径，但 `KeypointRetargeter` 当前固定读取 `pico4` 配置，也不会自动根据 checkpoint 选择配置。自定义标定的网络推理可使用现有的底层 API，显式传入同一 YAML：

```python
from pathlib import Path

import numpy as np
import torch

from mbadr.data import preprocess_keypoints
from mbadr.evaluate import Evaluator

device = torch.device("cpu")
ev = Evaluator(
    Path("configs/my_source/my_source_linker_l20.yaml"),
    "right", Path("out/net_l20_my_source.pt"), device, "l20",
)
raw = np.load("data/my_source_right.npy", allow_pickle=False)[0]
kp = preprocess_keypoints(raw, ev.hand, ev.rotation, normalize_scale=True)
with torch.no_grad():
    x = ev.net_solve(torch.from_numpy(kp[None]).float().to(device))
    q = ev.fk.expand(x)[0].cpu().numpy()
print(dict(zip(ev.chain.joint_names, q.tolist())))
```

这里假定 `raw` 已通过采集层有效性检查；此底层示例不附带上节的无效帧拒绝、新鲜度检查和 EMA。若加平滑，应对 `x` 的独立关节平滑后再展开 mimic。

当前 `Evaluator` / `mbadr.evaluate` 的 NLopt 参考求解器仍固定读取 `pico4` 配置；上例只调用网络 `net_solve`。比较专属 YAML 下的网络与参考求解结果时，需用同一 YAML 的 `retarget` 构造 `AdaptiveConfig` 和 `ReferenceOptimizer`，并确保双方使用相同预处理；仅给评估命令加 `--config` 尚不能保证两侧目标一致。
