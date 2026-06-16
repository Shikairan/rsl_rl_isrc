# Training Mode ZeroMQ 数据格式说明

当前 training mode 通过 ZeroMQ `PUB/SUB` 收训练动作数据。渲染端是 `SUB`，会连接到指定端点；Python 训练端通常作为 `PUB`，负责绑定端点并发送消息。

每条消息是 UTF-8 文本，不是裸二进制 float：

```text
<标签> <JSON 数字数组>
```

示例：

```text
c1:0 [0.009362,0.000010,0.435048,0.000202,-0.023450]
```

标签和 JSON 数组之间必须有一个英文空格。接收端会用第一个空格切开消息，空格前是标签，空格后是 JSON。

## 启动渲染端

训练可视化入口是 `AlphaRayDemoView` 的 training mode：

```bash
./target/bin/Release/AlphaRayDemoView \
  --mode=training \
  --scene=workspace/scenes/robots/bipedal_robots/h1/scene1.xml \
  --endpoint=tcp://127.0.0.1:6006 \
  --num-envs=64 \
  --frame-width=19
```

重要参数：

- `--endpoint`：ZeroMQ 地址。渲染端会连接这个地址，所以 Python 发送端一般要 `bind` 这个地址。
- `--num-envs`：并行机器人数量。
- `--frame-width`：每个机器人的一帧有多少个数字，必须填写。
- `--labels`：可选。填写后按标签列表映射环境编号。

## 消息格式

一条 ZeroMQ 消息只发一个完整字符串，不使用 multipart：

```text
label + " " + json_array
```

JSON 必须是数组，数组里只能放数字：

```json
[0.009362, 0.000010, 0.435048, 0.000202]
```

不能发送下面这种裸二进制：

```python
socket.send(struct.pack("<19f", *frame))
```

当前 runtime 和 `tools/zmq_subscriber.c` 都会把收到的内容当文本解析，所以裸二进制 float 不会被正确识别。

## 标签怎么对应环境

如果启动时没有传 `--labels`，接收端会用标签里冒号后面的数字作为环境编号：

```text
c1:0 -> 第 0 个环境
c1:1 -> 第 1 个环境
c1:2 -> 第 2 个环境
```

如果启动时传了：

```bash
--labels=c1:0,c1:1,c1:2
```

则标签列表按顺序映射：

```text
c1:0 -> 第 0 个环境
c1:1 -> 第 1 个环境
c1:2 -> 第 2 个环境
```

标签里不要放空格。中文全角冒号 `：` 会被接收端转成英文冒号 `:`，但建议统一使用英文冒号。

## 发送形式

(1) 一个环境：

```text
c1:0 [frame_width 个数字]
```

这表示只更新 `c1:0` 对应的那个环境。

(2) 多个环境：

```text
c1:0 [第0个环境的frame_width个数字, 第1个环境的frame_width个数字, 第2个环境的frame_width个数字]
```

这表示从 `c1:0` 对应的环境开始，连续更新多个环境。

数组长度必须是 `frame_width` 的整数倍。比如 `frame_width=19`：

- 更新 1 个环境：数组长度是 `19`
- 连续更新 2 个环境：数组长度是 `38`
- 连续更新 64 个环境：数组长度是 `1216`

动作数字顺序和 `configs/kinematics_config` 文件一致：第一行是 `frame_width`，后面每 `frame_width` 个数字是一帧。

## Python 发送示例

安装依赖：

```bash
pip install pyzmq
```

最小发送示例：

```python
import json
import time

import zmq

endpoint = "tcp://127.0.0.1:6006"
label = "c1:0"
frame = [
    0.009362, 0.000010, 0.435048, 0.000202, -0.023450,
    0.000048, 0.999725, -0.008132, 0.809511, -1.361737,
    0.006774, 0.809511, -1.361737, -0.008132, 0.809511,
    -1.361737, 0.006774, 0.809511, -1.361737,
]

ctx = zmq.Context()
socket = ctx.socket(zmq.PUB)
socket.setsockopt(zmq.SNDHWM, 1)
socket.bind(endpoint)

# PUB/SUB 刚启动时需要给订阅端一点连接时间。
time.sleep(0.5)

while True:
    # (1) 一个环境：c1:0 [frame_width 个数字]
    payload = json.dumps(frame, separators=(",", ":"))
    socket.send_string(f"{label} {payload}")
    time.sleep(1.0 / 60.0)
```

一次发送多个环境：

```python
# (2) 多个环境：c1:0 [第0个环境, 第1个环境, 第2个环境]
frames = [
    frame_env_0,
    frame_env_1,
    frame_env_2,
]

flat = []
for frame in frames:
    flat.extend(frame)

payload = json.dumps(flat, separators=(",", ":"))
socket.send_string(f"c1:0 {payload}")
```

这里 `c1:0` 表示从第 0 个环境开始填，后面的数组会依次填到第 0、1、2 个环境。

## 用测试接收脚本检查

仓库里有一个 C 版接收脚本：

```bash
sudo apt install libzmq3-dev libjansson-dev
gcc tools/zmq_subscriber.c -o /tmp/zmq_subscriber -lzmq -ljansson -lm
```

运行：

```bash
/tmp/zmq_subscriber tcp://127.0.0.1:6006 c1:0
```

它会连接到 Python 发布端，订阅 `c1:0`，并打印：

- 原始消息
- JSON 数组部分
- 数组长度、平均值、标准差

如果这个脚本能看到类似下面的输出，说明发送格式是对的：

```text
c1:0 [0.009362,0.00001,...]
[0.009362,0.00001,...]
Received label: c1:0, data len:19, data avg:..., data std:...
```

## 常见问题

1. 渲染端没有动：先确认 `--frame-width` 和数组长度一致。
2. 收不到消息：确认一端 `bind`，另一端 `connect`；当前渲染端是 `connect`，Python 端建议 `bind`。
3. 标签不生效：确认标签没有空格，并且 `--labels` 列表和发送标签一致。
4. 旧脚本 `tools/send_training_frames.py` 当前发送的是二进制 float，不符合这里的文本 JSON 格式，不能直接作为 training mode 的现行格式参考。
