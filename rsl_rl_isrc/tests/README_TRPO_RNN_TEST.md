# TRPO RNN 算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_trpo_rnn_training.py` 是基于 rsl_rl 框架的 **TRPO + RNN** 训练测试，在 TRPO 基础上支持 LSTM/GRU 递归网络，用于处理具有时序依赖的连续控制任务。

## 文件说明

### `test_trpo_rnn_training.py`
TRPO RNN 版本训练测试，包含：
- Pendulum-v1 连续动作环境
- RNN (LSTM/GRU) 策略网络
- 隐藏状态管理与重置
- 适配 RNN 的 rollout 长度

## 主要特性

### 1. RNN 支持
- **rnn_hidden_size**: 设为 > 0 启用 RNN，0 则禁用
- **rnn_type**: `'lstm'` 或 `'gru'`
- **rnn_num_layers**: RNN 层数（默认 1）

### 2. 与标准 TRPO 的差异
| 项目 | 标准 TRPO | TRPO RNN |
|------|-----------|----------|
| rollout_length | 1024 | 512（节省内存） |
| vf_iters | 100 | 5 |
| 隐藏状态 | 无 | 需管理并定期重置 |
| 内存占用 | 较低 | 较高 |

### 3. 隐藏状态管理
- 每次 update 后调用 `policy.reset(dones=None)`
- 定期环境重置时同步重置 RNN 隐藏状态

## 环境

### Pendulum-v1
- 与 `test_trpo_training.py` 相同
- 3 维观察，1 维连续动作

## 训练流程

### 1. 初始化（启用 RNN）
```python
policy = TRPOPolicy(
    num_obs=3,
    num_actions=1,
    rnn_hidden_size=64,   # > 0 启用 RNN
    rnn_type='lstm',
    rnn_num_layers=1,
    vf_iters=5,           # RNN 时减少迭代
    ...
)
# rollout_length 自动设为 512
```

### 2. 训练循环
```python
for update in range(num_updates):
    # ... 收集 rollout ...
    policy.update()
    policy.algorithm.storage.clear()
    if rnn_hidden_size > 0:
        policy.reset(dones=None)   # 重置 RNN 隐藏状态
```

### 3. 返回结果
额外包含 `rnn_config` 字典，记录 RNN 配置。

## 运行方式

### 直接运行
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_trpo_rnn_training.py
```

### 主程序默认配置
```python
num_episodes = 1000
num_envs = 4          # RNN 版本减少环境数以节省内存
rnn_hidden_size = 64
rnn_type = 'lstm'
rnn_num_layers = 1
```

## 参数配置

### RNN 相关
```python
rnn_hidden_size = 64   # 0 禁用 RNN
rnn_type = 'lstm'      # 'lstm' | 'gru'
rnn_num_layers = 1
rollout_length = 512    # RNN 时自动缩短
vf_iters = 5           # RNN 时减少
```

### TRPO 共用参数
- `max_kl=0.05`, `damping=0.1`, `action_bounds=(-2.0, 2.0)` 等与标准 TRPO 一致

## 输出示例

```
=== TRPO RNN 训练测试 ===
使用RNN模式 - 隐藏大小: 64, 类型: lstm, 层数: 1
...
TRPO RNN更新 (RNN: lstm-64) - 策略损失: 0.0234, 价值损失: 15.67, ...
训练完成!
RNN配置: {'rnn_hidden_size': 64, 'rnn_type': 'lstm', 'rnn_num_layers': 1}
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl (TRPOPolicy with RNN)
- tqdm
- numpy

## 注意事项

1. RNN 版本内存占用更高，建议适当减少 `num_envs` 和 `rollout_length`
2. 隐藏状态需在 episode 结束或 update 后正确重置
3. 适合部分可观测或具有时序结构的任务
