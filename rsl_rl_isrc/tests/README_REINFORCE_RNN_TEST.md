# REINFORCE RNN 算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_reinforce_rnn_training.py` 是 **REINFORCE + RNN** 的完整训练测试，基于 rsl_rl 框架，使用带有 LSTM/GRU 的策略网络处理时序依赖，支持**离散动作空间**。

## 文件说明

### `test_reinforce_rnn_training.py`
REINFORCE RNN 训练测试，包含：
- CartPole-v1 多环境并行
- RNN (LSTM/GRU) 策略网络
- Off-policy episodes 存储
- 隐藏状态传递与管理

## 主要特性

### 1. RNN 模式
- **rnn_hidden_size > 0**: 启用 RNN；`= 0` 退化为标准 REINFORCE
- **rnn_type**: `'lstm'` 或 `'gru'`
- **rnn_num_layers**: RNN 层数

### 2. 与标准 REINFORCE 的差异
| 项目 | 标准 REINFORCE | REINFORCE RNN |
|------|----------------|---------------|
| 策略网络 | MLP | MLP + RNN |
| hidden_states | None | 每个 transition 保存 |
| 适用场景 | 马尔可夫 | 部分可观测/时序 |

### 3. 数据流
- 每个 transition 保存 `reinforce.get_hidden_states()`
- 使用 `RolloutStorage.add_off_policy_transition` 与 `finish_episode`
- 定期按 `min_episodes_for_update` 做策略更新

## 环境

### CartPole-v1
- 4 维观察，2 维离散动作
- 成功标准：`final_avg_return > 50`

## 训练流程

### 1. 初始化（启用 RNN）
```python
reinforce = REINFORCEPolicy(
    num_obs=4,
    num_actions=2,
    rnn_hidden_size=128,   # > 0 启用 RNN
    rnn_type='lstm',
    rnn_num_layers=1,
    ...
)
storage = RolloutStorage(num_envs=num_envs, num_transitions_per_env=500, ...)
```

### 2. 训练循环
```python
while total_episodes_collected < num_episodes:
    actions_batch, action_log_probs_batch = reinforce.act(state_tensor)
    # 每个 transition 保存 hidden_states
    transition.hidden_states = reinforce.get_hidden_states()
    storage.add_off_policy_transition(transition, env_idx)
    if dones[env_idx]:
        storage.finish_episode(env_idx, gamma=gamma)
    if len(storage.off_policy_episodes) >= min_episodes_for_update:
        for _ in range(num_policy_updates):
            loss = reinforce.update(storage)
```

### 3. 模型保存
- 文件名区分 RNN/标准：`reinforce_rnn_model.pt` / `reinforce_standard_model.pt`

## 运行方式

### 直接运行
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_reinforce_rnn_training.py
```

### 运行单元测试
```bash
python -m pytest rsl_rl/tests/test_reinforce_rnn_training.py::TestREINFORCETraining::test_basic_training -v
```

### 主程序默认配置
```python
num_episodes = 12000
num_envs = 200
hidden_dim = 128
rnn_hidden_size = 64
rnn_type = 'lstm'
rnn_num_layers = 1
```

## 测试内容

| 测试方法 | 说明 |
|---------|------|
| `test_environment` | CartPole-v1 环境接口 |
| `test_basic_training` | RNN 模式基本训练（rnn_hidden_size=64） |
| `test_training_convergence` | 奖励随训练提高（rnn_hidden_size=128） |

## 参数配置

### RNN 相关
```python
rnn_hidden_size = 128   # 0 禁用 RNN
rnn_type = 'lstm'       # 'lstm' | 'gru'
rnn_num_layers = 1
```

### 训练相关
```python
num_episodes = 200
num_envs = 4
hidden_dim = 32
learning_rate = 1e-3
gamma = 0.98
min_episodes_for_update = max(10, num_envs // 2)
num_policy_updates = 3
```

## 输出示例

```
REINFORCE RNN完整训练测试
使用RNN模式 - 隐藏大小: 64, 类型: lstm, 层数: 1
...
策略更新 - 平均损失: -12.345, 已收集episodes: 15, ...
训练完成!
RNN训练状态: 成功
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl (REINFORCEPolicy with RNN)
- tqdm
- numpy

## 注意事项

1. RNN 模式训练更慢，可适当减少 `num_envs` 或 `num_episodes` 做快速验证
2. 隐藏状态需在 `add_off_policy_transition` 时正确传入
3. 与 `test_reinforce_training.py` 共用 `REINFORCEPolicy`，通过 `rnn_hidden_size` 切换模式
