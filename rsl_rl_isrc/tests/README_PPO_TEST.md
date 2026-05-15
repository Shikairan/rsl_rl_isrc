# PPO 算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_ppo_training.py` 是基于 rsl_rl 框架的 **PPO (Proximal Policy Optimization)** 算法完整训练测试，使用 clip loss 而非 TRPO 的 KL 约束，支持**离散动作空间**。

## 文件说明

### `test_ppo_training.py`
完整的 PPO 算法训练测试，包含：
- CartPole-v1 多环境并行训练
- Clip loss + GAE (Generalized Advantage Estimation)
- 离散动作空间支持
- 结果验证和单元测试

## 主要特性

### 1. 真正的 PPO 实现
- 使用 `PPOPolicy` 算法类
- Clip loss 限制策略更新幅度
- GAE 计算优势函数
- 标准 PPO 超参数配置

### 2. 多环境并行
- `AsyncVectorEnv` 并行采集
- 支持 configurable `num_envs`
- 提高样本采集效率

### 3. 离散动作空间
- CartPole-v1: 2 维离散动作（左/右）
- One-hot 编码动作表示
- 适配 rsl_rl 的 ActorCritic 输出

## 环境

### CartPole-v1 (gymnasium)
- **观察空间**: 4 维 `[cart_pos, cart_vel, pole_angle, pole_vel]`
- **动作空间**: 离散 2 维 `{0: 左, 1: 右}`
- **解决标准**: 平均回报 > 195

## 训练流程

### 1. 初始化
```python
env = gym.vector.AsyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)])
ppo = PPOPolicy(num_obs=4, num_actions=2, learning_rate=3e-4, gamma=0.99, ...)
ppo.init_storage(num_envs=num_envs, num_transitions_per_env=2048, ...)
```

### 2. 训练循环
```python
while total_episodes_collected < num_episodes:
    # 采样动作
    actions_batch, action_log_probs_batch, values_batch = ppo.act(state_tensor)
    action_indices = actions_batch.argmax(dim=-1).cpu().numpy()

    # 执行环境步骤
    next_states, rewards, terminated, truncated, infos = env.step(action_indices)

    # 添加 transition 到 storage
    ppo.storage.add_transitions(transition)

    # 收集足够后更新
    if ppo.storage.step >= num_transitions_per_env:
        ppo.storage.compute_returns(last_values, gamma, lam)
        losses = ppo.update()
```

### 3. 结果统计
- 实时回报和 episode 长度
- 移动平均计算
- 成功标准：`final_avg_return > 195`

## 运行方式

### 直接运行训练
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_ppo_training.py
```

### 运行单元测试
```bash
python -m pytest rsl_rl/tests/test_ppo_training.py::TestPPOTraining::test_basic_training -v
```

### 运行完整测试套件
```bash
python -m pytest rsl_rl/tests/test_ppo_training.py -v
```

## 测试内容

| 测试方法 | 说明 |
|---------|------|
| `test_environment` | 验证 CartPole-v1 环境接口 |
| `test_basic_training` | 基本训练流程（20 episodes） |
| `test_training_convergence` | 检查奖励是否随训练提高 |

## 参数配置

### 训练参数（主程序默认）
```python
num_episodes = 2000
num_envs = 16
hidden_dim = 64
learning_rate = 3e-4
gamma = 0.99
num_transitions_per_env = 2048
```

### 快速测试参数
```python
num_episodes = 20
num_envs = 2
hidden_dim = 32
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl (本地版本)
- tqdm
- numpy

## 预期输出

```
PPO完整训练测试
设备: cuda
训练轮数: 2000
环境数量: 16
...
策略更新 - 值损失: 0.1234, 代理损失: -0.0567, 熵损失: 0.0234, ...
训练完成!
最终平均奖励: 198.500
训练成功: True
```
