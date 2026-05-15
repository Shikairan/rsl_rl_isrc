# TRPO 算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_trpo_training.py` 是基于 rsl_rl 框架的 **TRPO (Trust Region Policy Optimization)** 算法训练测试，使用 KL 约束和共轭梯度，支持**连续动作空间**。

## 文件说明

### `test_trpo_training.py`
完整的 TRPO 算法训练测试，包含：
- Pendulum-v1 连续控制环境
- KL 约束策略更新
- Fisher 信息矩阵 + 共轭梯度
- 固定 rollout 长度训练

## 主要特性

### 1. TRPO 核心机制
- **KL 约束**: `max_kl=0.05` 限制策略变化
- **共轭梯度**: 近似自然梯度
- **Fisher 阻尼**: `damping=0.1` 提高数值稳定性
- **价值函数**: 独立训练，`vf_iters=100`

### 2. 连续动作空间
- Pendulum-v1: 3 维观察，1 维连续动作
- 动作边界：`[-2.0, 2.0]`
- 通过 `action_bounds` 传入算法

### 3. Rollout 收集
- 固定 rollout 长度 1024
- 跨 episode 边界收集
- 收集满后计算 returns 并更新

## 环境

### Pendulum-v1 (gymnasium)
- **观察空间**: 3 维 `[cos(θ), sin(θ), θ̇]`
- **动作空间**: 连续 1 维，范围 `[-2.0, 2.0]`
- **合理目标**: 平均回报 > -200

## 训练流程

### 1. 初始化
```python
env = gym.vector.AsyncVectorEnv([lambda: gym.make('Pendulum-v1') for _ in range(num_envs)])
action_bounds = (float(action_space.low[0]), float(action_space.high[0]))
policy = TRPOPolicy(num_obs=3, num_actions=1, max_kl=0.05, action_bounds=action_bounds, ...)
policy.init_storage(num_envs=num_envs, num_transitions_per_env=1024, ...)
```

### 2. 训练循环
```python
for update in range(num_updates):
    for step in range(rollout_length):
        actions = policy.act(obs)
        next_obs, rewards, dones, truncateds, infos = env.step(actions.cpu().numpy())
        policy.process_env_step(rewards, dones, infos, scale_factor=1.0)
        obs = next_obs

    policy.compute_returns(torch.tensor(obs, ...))
    value_loss, policy_loss = policy.update()
    policy.algorithm.storage.clear()
```

### 3. 成功标准
- `final_avg_reward > -200` 视为基本成功

## 运行方式

### 直接运行
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_trpo_training.py
```

### 主程序默认配置
```python
num_episodes = 100000
num_envs = 32
device = 'cuda' if available else 'cpu'
```

## 参数配置

### TRPO 关键参数
```python
max_kl = 0.05       # KL 散度约束
damping = 0.1       # Fisher 矩阵阻尼
vf_lr = 1e-2        # 价值函数学习率
vf_iters = 100      # 价值函数迭代次数
tau = 0.98          # GAE 时间窗口
rollout_length = 1024
```

### 环境相关
```python
action_bounds = (-2.0, 2.0)  # Pendulum-v1
```

## 输出示例

```
=== TRPO 训练测试 ===
使用设备: cuda
开始TRPO训练测试...
观察空间维度: 3
动作空间维度: 1
...
TRPO更新 - 策略损失: 0.0123, 价值损失: 12.34, 已收集episodes: 48, ...
训练完成!
最终平均奖励: -156.234
训练成功: True
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl (TRPOPolicy)
- tqdm
- numpy

## 注意事项

1. TRPO 计算开销较大，建议使用 GPU
2. `script_dir` 和 `project_root` 为硬编码路径，迁移时需调整
3. 定期重置环境（每 10 次 update）防止 episode 过长
