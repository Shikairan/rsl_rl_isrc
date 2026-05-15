# Runner 测试说明

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

本目录包含使用 rsl_rl Runner 接口的强化学习算法测试。这些测试使用统一的 `VecEnv` 接口和配置格式，与实际训练环境保持一致。

## Runner 测试文件

| 测试文件 | Runner 类 | 算法 | 环境类型 | 说明 |
|----------|-----------|------|----------|------|
| `test_ppo_runner.py` | `OnPolicyRunner` | PPO | DummyVecEnv (CartPole-like) | 使用正确的 PPO 类和完整 Runner 接口 |
| `test_trpo_runner.py` | `TRPORunner` | TRPO | DummyVecEnv (Pendulum-like) | 使用 TRPOPolicy 和 TRPO 专用 Runner |
| `test_sac_runner.py` | `SACRunner` | SAC | DummyVecEnv (Pendulum-like) | 使用 SAC 算法和 Off-policy Runner |
| `test_reinforce_runner.py` | `REINFORCERunner` | REINFORCE | DummyVecEnv (CartPole-like) | 使用 REINFORCEPolicy 和专用 Runner |

## 主要特性

### 1. 统一的接口
所有 Runner 测试都使用相同的接口模式：
```python
train_cfg = {
    "runner": {...},     # Runner 配置
    "algorithm": {...},  # 算法超参数
    "policy": {...}      # 网络架构配置
}

runner = RunnerClass(env=env, train_cfg=train_cfg, ...)
runner.learn(num_learning_iterations=100)
```

### 2. 虚拟环境
所有测试都使用简化的 `DummyVecEnv`：
- **CartPole-like**: 4 维观测，2 维离散动作
- **Pendulum-like**: 3 维观测，1 维连续动作
- 实现了完整的 `VecEnv` 接口
- 支持批量环境并行

### 3. 完整训练流程
每个测试包含：
- 环境初始化和配置
- Runner 创建和训练
- 结果评估和验证
- 单元测试用例

## 配置格式

### PPO Runner 配置
```python
train_cfg = {
    "runner": {
        "experiment_name": "ppo_cartpole",
        "num_steps_per_env": 100,
        "save_interval": 50
    },
    "algorithm": {
        "algorithm_class_name": "PPO",
        "num_learning_epochs": 4,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 3e-4
    },
    "policy": {
        "policy_class_name": "ActorCritic",
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "activation": "elu"
    }
}
```

### TRPO Runner 配置
```python
train_cfg = {
    "runner": {
        "experiment_name": "trpo_pendulum",
        "num_steps_per_env": 512,
        "save_interval": 50
    },
    "algorithm": {
        "max_kl": 0.05,
        "damping": 0.1,
        "vf_lr": 1e-2,
        "vf_iters": 20,
        "action_bounds": [-2.0, 2.0]
    },
    "policy": {
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "gamma": 0.99,
        "tau": 0.97
    }
}
```

### SAC Runner 配置
```python
train_cfg = {
    "runner": {
        "experiment_name": "sac_pendulum",
        "num_steps_per_env": 1,
        "save_interval": 50
    },
    "algorithm": {
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 256,
        "learning_starts": 1000,
        "update_frequency": 64,
        "policy_lr": 3e-4,
        "q_lr": 1e-3,
        "action_bounds": [-2.0, 2.0]
    },
    "policy": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "activation": "relu"
    }
}
```

### REINFORCE Runner 配置
```python
train_cfg = {
    "runner": {
        "experiment_name": "reinforce_cartpole",
        "num_steps_per_env": 100,
        "save_interval": 50
    },
    "algorithm": {
        "learning_rate": 1e-3,
        "gamma": 0.98,
        "num_learning_epochs": 1
    },
    "policy": {
        "action_space_type": "discrete",
        "hidden_dims": [64, 64],
        "activation": "tanh"
    }
}
```

## 运行方式

### 基本运行
```bash
cd /path/to/rsl_rl_withALL

# 运行单个测试
python rsl_rl_isrc/tests/test_ppo_runner.py
python rsl_rl_isrc/tests/test_trpo_runner.py
python rsl_rl_isrc/tests/test_sac_runner.py
python rsl_rl_isrc/tests/test_reinforce_runner.py
```

### 单元测试
```bash
# 运行所有 runner 测试的单元测试
python -m pytest rsl_rl_isrc/tests/test_*_runner.py -v
```

### 自定义配置
```python
from rsl_rl_isrc.runners import OnPolicyRunner

# 自定义配置
custom_cfg = {
    "runner": {"num_steps_per_env": 50},
    "algorithm": {"learning_rate": 1e-4},
    "policy": {"hidden_dims": [32, 32]}
}

runner = OnPolicyRunner(env=env, train_cfg=custom_cfg)
runner.learn(num_learning_iterations=10)
```

## 与直接算法测试的区别

| 特性 | 直接算法测试 | Runner 测试 |
|------|--------------|-------------|
| 环境接口 | gymnasium | VecEnv (rsl_rl) |
| 配置方式 | 函数参数 | 字典配置 |
| 分布式支持 | 无 | 完整支持 |
| 存储管理 | 手动 | 自动 |
| 日志系统 | 基础 | TensorBoard + 完整 |
| 模型保存 | 基础 | 完整检查点 |

## 虚拟环境说明

### DummyVecEnv 实现
- 继承自 `VecEnv` 抽象基类
- 实现了所有必需的方法：`step()`, `reset()`, `get_observations()`, `get_privileged_observations()`
- 支持批量操作和 episode 管理
- 简化的物理模拟（适合快速测试）

### 环境参数
- **CartPole-like**: 购物车杆平衡任务
  - 观测: `[cart_pos, cart_vel, pole_angle, pole_vel]`
  - 动作: 离散 2 维 (左推/右推)
  - 奖励: 存活奖励
- **Pendulum-like**: 摆控制任务
  - 观测: `[cosθ, sinθ, θ̇]`
  - 动作: 连续 1 维 `[-2, 2]`
  - 奖励: 角度惩罚

## 测试验证

每个 Runner 测试都包含：
1. **环境测试**: 验证 DummyVecEnv 的基本功能
2. **训练测试**: 验证完整的训练流程
3. **结果验证**: 检查训练结果的合理性
4. **单元测试**: 使用 unittest 框架

## 性能基准

| 算法 | 环境 | 目标性能 | 典型训练时间 |
|------|------|----------|--------------|
| PPO | CartPole | 奖励 > 50 | 100 次迭代 |
| TRPO | Pendulum | 奖励 > -500 | 50 次迭代 |
| SAC | Pendulum | 奖励 > -500 | 200 次迭代 |
| REINFORCE | CartPole | 奖励 > 30 | 200 次迭代 |

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```python
   # 减少环境数量或批次大小
   train_cfg["runner"]["num_steps_per_env"] = 50
   ```

2. **训练不收敛**
   ```python
   # 调整学习率
   train_cfg["algorithm"]["learning_rate"] = 1e-4
   ```

3. **环境重置问题**
   ```python
   # 检查 DummyVecEnv 的实现
   env = DummyVecEnv(num_envs=2)
   obs = env.reset(torch.arange(2))
   ```

## 扩展使用

这些 Runner 测试可以作为模板，用于：
- 测试新的算法实现
- 验证环境接口兼容性
- 基准测试性能
- 快速原型开发

所有测试都可以在有 PyTorch 的环境中运行，无需额外的机器人仿真依赖。