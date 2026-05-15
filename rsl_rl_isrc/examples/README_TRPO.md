# TRPO算法使用指南

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

本指南介绍如何在rsl_rl框架中使用新添加的TRPO (Trust Region Policy Optimization) 算法。

## 概述

TRPO是一种更稳定的policy gradient方法，通过trust region约束确保策略更新的单调改进，避免了传统policy gradient方法的剧烈震荡。

## 核心特性

- **Trust Region约束**: 使用KL散度限制策略更新的大小
- **共轭梯度法**: 高效求解trust region优化问题
- **线性搜索**: 确保约束满足的步长选择
- **GAE优势函数**: 使用Generalized Advantage Estimation提高稳定性
- **On-Policy学习**: 使用最新的策略收集数据进行更新

## 主要优势

相比传统PPO：
- **更强的收敛保证**: Trust region理论保证了策略改进
- **更稳定的训练**: 避免了大的策略更新导致的性能崩溃
- **更精确的优化**: 直接优化真实的目标函数而非近似

## 文件结构

新增的文件：
```
rsl_rl_isrc/
├── algorithms/
│   ├── trpo.py              # TRPO算法实现
│   └── __init__.py          # 添加TRPO导入
├── runners/
│   ├── trpo_runner.py       # TRPO训练runner
│   └── __init__.py          # 添加TRPORunner导入
├── storage/
│   └── rollout_storage.py   # 支持GAE计算
└── examples/
    ├── trpo_example.py      # 使用示例
    └── README_TRPO.md       # 本文档
```

## 基本使用

### 1. 配置设置

```python
config = {
    "algorithm": {
        "algorithm_class_name": "TRPO",          # 指定使用TRPO算法
        "num_learning_epochs": 1,                # 每次更新的epoch数
        "learning_rate": 1e-2,                   # 价值函数学习率
        "gamma": 0.99,                           # 折扣因子
        "lam": 0.95,                             # GAE参数 (0.9-0.99)
        "kl_constraint": 0.0005,                 # KL散度约束 (0.0001-0.01)
        "alpha": 0.5                             # 线性搜索参数 (0.1-0.9)
    },
    "policy": {
        "policy_class_name": "ActorCritic",     # 使用ActorCritic
        "actor_hidden_dims": [64, 64],          # actor网络结构
        "critic_hidden_dims": [64, 64],         # critic网络结构
        "activation": "elu",
        "init_noise_std": 1.0
    },
    "runner": {
        "experiment_name": "trpo_experiment",
        "num_steps_per_env": 100,               # 每个环境收集的步数
        "save_interval": 50
    }
}
```

### 2. 初始化

```python
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.algorithms import TRPO
from rsl_rl_isrc.runners import TRPORunner

# 创建策略和价值网络
actor_critic = ActorCritic(
    num_obs=env.num_obs,
    num_critic_obs=env.num_obs,
    num_actions=env.num_actions,
    **config["policy"]
).to(device)

# 创建TRPO runner
runner = TRPORunner(
    env=env,
    train_cfg=config,
    log_dir=log_dir,
    device=device
)
```

### 3. 训练

```python
# 开始训练
runner.learn(num_learning_iterations=1000)
```

## 核心API

### TRPO类

```python
class TRPO:
    def __init__(self, actor_critic, learning_rate=1e-2, gamma=0.99, lam=0.95,
                 kl_constraint=0.0005, alpha=0.5, device='cpu', **kwargs)

    def act(self, observations)  # 采样动作
    def update(self, storage)    # 更新策略和价值函数
    def save(self, path)         # 保存模型
    def load(self, path)         # 加载模型

    # 核心TRPO方法
    def hessian_matrix_vector_product(self, states, old_action_dists, vector)
    def conjugate_gradient(self, grad, states, old_action_dists)
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec)
    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage)
    def compute_advantage(self, gamma, lam, td_delta)
```

## 算法详解

### Trust Region优化

TRPO的核心思想是约束策略更新的大小，确保每次更新都能改进性能：

```
maximize: E[π_θ(a|s)] / E[π_θ_old(a|s)]
subject to: D_KL(π_θ_old || π_θ) ≤ δ
```

### 关键组件

1. **共轭梯度法**: 求解trust region子问题
   - 计算海森矩阵向量积 Hx
   - 迭代求解 Hx = g

2. **线性搜索**: 确保KL约束满足
   - 从大步长开始尝试
   - 找到满足约束的最大步长

3. **GAE优势函数**: 降低方差，提高稳定性
   ```
   A_t = δ_t + γλ δ_{t+1} + ... + (γλ)^{T-t-1} δ_{T-1}
   ```

### 超参数调优

- **kl_constraint**: KL散度限制
  - 小: 更保守的更新，收敛慢但稳定
  - 大: 更激进的更新，收敛快但可能不稳定
  - 推荐: 0.0001-0.01

- **lam**: GAE参数
  - 控制bias-variance tradeoff
  - 推荐: 0.9-0.99

- **alpha**: 线性搜索衰减因子
  - 推荐: 0.1-0.9

## 训练流程

1. **数据收集**: 使用当前策略与环境交互
2. **优势计算**: 使用GAE计算优势函数
3. **价值更新**: 标准MSE损失更新critic
4. **策略更新**: TRPO trust region优化更新actor
5. **参数同步**: 分布式环境下的参数广播

## 与PPO对比

| 特性 | TRPO | PPO |
|------|------|-----|
| 理论保证 | 有 | 无 |
| 计算复杂度 | 高 | 低 |
| 收敛速度 | 慢 | 快 |
| 稳定性 | 高 | 中 |
| 实现复杂度 | 高 | 低 |

## 注意事项

1. **计算开销**: TRPO需要计算海森矩阵，计算量较大
2. **内存使用**: 存储完整的trajectory数据
3. **超参数敏感**: 对KL约束等超参数较为敏感
4. **收敛保证**: 理论上有收敛保证，但实践中仍需调参

## 故障排除

1. **训练不稳定**:
   - 降低`kl_constraint`
   - 增加`lam`值
   - 减少学习率

2. **收敛太慢**:
   - 适当增加`kl_constraint`
   - 调整`alpha`参数
   - 增加训练步数

3. **性能下降**:
   - 检查KL约束是否过大
   - 验证优势函数计算
   - 检查critic学习是否正常

## 示例代码

查看 `trpo_example.py` 获取完整的使用示例。

## 扩展

TRPO可以扩展为：
- **ACKTR**: 使用Kronecker-factored Approximate Curvature进行加速
- **PPO**: 使用clipping近似trust region约束
- **Natural PG**: 使用自然梯度而非普通梯度

这些扩展提供了不同的trade-off，在计算效率和收敛保证之间平衡。
