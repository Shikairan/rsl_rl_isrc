# REINFORCE算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_reinforce_training.py` 是一个完整的REINFORCE算法训练测试文件，基于`hand-on-rl`目录中的经典实现，但完全适配rsl_rl框架。

## 文件说明

### `test_reinforce_training.py`
完整的REINFORCE算法训练测试，包含：
- 模拟CartPole环境 (`CartPoleLikeEnv`)
- 完整的训练循环
- 结果验证和统计
- 单元测试

## 主要特性

### 1. Headless设计
- 不依赖图形界面，完全可以在服务器环境运行
- 使用自定义的`CartPoleLikeEnv`模拟环境
- 避免了外部gym依赖

### 2. 完整训练流程
基于`hand-on-rl/REINFORCE.py`的训练逻辑：
- 分批次训练 (10个iteration)
- 每个iteration包含多个episode
- 实时进度显示和统计
- 移动平均计算

### 3. 适配rsl_rl框架
- 使用`ActorCritic`模块
- 利用`RolloutStorage`的off-policy功能
- 集成`REINFORCE`算法类

## 环境模拟

### CartPoleLikeEnv
简化的CartPole环境模拟：
```python
class CartPoleLikeEnv:
    def __init__(self, max_steps=200):
        # 4维状态: [cart_pos, cart_vel, pole_angle, pole_vel]
        # 2维动作: [left, right]
```

### 物理模型
大幅简化的CartPole物理：
- 力学更新
- 角度和位置约束
- 奖励函数（存活奖励+失败惩罚）

## 训练流程

### 1. 初始化
```python
env = CartPoleLikeEnv(max_steps=200)
actor_critic = ActorCritic(num_obs=4, num_critic_obs=4, num_actions=2, ...)
reinforce = REINFORCE(actor_critic, learning_rate=1e-3, gamma=0.98)
storage = RolloutStorage(num_envs=1, num_transitions_per_env=200, ...)
```

### 2. 训练循环
```python
for iteration in range(10):  # 10个大迭代
    for episode in range(num_episodes // 10):  # 每个迭代的episode
        # 收集一个episode的数据
        while not done:
            action, log_prob = reinforce.act(state)
            next_state, reward, done = env.step(action)
            # 存储transition
            storage.add_off_policy_transition(transition, 0)

        # 完成episode并计算returns
        storage.finish_episode(0, gamma=0.98)

        # 更新策略
        loss = reinforce.update(storage)
```

### 3. 结果统计
- 实时奖励和长度统计
- 移动平均计算
- 训练效果验证

## 运行方式

### 直接运行训练
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_reinforce_training.py
```

### 运行单元测试
```bash
cd /path/to/rsl_rl
python -m pytest rsl_rl/tests/test_reinforce_training.py::TestREINFORCETraining::test_basic_training -v
```

### 运行完整测试套件
```bash
cd /path/to/rsl_rl
python -m pytest rsl_rl/tests/test_reinforce_training.py -v
```

## 测试内容

### 1. 基础功能测试
- ✅ 环境初始化和重置
- ✅ 动作执行和状态更新
- ✅ 奖励计算和终止条件

### 2. 训练功能测试
- ✅ REINFORCE算法初始化
- ✅ Episode数据收集
- ✅ 策略更新和损失计算
- ✅ 训练统计和验证

### 3. 收敛测试
- ✅ 多轮训练效果
- ✅ 奖励提高趋势
- ✅ 超参数敏感性

## 参数配置

### 训练参数
```python
num_episodes = 100     # 总训练轮数
hidden_dim = 32        # 网络隐藏层维度
learning_rate = 1e-3   # 学习率
gamma = 0.98          # 折扣因子
```

### 环境参数
```python
max_steps = 200       # 最大episode长度
state_dim = 4         # 状态维度
action_dim = 2        # 动作维度
```

## 预期结果

### 训练输出示例
```
开始REINFORCE训练测试...
设备: cpu
训练轮数: 100
隐藏层维度: 32
学习率: 0.001
折扣因子: 0.98

迭代 1: 100%|██████████| 10/10 [00:01<00:00,  8.45it/s, episode=10, return=45.200, length=45.2, loss=-12.345]
迭代 1 完成 - 平均奖励: 45.200, 平均长度: 45.2

训练完成!
总轮数: 100
最终平均奖励: 78.500
最终平均长度: 78.5
训练成功: True
```

### 成功标准
- 平均奖励 > 50（超过随机策略）
- 奖励呈上升趋势
- 无NaN或Inf值
- 训练过程稳定

## 与hand-on-rl对比

| 特性 | hand-on-rl REINFORCE | rsl_rl测试版本 |
|------|---------------------|----------------|
| 环境 | CartPole-v1 (gym) | CartPoleLikeEnv (模拟) |
| 界面 | matplotlib绘图 | headless控制台输出 |
| 框架 | 原生PyTorch | rsl_rl ActorCritic模块 |
| 存储 | 字典存储 | RolloutStorage off-policy |
| 测试 | 手动验证 | 自动化单元测试 |

## 故障排除

### 1. 训练不收敛
```python
# 尝试降低学习率
learning_rate = 5e-4

# 或者增加折扣因子
gamma = 0.99
```

### 2. 奖励为0
- 检查环境终止条件
- 验证奖励计算逻辑
- 确认动作空间映射

### 3. 内存不足
```python
# 减少存储的episodes
if len(storage.off_policy_episodes) > 20:
    storage.clear_off_policy_episodes()
```

### 4. 训练过慢
- 减少网络隐藏层维度
- 缩短episode最大长度
- 使用更简单的环境

## 扩展

### 添加更多环境
```python
class MountainCarLikeEnv:
    """MountainCar环境模拟"""
    # 实现MountainCar的简化版本

class PendulumLikeEnv:
    """连续控制任务"""
    # 实现连续动作空间
```

### 添加基准测试
```python
def benchmark_algorithms():
    """比较不同算法的性能"""
    algorithms = ['reinforce', 'ppo', 'trpo']
    results = {}

    for alg in algorithms:
        results[alg] = run_training(alg, env)
        plot_comparison(results)
```

这个测试文件提供了一个完整的、headless的REINFORCE算法验证环境，可以在任何安装了PyTorch的环境中运行，无需图形界面支持。
