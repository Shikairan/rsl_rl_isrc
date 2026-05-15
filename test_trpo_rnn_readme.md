# TRPO RNN Training Test

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

这个文件 `test_trpo_rnn_training.py` 是用于测试 TRPO 算法中 RNN 功能的文件。

## 功能特性

- 支持标准的 TRPO 训练（无 RNN）
- 支持 RNN 版本的 TRPO 训练（LSTM/GRU）
- 自动检测 RNN 配置并相应调整网络结构
- 包含完整的训练循环和性能评估

## 使用方法

### 基本使用

```python
from rsl_rl_isrc.tests.test_trpo_rnn_training import run_trpo_training

# 运行标准 TRPO 训练
results = run_trpo_training(
    num_episodes=100,
    num_envs=4,
    rnn_hidden_size=0,  # 0 = 不使用 RNN
    device='cpu'
)

# 运行 RNN TRPO 训练
results = run_trpo_training(
    num_episodes=100,
    num_envs=4,
    rnn_hidden_size=64,  # > 0 = 启用 RNN
    rnn_type='lstm',     # 'lstm' 或 'gru'
    rnn_num_layers=1,    # RNN 层数
    device='cpu'
)
```

### 测试类方法

```python
from rsl_rl_isrc.tests.test_trpo_rnn_training import TestTRPORNNTraining

test_class = TestTRPORNNTraining()

# 测试基本 RNN 功能
test_class.test_basic_rnn_training()

# 测试训练收敛性
test_class.test_training_convergence()

# 测试连续动作空间
test_class.test_continuous_actions()
```

### 命令行运行

```bash
cd /home/data/rl/ppo/pymotrisim_3_mujoco/rsl_rl_withALL
python -m rsl_rl_isrc.tests.test_trpo_rnn_training
```

## RNN 参数说明

- `rnn_hidden_size`: RNN 隐藏层大小，设置为 0 禁用 RNN，大于 0 启用 RNN
- `rnn_type`: RNN 类型，支持 'lstm' 和 'gru'
- `rnn_num_layers`: RNN 层数，通常设置为 1 或 2

## 输出结果

训练完成后会返回包含以下信息的字典：

- `episode_rewards`: 每个 episode 的奖励列表
- `episode_lengths`: 每个 episode 的长度列表
- `final_avg_reward`: 最后 100 个 episodes 的平均奖励
- `final_avg_length`: 最后 100 个 episodes 的平均长度
- `reward_moving_avg`: 奖励的移动平均
- `success`: 是否达到成功标准（Pendulum 环境 > -200）
- `rnn_config`: RNN 配置信息

## 注意事项

1. RNN 版本会自动处理隐藏状态的管理
2. 在 episode 结束时会自动重置 RNN 隐藏状态
3. 使用 Pendulum-v1 环境进行测试（连续动作空间）
4. RNN 版本可能需要更多计算资源和训练时间