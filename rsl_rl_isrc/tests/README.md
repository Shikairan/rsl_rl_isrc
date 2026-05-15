# rsl_rl 测试目录

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

本目录包含 rsl_rl 框架下各强化学习算法的训练测试脚本及对应说明。

## 文件索引

### 直接算法测试（使用 gymnasium 环境）

| 测试文件 | 对应 README | 算法 | 动作空间 | 环境 |
|----------|-------------|------|----------|------|
| `test_reinforce_training.py` | [README_REINFORCE_TEST.md](README_REINFORCE_TEST.md) | REINFORCE | 离散 | CartPole-v1 |
| `test_reinforce_rnn_training.py` | [README_REINFORCE_RNN_TEST.md](README_REINFORCE_RNN_TEST.md) | REINFORCE + RNN | 离散 | CartPole-v1 |
| `test_ppo_training.py` | [README_PPO_TEST.md](README_PPO_TEST.md) | PPO | 离散 | CartPole-v1 |
| `test_trpo_training.py` | [README_TRPO_TEST.md](README_TRPO_TEST.md) | TRPO | 连续 | Pendulum-v1 |
| `test_trpo_rnn_training.py` | [README_TRPO_RNN_TEST.md](README_TRPO_RNN_TEST.md) | TRPO + RNN | 连续 | Pendulum-v1 |
| `test_sac_training.py` | [README_SAC_TEST.md](README_SAC_TEST.md) | SAC | 连续 | Pendulum-v1 |

### Runner 测试（使用 VecEnv 接口）

| 测试文件 | 算法 | 动作空间 | 环境 | 说明 |
|----------|------|----------|------|------|
| `test_ppo_runner.py` | PPO + OnPolicyRunner | 离散 | DummyVecEnv (CartPole-like) | 使用正确的 PPO 类和 Runner 接口 |
| `test_trpo_runner.py` | TRPO + TRPORunner | 连续 | DummyVecEnv (Pendulum-like) | 使用 TRPOPolicy 和 Runner 接口 |
| `test_sac_runner.py` | SAC + SACRunner | 连续 | DummyVecEnv (Pendulum-like) | 使用 SAC 和 Runner 接口 |
| `test_reinforce_runner.py` | REINFORCE + REINFORCERunner | 离散 | DummyVecEnv (CartPole-like) | 使用 REINFORCEPolicy 和 Runner 接口 |

## 快速运行

### 直接算法测试（使用 gymnasium 环境）
```bash
# 进入项目根目录
cd /path/to/rsl_rl_withALL

# 运行任意直接算法测试
python rsl_rl_isrc/tests/test_ppo_training.py
python rsl_rl_isrc/tests/test_sac_training.py
python rsl_rl_isrc/tests/test_trpo_training.py
python rsl_rl_isrc/tests/test_reinforce_training.py
```

### Runner 测试（使用 VecEnv 接口）
```bash
# 运行使用 Runner 的测试（推荐）
python rsl_rl_isrc/tests/test_ppo_runner.py         # PPO + OnPolicyRunner ✓
python rsl_rl_isrc/tests/test_trpo_runner.py        # TRPO + TRPORunner
python rsl_rl_isrc/tests/test_sac_runner.py         # SAC + SACRunner
python rsl_rl_isrc/tests/test_reinforce_runner.py   # REINFORCE + REINFORCERunner
```

### 单元测试
```bash
# 运行所有测试的单元测试
python -m pytest rsl_rl_isrc/tests/ -v

# 运行特定测试
python -m pytest rsl_rl_isrc/tests/test_ppo_runner.py::TestPPORunner::test_basic_runner_training -v
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl_isrc
- tqdm
- numpy

详细说明请参阅各测试对应的 README 文件。
