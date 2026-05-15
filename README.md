# rsl_rl_isrc

基于 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 思路的 **PyTorch** 强化学习组件库，在独立包名 `rsl_rl_isrc` 下由 **ISRC** 维护与扩展。提供 **PPO、TRPO、REINFORCE、SAC** 等算法、`RolloutStorage` / `ReplayBuffer`、训练 **Runner**，以及可选的 **HTTP 遥测**（仿真张量与步后观测上报）。

## 致谢

- 感谢 rsl_rl 原团队（Nikita Rudin / ETH Zurich 等）的开源基础。  
- 本仓库扩展与工程化由 **ISRC** 完成；许可证为 **BSD-3-Clause**（见各源文件头与 `setup.py`）。

## 功能概览

| 模块 | 说明 |
|------|------|
| `rsl_rl_isrc.algorithms` | `PPO`、`TRPO`、`REINFORCEPolicy`、`TRPOPolicy`、`SAC` 等 |
| `rsl_rl_isrc.runners` | `OnPolicyRunner`、`TRPORunner`、`REINFORCERunner`、`SACRunner` |
| `rsl_rl_isrc.modules` | Actor-Critic（含 RNN）、TRPO 网络、SAC 网络、单 Actor 等 |
| `rsl_rl_isrc.storage` | `RolloutStorage`、`ReplayBuffer` |
| `rsl_rl_isrc.env` | 向量环境抽象 `VecEnv` |
| `rsl_rl_isrc.sockets` | `send_post_request`（仿真数据 POST）、`StepObsPublisher`（按服务端指令切片上报观测） |
| `rsl_rl_isrc.utils` | 轨迹切分/补全、TRPO 辅助（共轭梯度、参数拉平等）、`RunningMeanStd` |
| `rsl_rl_isrc.examples` | TRPO / REINFORCE 示例脚本与说明 |

## 环境要求

- **Python** ≥ 3.8  
- **PyTorch**、**NumPy** 等见 [`setup.py`](setup.py) 中 `install_requires`  
- 运行带 **gymnasium** 的测试：`pip install -e ".[dev]"`

## 安装

在项目根目录（含 `setup.py` 的目录）执行：

```bash
pip install -e .
```

开发/测试依赖：

```bash
pip install -e ".[dev]"
```

## 快速使用

```python
from rsl_rl_isrc.runners import OnPolicyRunner, TRPORunner, SACRunner, REINFORCERunner
from rsl_rl_isrc.env import VecEnv
# 实现 VecEnv 后，将 env 与 train_cfg 传入对应 Runner，调用 runner.learn(...)
```

配置字典通常包含 `runner`（如 `experiment_name`、`num_steps_per_env`、`save_interval`）、`algorithm`、`policy` 等键；具体字段请参考各 **Runner** 实现与 [`rsl_rl_isrc/examples/`](rsl_rl_isrc/examples/) 中的示例。

## 可选：HTTP 遥测环境变量

| 变量 | 作用 |
|------|------|
| `RSL_RL_ISRC_POST_URL` | `send_post_request` 使用的 POST 地址（默认见 `rsl_rl_isrc/sockets/http_post.py`） |
| `RSL_RL_ISRC_OBS_POST_URL` | 若设置，`StepObsPublisher` 在 `env.step` 后按指令切片上报观测；未设置则不发送 |

## 测试与文档

- 测试脚本与索引：[`rsl_rl_isrc/tests/README.md`](rsl_rl_isrc/tests/README.md)  
- 各算法/Runner 的详细运行说明见 `rsl_rl_isrc/tests/README_*.md` 与 `rsl_rl_isrc/examples/README_*.md`。

示例（Runner 冒烟）：

```bash
python rsl_rl_isrc/tests/test_ppo_runner.py
```

使用 **pytest**：

```bash
pytest rsl_rl_isrc/tests/ -v
```

## 分布式训练说明（简述）

部分 Runner 在 `torch.distributed` 初始化后会对参数做 `broadcast` / `barrier`；PPO 等场景下常见模式为 **rank 0 执行 `update` 再下发权重**。若启用 `StepObsPublisher`，多进程下会按服务端返回的 `state` 指令做 **单 rank POST** 与 **指令张量广播**，详见 `rsl_rl_isrc/sockets/http_post.py` 中 `StepObsPublisher` 的文档字符串。

## 许可证

**BSD-3-Clause**（与 rsl_rl 系列常见协议一致；以仓库内 `LICENSE` 及源文件声明为准）。
