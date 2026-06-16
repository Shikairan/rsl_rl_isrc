# Isaac Gym G1 + PPO 测试说明

## 依赖

- NVIDIA Isaac Gym（Preview）已安装并可 `import isaacgym`
- CUDA GPU（`cuda:0`）
- 本仓库 `rsl_rl_isrc/robotmodel/g1_description`（Isaac 训练默认 `g1_12dof.urdf`，含 `meshes/`）

## 运行

```bash
# 正式训练（默认内置 ObsInstrServer）
python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 4096 --max-iterations 10000

# 短训 + 打印 obs 摘要
python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 128 --max-iterations 5 --print-obs

# 纯训练（无 ZMQ）
python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --no-zmq-obs --num-envs 64 --max-iterations 5

# pytest（若标记 isaac）
pytest rsl_rl_isrc/tests/test_ppo_g1_isaac.py -v -m isaac
```

## 分布式（多机多卡）

`rsl_rl_isrc` 的 PPO Runner 在 `torch.distributed` 初始化后采用：

- 各 rank 本地 rollout / 推理
- `RolloutStorage.broadcast()` gather 到 rank0
- 仅 rank0 执行 backward/update
- 参数广播回所有 rank

可直接使用 `test_ppo_g1_isaac_ddp.py`：

```bash
# 单机 2 卡 smoke
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
  --num-envs 64 --max-iterations 5 --print-obs

# 多机 2x2（节点0）
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
  --master_addr=10.0.0.1 --master_port=29500 \
  rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
  --num-envs 64 --max-iterations 5 --obs-server-host 10.0.0.1 --print-obs

# 多机 2x2（节点1）
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
  --master_addr=10.0.0.1 --master_port=29500 \
  rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
  --num-envs 64 --max-iterations 5 --obs-server-host 10.0.0.1 --print-obs
```

> `--num-envs` 语义为每个 rank 的本地并行环境数，全局吞吐约为 `num_envs * world_size`。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `G1_NUM_ENVS` | `4096` | 并行环境数 |
| `RSL_RL_ISRC_OBS_PULL_PORT` | `15555` | ObsInstrServer PULL（训练进程 bind） |
| `RSL_RL_ISRC_CTRL_REP_PORT` | `15556` | 外部指令 REP |
| `RSL_RL_ISRC_OBS_PRINT` | `0` | 设为 `1` 时打印 obs 摘要 |
| `RSL_RL_ISRC_OBS_RELAY_URL` | 空 | 非空则将 `obs_step` HTTP POST 到该 URL（训练遥测唯一通路） |
| `RSL_RL_ISRC_OBS_SERVER_HOST` | 空 | 多机 DDP 下非 rank0 连接 rank0 ObsInstrServer 的主机名/IP（未设置时可用 `--obs-server-host`） |

## VecEnv 与机器人状态导出

G1 训练使用 [`IsaacG1VecEnv`](../env/isaac_gym/isaac_g1_vec_env.py)，继承 [`StateExportVecEnv`](../env/state_export_vec_env.py)：

- 策略 `obs` 维数由用户 `compute_observations` 决定（G1 当前为 47 维）。
- 遥测位姿/关节**不解析 obs**，从 VecEnv 上的仿真真值读取：
  - `base_pos` (N, 3)
  - `base_quat` (N, 4)，**xyzw**
  - `dof_pos` (N, num_dof)

自写 env 可 `StateExportVecEnv(MyEnv)` 包装，内层提供上述张量即可。

## obs_step 消息（ZMQ）与 HTTP 中继

`StepObsPublisher.push` → ZMQ → `ObsInstrServer` 收到完整 JSON；可选 `RSL_RL_ISRC_OBS_RELAY_URL` 仅 POST 位姿行：

| 通道 | 载荷 |
|------|------|
| ZMQ（15555） | 完整 `obs_step`：`obs` + 可选 `base_pos` / `base_quat` / `dof_pos` |
| HTTP relay | `[[base_pos, base_quat, dof_pos], ...]` 每 env 一行；缺字段或非 list → 该槽 `[]` |

## obs 与动态 env 切片

`test_ppo_g1_isaac.py` 使用 `G1OnPolicyTestRunner`，自动启动 `ObsInstrServer` 并绑定 `StepObsPublisher`：

- **15555**：PULL，接收 rollout 数据
- **15556**：REP，外部发指令改监控范围

默认指令 `_instr`：`[0, 0, 0, min(64, num_envs)]`。

另开终端修改 env index：

```bash
python rsl_rl_isrc/tests/zmq_obs_ctrl_client.py --state 0 0 10 20

# 多机时指定 rank0 机器
python rsl_rl_isrc/tests/zmq_obs_ctrl_client.py --host 10.0.0.1 --state 1 0 0 32
```

消息格式：`{"state": [sender_rank, aux, env_start, env_end)}`。

## 代码位置

- `rsl_rl_isrc/tests/test_ppo_g1_isaac.py` — 训练入口
- `rsl_rl_isrc/env/isaac_gym/test_runner.py` — `G1OnPolicyTestRunner`
- `rsl_rl_isrc/env/state_export_vec_env.py` — `StateExportVecEnv`
- `rsl_rl_isrc/sockets/http_post.py` — `StepObsPublisher`
- `rsl_rl_isrc/sockets/obs_server.py` — `ObsInstrServer`
- `rsl_rl_isrc/tests/zmq_obs_ctrl_client.py` — 15556 指令客户端

`zmq_obs_pull_server.py` 已废弃（不可与训练进程同时 bind 15555）。

## HTTP obs 中继（可选）

```bash
# 终端 1：POST 接收并打印
python rsl_rl_isrc/tests/http_post_server.py

# 终端 2：训练并 relay 到 HTTP（无需 RSL_RL_ISRC_POST_URL）
export RSL_RL_ISRC_OBS_RELAY_URL=http://127.0.0.1:18888/post
python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 64 --max-iterations 5
```

## 网络与端口注意事项（多机）

- `torchrun` 需要 `MASTER_ADDR:MASTER_PORT`（例：`10.0.0.1:29500`）可互通
- ObsInstrServer 默认使用 `15555`（PULL）和 `15556`（REP）
- 防火墙需放通 `29500`、`15555`、`15556`
