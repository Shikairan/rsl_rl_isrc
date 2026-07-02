# 仅供 tests 使用：带 ObsInstrServer 的 OnPolicyRunner，支持 obs 上报与 _instr 指令同步。

from __future__ import annotations

from typing import Optional

from rsl_rl_isrc.runners.on_policy_runner import OnPolicyRunner
from rsl_rl_isrc.sockets import ObsInstrServer
from rsl_rl_isrc.sockets.obs_server import default_obs_env_hi


class G1OnPolicyTestRunner(OnPolicyRunner):
    """单进程 PPO Runner：绑定 ``ObsInstrServer``，每轮学习迭代前 ``sync_instr()``。

    - ``StepObsPublisher.push(obs)`` 在父类 ``learn`` 的 rollout 中调用
    - 绑定后 publisher 与 server 共享 ``_instr`` 张量
    - 测试可通过 :meth:`set_instruction` 或 ZMQ REP 更新切片指令
    - 单次 ``super().learn(N)``（非 ``learn(1)`` 套娃），与 LSTM PPO 兼容
    """

    def __init__(
        self,
        env,
        train_cfg,
        log_dir=None,
        device="cpu",
        checkpoint_dir=None,
        *,
        enable_obs_server: bool = True,
        obs_pull_port: Optional[int] = None,
        ctrl_rep_port: Optional[int] = None,
        obs_server_host: str = "localhost",
        print_obs: bool = False,
    ):
        super().__init__(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        self.obs_server: Optional[ObsInstrServer] = None
        if enable_obs_server:
            self.obs_server = ObsInstrServer(
                rank=self.rank,
                task=self.task,
                num_envs=self.env.num_envs,
                obs_pull_port=obs_pull_port,
                ctrl_rep_port=ctrl_rep_port,
                print_obs=print_obs,
            )
            self.obs_server.start()
            target_host = "localhost" if self.rank == 0 else obs_server_host
            self.obs_server.bind_publisher(
                self.step_obs, env=self.env, host=target_host
            )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._log_iter_denominator = int(num_learning_iterations)
        self._log_iter_start = int(self.current_learning_iteration)
        try:
            super().learn(
                num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
                pre_iter_callback=self._pre_iter_sync_instr,
            )
        finally:
            self._log_iter_denominator = None
            self._log_iter_start = 0
            if self.obs_server is not None:
                self.obs_server.stop()
                self.obs_server = None

    def _pre_iter_sync_instr(self, it: int) -> None:
        if self.obs_server is not None:
            self.obs_server.sync_instr()

    def set_instruction(self, instr: list) -> bool:
        """测试辅助：直接更新 obs 切片指令 ``[rank, aux, env_lo, env_hi)``。"""
        if self.obs_server is None:
            return False
        return self.obs_server._apply_instr_update({"state": instr})

    def get_instruction(self) -> list:
        if self.obs_server is None:
            hi = default_obs_env_hi(self.env.num_envs)
            return [0, 0, 0, hi]
        return self.obs_server.get_instr().tolist()
