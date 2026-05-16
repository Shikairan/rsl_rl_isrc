# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""PPO 训练运行器模块。

负责与环境 ``VecEnv`` 交互、``PPO`` 算法更新、分布式下参数 ``broadcast``、
TensorBoard 日志，以及可选的 HTTP 遥测（``socket_send`` / ``StepObsPublisher``）。
"""

import time
import os
from collections import deque
import statistics
from torch.utils.tensorboard import SummaryWriter
import torch
from multiprocessing import Pool
from rsl_rl_isrc.algorithms import PPO
from rsl_rl_isrc.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.sockets import send_post_request, StepObsPublisher
from rsl_rl_isrc.sockets.http_post import _POST_TIMEOUT
import torch.distributed as dist


class OnPolicyRunner:
    """封装 PPO 训练：构建 ``ActorCritic``、``PPO``、驱动 ``learn`` 与检查点/日志。"""

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg=train_cfg["runner"]
        self.task = self.cfg["experiment_name"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = device
        self.env = env
        #print(self.env.num_dof)
        self.pool = Pool(processes=60) if dist.is_initialized() else None
        self.state_tag = torch.tensor([0,0,0,64]).to(self.device)
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        # 安全的类名解析（用白名单替代 eval()，避免任意代码执行）
        _policy_registry = {
            "ActorCritic":          ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        _alg_registry = {
            "PPO": PPO,
        }
        policy_class_name = self.policy_cfg.get("policy_class_name", "ActorCritic")
        alg_class_name    = self.alg_cfg.get("algorithm_class_name", "PPO")
        actor_critic_class = _policy_registry.get(policy_class_name)
        if actor_critic_class is None:
            raise ValueError(f"未知策略类 '{policy_class_name}'，支持: {list(_policy_registry.keys())}")
        alg_class = _alg_registry.get(alg_class_name)
        if alg_class is None:
            raise ValueError(f"未知算法类 '{alg_class_name}'，支持: {list(_alg_registry.keys())}")

        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        alg_kwargs = {k: v for k, v in self.alg_cfg.items() if k != "algorithm_class_name"}
        self.alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
        self.retstate_list = []
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.step_obs = StepObsPublisher(self.rank, self.task, self.env.num_envs)

        self.env.reset(torch.arange(self.env.num_envs))
   

    def socket_send(self):
        """按 ``state_tag`` 将部分 env 的位姿/关节经进程池异步 POST 到远端（与渲染或调度服务对接）。

        仅当 ``state_tag[0]`` 等于本进程 ``rank`` 且 ``state_tag[1] != 1`` 时发送；
        返回的 ``state`` 在 ``learn`` 中由指定 rank 拉取并 ``broadcast`` 更新 ``state_tag``。
        """
        if not dist.is_initialized() or self.pool is None:
            return
        if not hasattr(self.env, 'base_pos') or not hasattr(self.env, 'base_quat') or not hasattr(self.env, 'dof_pos'):
            return
        if self.state_tag[0] != self.rank:
            return
        if self.state_tag[1] == 1.0:
            return
        try:
            location = self.env.base_pos[self.state_tag[2]:self.state_tag[3], :].clone()
            location[:, 1] = location[:, 1] - torch.arange(
                int(self.state_tag[2].item()), int(self.state_tag[3].item()), device=location.device
            ).float() * 3
            tt = torch.cat((
                location,
                self.env.base_quat[self.state_tag[2]:self.state_tag[3], :],
                self.env.dof_pos[self.state_tag[2]:self.state_tag[3], :]
            ), dim=1).cpu().tolist()
            ret = self.pool.apply_async(send_post_request, args=(tt, self.rank, self.task))
            self.retstate_list.append(ret)
        except Exception:
            pass
 
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """主训练循环：多轮 ``num_steps_per_env`` 采集、回报计算、分布式存储同步、rank0 上 ``PPO.update``。

        参数:
            num_learning_iterations: 追加的外层迭代次数（在 ``current_learning_iteration`` 基础上递增）。
            init_at_random_ep_len: 若为 True，在首轮随机化各并行环境的剩余回合长度（探索起点多样性）。

        分布式约定:
            每轮结束由 ``state_tag[0]`` 指定 rank 取回 HTTP 异步结果并广播 ``state_tag``；
            仅 rank0 执行 ``alg.update``，随后对所有 rank 广播 actor_critic 参数。
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None and (not dist.is_initialized() or dist.get_rank() == 0):
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        if dist.is_initialized():
            for param in self.alg.actor_critic.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            reslist = []
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    self.step_obs.push(obs)
                    self.socket_send() 
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            if dist.is_initialized():
                dist.barrier()
                # state_tag 遥测同步：由 state_tag[0] 负责更新，所有 rank 参与 broadcast
                rank_val = int(self.state_tag[0].item()) if torch.is_tensor(self.state_tag[0]) else int(self.state_tag[0])
                if dist.get_rank() == rank_val:
                    try:
                        tmp_state = self.retstate_list[-1].get(timeout=_POST_TIMEOUT + 2) if self.retstate_list else {}
                        if isinstance(tmp_state, dict) and 'error' not in tmp_state:
                            self.state_tag = torch.tensor(tmp_state.get('state', self.state_tag.cpu().tolist()), device=self.device)
                    except Exception as e:
                        print(e)
                    finally:
                        self.retstate_list = []  # 无论成功或异常，始终清理防止内存泄漏
                # 所有 rank 参与 broadcast（集体操作，必须在 if/try 块外调用）
                dist.broadcast(self.state_tag, src=rank_val)
                dist.barrier()
                self.alg.storage.broadcast()

                if dist.get_rank() == 0:
                    mean_value_loss, mean_surrogate_loss = self.alg.update()
                    dist.barrier()
                else:
                    self.alg.storage.clear()
                    dist.barrier()

                for param in self.alg.actor_critic.parameters():
                    dist.broadcast(param.data, src=0)
                dist.barrier()
            else:
                mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            
            if self.log_dir is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                self.log(locals())
            if it % self.save_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        """将本轮 ``locs`` 中的损失、FPS、回合统计写入 TensorBoard 并打印文本摘要（仅 rank0 调用）。"""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)


    def save(self, path, infos=None):
        """保存 ``actor_critic``、优化器状态与当前迭代号到 ``path``（PyTorch ``torch.save``）。"""
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        """从检查点 ``path`` 恢复模型与（可选）优化器，并恢复 ``current_learning_iteration``。"""
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """返回 ``actor_critic.act_inference``，用于部署或评估（eval 模式，可选 ``device`` 迁移）。"""
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
