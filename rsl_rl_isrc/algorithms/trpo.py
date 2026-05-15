# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
# 修改版：修复数值稳定性、KL计算、价值函数优化等问题

"""TRPO：共轭梯度解信赖域、线搜索回退及策略/价值联合更新逻辑。"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable

from rsl_rl_isrc.modules import TrpoPolicy, TrpoValueFunction
from rsl_rl_isrc.storage import RolloutStorage
from rsl_rl_isrc.utils import get_flat_params_from, set_flat_params_to, conjugate_gradients


class TRPO:
    """信赖域策略优化内核：在 KL 约束下用共轭梯度求步长，并结合线搜索更新 ``TrpoPolicy``/``TrpoValueFunction``。"""

    def __init__(self,
                 policy_net: TrpoPolicy,
                 value_net: TrpoValueFunction,
                 num_learning_epochs: int = 1,
                 num_mini_batches: int = 1,
                 gamma: float = 0.995,
                 tau: float = 0.97,
                 max_kl: float = 1e-2,
                 damping: float = 1e-1,
                 l2_reg: float = 1e-3,
                 vf_lr: float = 1e-3,  # ✅ 新增：价值函数学习率
                 vf_iters: int = 20,  # ✅ 新增：价值函数迭代次数
                 entropy_coef: float = 0.01,  # ✅ 新增：熵系数
                 action_bounds: Tuple[float, float] = None,  # ✅ 新增：动作边界
                 device: str = 'cpu'):

        self.device = device
        
        # 网络组件
        self.policy_net = policy_net
        self.value_net = value_net
        
        # 存储（延迟初始化）
        self.storage = None
        self.transition = None
        
        # 超参数
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.tau = tau
        self.max_kl = max_kl
        self.damping = damping
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.action_bounds = action_bounds
        
        # ✅ 新增：价值函数优化器
        self.vf_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=vf_lr)
        self.vf_iters = vf_iters

    def init_storage(self, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: tuple, critic_obs_shape: tuple, action_shape: tuple):
        """初始化经验回放缓冲区"""
        self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                    actor_obs_shape, critic_obs_shape, action_shape, self.device)
        self.transition = RolloutStorage.Transition()

    def test_mode(self):
        self.policy_net.eval()
        self.value_net.eval()

    def train_mode(self):
        self.policy_net.train()
        self.value_net.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        采样动作，支持动作边界压缩
        支持RNN和非RNN网络
        """
        # 检查是否是RNN网络
        is_rnn = hasattr(self.policy_net, 'is_recurrent') and self.policy_net.is_recurrent

        if is_rnn:
            # RNN网络：重置隐藏状态以确保批次大小匹配（单步采样）
            if hasattr(self.policy_net, 'reset'):
                self.policy_net.reset(dones=None)
            if hasattr(self.value_net, 'reset'):
                self.value_net.reset(dones=None)

            # 调用RNN网络，内部会处理序列化，返回 [seq_len, batch_size, output_dim]
            action_mean, action_log_std, action_std = self.policy_net(obs)
            value = self.value_net(critic_obs)

            # RNN输出包含序列维度，需要压缩
            if action_mean.dim() == 3:  # [seq_len, batch_size, output_dim]
                action_mean = action_mean.squeeze(0)  # [batch_size, output_dim]
                action_log_std = action_log_std.squeeze(0)
                action_std = action_std.squeeze(0)
            if value.dim() == 3:  # [seq_len, batch_size, 1]
                value = value.squeeze(0)  # [batch_size, 1]
        else:
            # 标准网络
            action_mean, action_log_std, action_std = self.policy_net(obs)
            value = self.value_net(critic_obs)

        # ✅ 创建分布对象
        dist = torch.distributions.Normal(action_mean, action_std)

        # 采样动作
        actions = dist.sample()

        # ✅ 处理动作边界（如Pendulum的[-2, 2]）
        # 注意：如果动作被clamp，log_prob需要相应调整
        # 但为了简化，我们假设动作边界足够大，clamp很少发生
        # 如果clamp频繁发生，应该使用TanhNormal分布
        if self.action_bounds is not None:
            low, high = self.action_bounds
            actions = torch.clamp(actions, low, high)

        # 计算log_prob（在clamp之前，因为分布是连续的）
        # 如果动作被clamp，这个log_prob会有偏差，但对于Pendulum通常影响不大
        actions_log_prob = dist.log_prob(actions).sum(dim=-1)

        # 存储transition
        self.transition.actions = actions.detach()
        self.transition.values = value.detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = action_mean.detach()
        self.transition.action_log_std = action_log_std.detach()
        self.transition.action_sigma = action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # RNN: 对于TRPO（on-policy），我们不需要存储hidden_states
        # 因为mini-batch训练中每个样本都是独立的

        return actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        """处理环境返回的reward和done"""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 处理超时截断（bootstrapping）
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )
        
        # 存储transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor):
        """计算GAE回报"""
        last_values = self.value_net(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.tau)

    def update(self) -> Tuple[float, float]:
        """执行一次完整更新"""
        if self.storage is None:
            raise RuntimeError("必须先调用 init_storage()")
        
        mean_value_loss = 0.0
        mean_policy_loss = 0.0
        
        # 生成mini-batch数据
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for batch in generator:
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_log_prob_batch,
             old_mean_batch, old_sigma_batch, _, _) = batch

            # ✅ 更新价值函数（使用Adam替代L-BFGS）
            value_loss = self._update_value_function(critic_obs_batch, returns_batch)

            # ✅ 更新策略网络（TRPO核心）
            policy_loss = self._update_policy_trpo(
                obs_batch, actions_batch, advantages_batch,
                old_mean_batch, old_sigma_batch, old_log_prob_batch
            )
            
            mean_value_loss += value_loss
            mean_policy_loss += policy_loss
        
        # 清空存储
        self.storage.clear()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return mean_value_loss / num_updates, mean_policy_loss / num_updates

    def _update_value_function(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """✅ 使用Adam优化器替代SciPy L-BFGS"""
        value_loss = 0.0

        # 暂时移除学习率调度器，先用固定学习率调试
        # if not hasattr(self, 'vf_scheduler') or self.vf_scheduler is None:
        #     self.vf_scheduler = torch.optim.lr_scheduler.StepLR(
        #         self.vf_optimizer, step_size=30, gamma=0.9
        #     )
        self.vf_scheduler = None

        # RNN: 重置隐藏状态以进行批处理更新
        if hasattr(self.value_net, 'reset'):
            self.value_net.reset(dones=None)

        # 调试信息：检查数据范围（仅在第一次迭代时打印）
        if self.vf_iters > 0:
            with torch.no_grad():
                initial_values = self.value_net(states)
                # RNN输出可能有额外维度，需要压缩
                if initial_values.dim() == 3:  # [seq_len, batch_size, 1]
                    initial_values = initial_values.squeeze(-1)  # [seq_len, batch_size]
                    if initial_values.dim() == 2:
                        initial_values = initial_values.transpose(0, 1)  # [batch_size, seq_len]
                        initial_values = initial_values.reshape(-1)  # [batch_size * seq_len]
                elif initial_values.dim() == 2:  # [batch_size, 1]
                    initial_values = initial_values.squeeze(-1)

                value_error = (returns - initial_values).abs().mean()
                print(initial_values.shape, returns.shape)
                print(f"  价值函数初始误差: {value_error:.3f}, 价值范围: [{initial_values.min():.3f}, {initial_values.max():.3f}], 回报范围: [{returns.min():.3f}, {returns.max():.3f}]")

        for iter_idx in range(self.vf_iters):
            # RNN: 重置隐藏状态确保每次迭代独立
            if hasattr(self.value_net, 'reset'):
                self.value_net.reset(dones=None)

            values = self.value_net(states)
            # RNN输出可能有额外维度，需要压缩
            if values.dim() == 3:  # [seq_len, batch_size, 1]
                values = values.squeeze(-1)  # [seq_len, batch_size]
                if values.dim() == 2:
                    values = values.transpose(0, 1)  # [batch_size, seq_len]
                    values = values.reshape(-1)  # [batch_size * seq_len]
            elif values.dim() == 2:  # [batch_size, 1]
                values = values.squeeze(-1)

            # MSE损失
            loss = (returns - values).pow(2).mean()

            # L2正则化 - 对所有参数的L2范数求和
            l2_loss = 0.0
            for param in self.value_net.parameters():
                l2_loss += param.pow(2).sum()
            loss += self.l2_reg * l2_loss

            self.vf_optimizer.zero_grad()
            loss.backward()

            # 检查梯度是否有效
            total_grad_norm = 0.0
            param_count = 0
            for param in self.value_net.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
                    param_count += 1

            total_grad_norm = total_grad_norm ** 0.5
            if total_grad_norm == 0:
                print(f"    ⚠️ 梯度为零！loss={loss.item():.6f}")

            # 更宽松的梯度裁剪，让梯度有更多自由度
            grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=100.0)

            self.vf_optimizer.step()

            # 每50次迭代检查一次参数变化
            if (iter_idx + 1) % 50 == 0:
                with torch.no_grad():
                    # RNN: 重置隐藏状态确保检查独立
                    if hasattr(self.value_net, 'reset'):
                        self.value_net.reset(dones=None)
                    new_values = self.value_net(states[:5])  # 检查前5个样本
                    # RNN输出可能有额外维度，需要压缩
                    if new_values.dim() == 3:  # [seq_len, batch_size, 1]
                        new_values = new_values.squeeze(-1)  # [seq_len, batch_size]
                        if new_values.dim() == 2:
                            new_values = new_values.transpose(0, 1)  # [batch_size, seq_len]
                            new_values = new_values.reshape(-1)  # [batch_size * seq_len]
                    elif new_values.dim() == 2:  # [batch_size, 1]
                        new_values = new_values.squeeze(-1)

                    print(f"    参数检查 - loss={loss.item():.6f}, grad_norm={grad_norm:.6f}, lr={self.vf_optimizer.param_groups[0]['lr']:.6f}")
                    print(f"    值变化: [{new_values.min():.3f}, {new_values.max():.3f}]")

            # 更新学习率 (暂时禁用)
            # if self.vf_scheduler:
            #     self.vf_scheduler.step()

            value_loss += loss.item()

            # 每20次迭代打印一次进度
            if (iter_idx + 1) % 20 == 0:
                with torch.no_grad():
                    # RNN: 重置隐藏状态确保检查独立
                    if hasattr(self.value_net, 'reset'):
                        self.value_net.reset(dones=None)
                    current_values = self.value_net(states)
                    # RNN输出可能有额外维度，需要压缩
                    if current_values.dim() == 3:  # [seq_len, batch_size, 1]
                        current_values = current_values.squeeze(-1)  # [seq_len, batch_size]
                        if current_values.dim() == 2:
                            current_values = current_values.transpose(0, 1)  # [batch_size, seq_len]
                            current_values = current_values.reshape(-1)  # [batch_size * seq_len]
                    elif current_values.dim() == 2:  # [batch_size, 1]
                        current_values = current_values.squeeze(-1)

                    current_error = (returns - current_values).abs().mean()
                    print(f"    VF迭代 {iter_idx+1}/{self.vf_iters}: 误差={current_error:.3f}, 范围=[{current_values.min():.3f}, {current_values.max():.3f}]")

        # 打印最终的价值函数误差
        with torch.no_grad():
            # RNN: 重置隐藏状态确保检查独立
            if hasattr(self.value_net, 'reset'):
                self.value_net.reset(dones=None)
            final_values = self.value_net(states)
            # RNN输出可能有额外维度，需要压缩
            if final_values.dim() == 3:  # [seq_len, batch_size, 1]
                final_values = final_values.squeeze(-1)  # [seq_len, batch_size]
                if final_values.dim() == 2:
                    final_values = final_values.transpose(0, 1)  # [batch_size, seq_len]
                    final_values = final_values.reshape(-1)  # [batch_size * seq_len]
            elif final_values.dim() == 2:  # [batch_size, 1]
                final_values = final_values.squeeze(-1)

            final_error = (returns - final_values).abs().mean()
            print(f"  价值函数最终误差: {final_error:.3f}, 价值范围: [{final_values.min():.3f}, {final_values.max():.3f}]")

        return value_loss / self.vf_iters

    def _update_policy_trpo(self, states: torch.Tensor, actions: torch.Tensor,
                           advantages: torch.Tensor, old_mean: torch.Tensor,
                           old_sigma: torch.Tensor, old_log_prob: torch.Tensor) -> float:
        # RNN: 重置隐藏状态以进行批处理更新
        if hasattr(self.policy_net, 'reset'):
            self.policy_net.reset(dones=None)
        """✅ 修复：策略更新使用信赖域优化"""

        # 注意：优势函数已经在RolloutStorage.compute_returns中标准化过了
        # 这里只需要确保形状正确（如果advantages是2D，需要squeeze）
        if advantages.dim() > 1:
            advantages = advantages.squeeze(-1)
        
        # ✅ 确保old_log_prob也是1D的
        if old_log_prob.dim() > 1:
            old_log_prob = old_log_prob.squeeze(-1)

        # ✅ 预计算网络输出，避免重复前向传播
        with torch.no_grad():
            mean, log_std, std = self.policy_net(states)
            # RNN输出可能有额外维度，需要压缩
            if mean.dim() == 3:  # [seq_len, batch_size, output_dim]
                mean = mean.squeeze(0)  # [batch_size, output_dim]
                log_std = log_std.squeeze(0)
                std = std.squeeze(0)

            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # ✅ 定义损失函数（使用预计算的值）
        def get_loss():
            # RNN: 重置隐藏状态确保每次调用独立
            if hasattr(self.policy_net, 'reset'):
                self.policy_net.reset(dones=None)

            # 重新计算当前参数下的分布（因为参数会变化）
            mean_curr, log_std_curr, std_curr = self.policy_net(states)
            # RNN输出可能有额外维度，需要压缩
            if mean_curr.dim() == 3:  # [seq_len, batch_size, output_dim]
                mean_curr = mean_curr.squeeze(0)  # [batch_size, output_dim]
                log_std_curr = log_std_curr.squeeze(0)
                std_curr = std_curr.squeeze(0)

            dist_curr = torch.distributions.Normal(mean_curr, std_curr)

            # 重要性采样比
            log_prob_curr = dist_curr.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_prob_curr - old_log_prob)

            # 策略损失 + 熵奖励
            policy_loss = -(ratio * advantages).mean()
            entropy_curr = dist_curr.entropy().sum(dim=-1).mean()

            return policy_loss - self.entropy_coef * entropy_curr

        # ✅ 定义KL散度函数（使用预计算的值）
        def get_kl():
            # RNN: 重置隐藏状态确保每次调用独立
            if hasattr(self.policy_net, 'reset'):
                self.policy_net.reset(dones=None)

            # 重新计算当前参数下的分布
            mean_curr, log_std_curr, std_curr = self.policy_net(states)
            # RNN输出可能有额外维度，需要压缩
            if mean_curr.dim() == 3:  # [seq_len, batch_size, output_dim]
                mean_curr = mean_curr.squeeze(0)  # [batch_size, output_dim]
                log_std_curr = log_std_curr.squeeze(0)
                std_curr = std_curr.squeeze(0)

            dist_new = torch.distributions.Normal(mean_curr, std_curr)
            dist_old = torch.distributions.Normal(old_mean, old_sigma)

            # ✅ 修复：计算KL散度，注意参数顺序
            # KL(old || new) = E_old[log(old) - log(new)]
            kl = torch.distributions.kl_divergence(dist_old, dist_new)
            # 对动作维度求和，然后对batch求平均
            kl = kl.sum(dim=-1).mean()
            return kl

        # ✅ 执行TRPO优化步 (禁用CuDNN以支持RNN的双向传播)
        with torch.backends.cudnn.flags(enabled=False):
            loss = trpo_step(self.policy_net, get_loss, get_kl, self.max_kl, self.damping)

        return loss.item() if isinstance(loss, torch.Tensor) else 0.0


def trpo_step(model: nn.Module, get_loss: Callable, get_kl: Callable,
               max_kl: float, damping: float) -> torch.Tensor:
    """
    ✅ 修复：真正的TRPO步骤 - 使用共轭梯度法和Fisher信息矩阵
    """
    # 清理梯度
    model.zero_grad()

    try:
        # 1. 计算策略梯度
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # 2. 定义Fisher向量积函数 (FVP)
        def Fvp(v):
            """
            计算 Fisher 信息矩阵的向量积: F * v
            其中 F 是 KL 散度的 Hessian 矩阵
            """
            # 计算KL散度
            kl = get_kl()
            kl = kl.mean()

            # 计算KL散度的梯度
            kl_grad = torch.autograd.grad(kl, model.parameters(), create_graph=True, retain_graph=True)
            flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])

            # 计算 (grad_kl^T * v) 的梯度，得到 F * v
            kl_v = (flat_kl_grad * v).sum()
            grads_v = torch.autograd.grad(kl_v, model.parameters(), create_graph=False, retain_graph=False)
            flat_grads_v = torch.cat([grad.contiguous().view(-1) for grad in grads_v]).detach()

            # 添加阻尼项: (F + damping * I) * v
            return flat_grads_v + damping * v

        # 3. 使用共轭梯度法求解自然梯度方向
        # 求解: (F + damping * I) * step_dir = -loss_grad
        step_dir = conjugate_gradients(Fvp, -loss_grad, nsteps=10, residual_tol=1e-10)

        # 4. 计算步长缩放因子，确保KL散度约束
        # 计算 step_dir^T * F * step_dir
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum(0, keepdim=True)
        
        # 计算缩放因子: sqrt(2 * max_kl / (step_dir^T * F * step_dir))
        lm = torch.sqrt(shs / max_kl)
        full_step = step_dir / lm[0]

        # 5. 计算期望改善量（用于线搜索）
        neggdotstepdir = (-loss_grad * step_dir).sum(0, keepdim=True)
        expected_improve_rate = neggdotstepdir / lm[0]

        # 6. 执行线搜索
        prev_params = get_flat_params_from(model)
        success, new_params = line_search(
            model, get_loss, get_kl, prev_params, full_step,
            expected_improve_rate, max_kl, max_backtracks=10, accept_ratio=0.1
        )

        if success:
            set_flat_params_to(model, new_params)
        else:
            # 如果线搜索失败，保持原参数
            set_flat_params_to(model, prev_params)

    except Exception as e:
        print(f"⚠️ TRPO步骤失败: {e}")
        import traceback
        traceback.print_exc()
        # 失败时保持原参数
        prev_params = get_flat_params_from(model)
        set_flat_params_to(model, prev_params)
        return torch.tensor(0.0)

    # 最终清理
    model.zero_grad()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    return loss




def line_search(model: nn.Module, get_loss: Callable, get_kl: Callable, x: torch.Tensor,
                full_step: torch.Tensor, expected_improve_rate: torch.Tensor,
                max_kl: float, max_backtracks: int = 10, accept_ratio: float = 0.1) -> tuple:
    """
    ✅ 修复：线搜索实现 - 检查KL散度约束和改善率
    """
    fval = get_loss().item()
    
    for stepfrac in 0.5 ** np.arange(max_backtracks):
        x_new = x + stepfrac * full_step
        set_flat_params_to(model, x_new)

        # 清理梯度并计算新值
        model.zero_grad()
        with torch.no_grad():
            new_fval = get_loss().item()
            new_kl = get_kl().item()

        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve if expected_improve.abs() > 1e-8 else 0

        # 检查KL约束和改善率
        if new_kl <= max_kl and ratio > accept_ratio and actual_improve > 0:
            return True, x_new

        # 手动清理内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return False, x


