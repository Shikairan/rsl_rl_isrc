# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""轨迹切分/补零、参数向量拉平、共轭梯度与线搜索等通用算子。"""

import math
import torch
import numpy as np
import torch.nn.functional as F

def pad_to_fixed(padded, fixed_len, batch_first=False):
    """在已有 ``pad_sequence`` 结果上再补零/截断到固定时间长度 ``fixed_len``。

    参数:
        padded: 已对齐 batch 内最长序列的张量。
        fixed_len: 目标时间维长度。
        batch_first: True 时形状 ``(B, T, D)``，否则 ``(T, B, D)``。
    """
    # 1. 先常规补齐到 batch 内最长
    #padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    # 2. 再补到固定长度
    if batch_first:
        # padded 形状 (B, T_max, D)
        pad_len = fixed_len - padded.size(1)
        if pad_len > 0:
            padded = F.pad(padded, (0, 0, 0, pad_len), "constant", 0)
    else:
        # padded 形状 (T_max, B, D)
        pad_len = fixed_len - padded.size(0)
        if pad_len > 0:
            padded = F.pad(padded, (0, 0, 0, 0, 0, pad_len), "constant", 0)
    return padded



def split_and_pad_trajectories(tensor, dones):
    """在 ``dones`` 处切段并行轨迹，再拼接并用零填充到最长段，并返回有效步掩码。

    维度约定：``tensor`` 为 ``[时间, 环境数, …]``；与 ``unpad_trajectories`` 互为逆操作。

    返回:
        (padded_trajectories, trajectory_masks)，后者布尔掩码标出非填充位置。
    """
    dones = dones.clone()
    #print("split_and_pad_trajectories ///dones:", dones.shape)
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    #print("\n util.split_and_pad_trajectories:tensor", tensor.shape, trajectory_lengths)
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    #print("\n len(trajectory_lengths_list):", len(trajectory_lengths_list), np.max(np.array(trajectory_lengths_list)))


    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    padded_trajectories = pad_to_fixed(padded_trajectories, tensor.shape[0] ,False)


    #print("\n util.split_and_pad_trajectories.trajectories:",padded_trajectories.shape)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """``split_and_pad_trajectories`` 的逆：按掩码去掉填充并恢复紧凑时间序列形状。"""
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


def normal_entropy(std):
    """对角高斯 ``N(0, std^2)`` 的熵（按维求和）。"""
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    """在给定均值/标准差下计算 ``x`` 的对角高斯对数密度（按维求和）。"""
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    """将 ``model`` 所有参数展平为一维向量（用于 TRPO 等自然策略梯度实现）。"""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """把一维向量按 ``model`` 各参数形状写回 ``param.data``。"""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    """展平网络梯度；``grad_grad=True`` 时取 ``param.grad.grad``（二阶场景）。"""
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    """共轭梯度法解 ``A x = b``，其中 ``Avp(v)`` 返回矩阵向量积 ``A @ v``（TRPO 中求自然梯度方向）。"""
    x = torch.zeros(b.size(), device=b.device, dtype=b.dtype)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    # 添加数值稳定性检查
    if rdotr < residual_tol:
        return x

    for i in range(nsteps):
        _Avp = Avp(p)
        p_Avp = torch.dot(p, _Avp)

        # 防止除零
        if p_Avp.abs() < 1e-8:
            break

        alpha = rdotr / p_Avp
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)

        # 检查收敛
        if new_rdotr < residual_tol:
            break

        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
    return x


def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """沿 ``fullstep`` 方向回溯线搜索，直到实际改进满足 ``accept_ratio`` 或耗尽步数。"""
    fval = f(True).data
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x


class RunningMeanStd:
    """滑动均值与方差，用于观测归一化；``__call__`` 返回裁剪后的标准化张量。"""

    def __init__(self, shape, epsilon=1e-4, clip=10.):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.clip = clip

    @torch.no_grad()
    def __call__(self, x, update=True):
        """用滑动统计量归一化 ``x``；``update=False`` 时仅做仿射变换不更新统计。"""
        if update:
            self.update(x)
        return torch.clamp((x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + 1e-8),
                            -self.clip, self.clip)

    def update(self, x):
        """用当前 batch 增量更新滑动均值与方差（Welford 风格合并）。"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean.to(x.device)
        tot_count = self.count + batch_count
        self.mean += delta.cpu() * batch_count / tot_count
        self.var = (self.var * self.count + batch_var.cpu() * batch_count +
                    (delta ** 2).cpu() * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count
