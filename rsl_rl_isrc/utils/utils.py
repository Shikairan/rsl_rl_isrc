# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
import torch
import numpy as np
import torch.nn.functional as F

def pad_to_fixed(padded, fixed_len, batch_first=False):
    """
    sequences: List[Tensor], 每条 (T_i, feat_dim)
    fixed_len: 想要的时间步长度
    return:    Tensor, 形状 (batch, fixed_len, feat_dim) 或 (fixed_len, batch, feat_dim)
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
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
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
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
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
    """Line search implementation"""
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
    """Running mean and std for observation normalization"""

    def __init__(self, shape, epsilon=1e-4, clip=10.):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.clip = clip

    @torch.no_grad()
    def __call__(self, x, update=True):
        """Normalize input x using running statistics"""
        if update:
            self.update(x)
        return torch.clamp((x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + 1e-8),
                            -self.clip, self.clip)

    def update(self, x):
        """Update running statistics with batch x"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean.to(x.device)
        tot_count = self.count + batch_count
        self.mean += delta.cpu() * batch_count / tot_count
        self.var = (self.var * self.count + batch_var.cpu() * batch_count +
                    (delta ** 2).cpu() * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count
