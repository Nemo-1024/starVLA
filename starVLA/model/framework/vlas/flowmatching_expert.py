#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .cross_attention_dit import DiT, AlternateVLDiT



class ContinuousTimeEncoder(nn.Module):
    """Maps continuous time in seconds to sinusoidal embeddings."""

    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"`embedding_dim` must be > 0, got {embedding_dim}.")
        self.embedding_dim = int(embedding_dim)
        half_dim = max(1, self.embedding_dim // 2)
        denom = max(half_dim - 1, 1)
        freq_idx = torch.arange(half_dim, dtype=torch.float32)
        freqs = torch.exp(-math.log(float(max_period)) * freq_idx / float(denom))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        angles = t.unsqueeze(-1) * self.freqs.to(device=t.device)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embedding_dim:
            emb = F.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
        return emb[..., : self.embedding_dim]


def build_time_grid(
    horizon_sec: float,
    hz: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-sample natural-time grids t_i = i / hz and hz-derived token masks.

    Returns:
        t_grid: (B, T) float timestamps in seconds.
        time_valid: (B, T) bool mask from floor(horizon_sec * hz).
        n_tokens: (B,) long effective token count per sample.
    """
    if hz.ndim != 1:
        raise ValueError(f"`hz` must have shape [B], got {tuple(hz.shape)}.")
    if seq_len <= 0:
        raise ValueError(f"`seq_len` must be > 0, got {seq_len}.")
    if horizon_sec <= 0:
        raise ValueError(f"`horizon_sec` must be > 0, got {horizon_sec}.")

    device = hz.device
    bsz = hz.shape[0]
    hz_f = hz.to(device=device, dtype=torch.float32)
    if torch.any(hz_f <= 0):
        bad_idx = int(torch.nonzero(hz_f <= 0, as_tuple=False)[0].item())
        raise ValueError(f"`hz` must be > 0 for all samples, got hz[{bad_idx}]={float(hz_f[bad_idx])}.")

    n_tokens = torch.floor(float(horizon_sec) * hz_f).to(dtype=torch.long)
    if torch.any(n_tokens < 1):
        bad_idx = int(torch.nonzero(n_tokens < 1, as_tuple=False)[0].item())
        raise ValueError(
            f"Invalid effective token count from horizon_sec*hz: sample={bad_idx}, "
            f"horizon_sec={horizon_sec}, hz={float(hz_f[bad_idx])}, floor={int(n_tokens[bad_idx])}. "
            "Increase `horizon_sec` or ensure hz>=1."
        )
    n_tokens = torch.clamp(n_tokens, max=int(seq_len))
    token_idx = torch.arange(int(seq_len), dtype=torch.float32, device=device).unsqueeze(0).expand(bsz, -1)
    t_grid = token_idx / hz_f.unsqueeze(1)
    time_valid = token_idx.to(dtype=torch.long) < n_tokens.unsqueeze(1)
    return t_grid, time_valid, n_tokens


def normalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """归一化到[0,1]范围"""
    return (x - min_val) / (max_val - min_val)


def unnormalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """反归一化"""
    return x * (max_val - min_val) + min_val


def safe_arcsin(value: torch.Tensor) -> torch.Tensor:
    """安全的arcsin函数，确保输入在[-1,1]范围内"""
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value: torch.Tensor) -> torch.Tensor:
    """将ALOHA夹爪位置转换为角度空间 - 保持原有实现"""
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value: torch.Tensor) -> torch.Tensor:
    """从角度空间转换为ALOHA夹爪位置 - 保持原有实现"""
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value: torch.Tensor) -> torch.Tensor:
    """aloha_gripper_from_angular的逆函数 - 保持原有实现"""
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)
    


def swish(x):
    return x * torch.sigmoid(x)

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)

    def forward(self, actions: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        x = swish(self.W1(actions, cat_ids))
        return self.W2(x, cat_ids)

class VectorMLP(nn.Module):
    """简单向量投影器，用于 h_vlm / state 等向量输入到 hidden_dim"""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            x = x.unsqueeze(1)
        return self.net(x)



        
"""
Conditional Flow Matching head with Action Manifold Learning (AML).
Network predicts denoised action A_hat; velocity v_hat = (A_hat - A_t^tau) / (1 - tau)
is derived from A_hat for both training loss (MSE on v) and inference ODE integration.
CFG is applied in action space. Linear path: t=0 noise, t=1 data.
"""


@dataclass
class ConditionalFlowMatchingConfig:
    # 动作维度（输出）
    action_dim: int = 7
    # Flow 维度与步数
    hidden_dim: int = 768
    num_layers: int = 12  #DiT层数
    num_steps: int = 20
    cfg_drop_prob: float = 0.0
        # Inference controls
    cfg_guidance_scale: float = 1.0
    num_inference_steps: int = 4
    num_timestep_buckets: int = 1000

    # 可学习编码器（内部构造 cond）所需配置
    vlm_dim: int = 2048
    vision_dim: int = 768
    num_vision_tokens: int = 256
    flow_action_num_queries: int = 8
    horizon_sec: float = 1.0
    use_state: bool = True
    state_dim: int = 8
    state_dropout_prob: float = 0.5
    num_embodiments: int = 32
    # num_vision_queries: int = 64
    # qformer_layers: int = 2
    # enc_num_heads: int = 4
    # enc_hidden_dim: int = 512  # Enc(h_*) 输出维度；cond_dim = enc_hidden_dim * 4
    
    # AlternateVLDiT 相关配置
    interleave_self_attention: bool = True  # 让所有层都关注 VLM 特征
    use_alternate_vldit: bool = False  # 是否使用交替注意力模式
    attend_text_every_n_blocks: int = 2  # 每多少个块关注一次VLM特征
    

    # 噪声采样配置（与 GR00T 对齐）
    noise_beta_alpha: float = 1.5  # Beta 分布的 alpha 参数
    noise_beta_beta: float = 1.0   # Beta 分布的 beta 参数
    noise_s: float = 0.999         # 时间变换的缩放因子
    # 时间调度：`beta`(旧版) | `jit_lognormal`(JiT)
    noise_schedule: str = "jit_lognormal"
    # JiT 风格 t 采样参数：t = sigmoid(N(P_mean, P_std))
    P_mean: float = -0.8
    P_std: float = 0.8
    # JiT 风格稳定项：分母 (1-t) 的下界
    t_eps: float = 5.0e-2
    # 采样阶段稳定项（对齐 JiT PR#39）：应显著小于训练 t_eps，避免最后一步残留噪声
    sample_eps: float = 1.0e-5
    # 噪声缩放：e ~ N(0, noise_scale^2)
    noise_scale: float = 1.0


class ConditionalFlowMatchingHead(nn.Module):
    def __init__(self, config: Optional[ConditionalFlowMatchingConfig] = None):
        super().__init__()
        self.config = config or ConditionalFlowMatchingConfig()

        # 内部可学习编码器：将 (h_t, h_t1*, h_vlm, state) -> cond
               
        # self.enc_h_t_t1 = LAMEncoder(
        #     context_dim=self.config.vision_dim,
        #     query_dim=self.config.hidden_dim,
        #     num_queries=self.config.num_vision_queries,
        #     num_layers=self.config.qformer_layers,
        # )
        self.enc_vlm = VectorMLP(in_dim=self.config.vlm_dim, hidden_dim=self.config.vision_dim)
        # self.enc_a_p_to_a = VectorMLP(in_dim=2 * self.config.hidden_dim, hidden_dim=self.config.hidden_dim)
        if self.config.use_state:
            self.enc_state = VectorMLP(in_dim=self.config.state_dim, hidden_dim=self.config.hidden_dim)
            self.state_mask_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
            nn.init.normal_(self.state_mask_token, mean=0.0, std=0.02)
        else:
            self.enc_state = None
            self.state_mask_token = None
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.config.action_dim,
            hidden_size=self.config.hidden_dim,
            num_embodiments=self.config.num_embodiments,
        )
        self.time_encoder = ContinuousTimeEncoder(embedding_dim=self.config.hidden_dim)
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.config.num_embodiments,
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.action_dim,
        )
        self.flow_action_query = nn.Parameter(
            torch.randn(int(self.config.flow_action_num_queries), int(self.config.vlm_dim)) * 0.02
        )
        
        # 根据配置选择 DiT 类型
        DiTClass = AlternateVLDiT if self.config.use_alternate_vldit else DiT
        
        dit_kwargs = {
            "num_attention_heads": 16,
            "attention_head_dim": int(self.config.hidden_dim // 16),
            "output_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "interleave_self_attention": self.config.interleave_self_attention,
            "cross_attention_dim": self.config.vision_dim,  # default None 修改是为了确保cond被交叉注意力关注到
        }
        
        if self.config.use_alternate_vldit:
            dit_kwargs["attend_text_every_n_blocks"] = self.config.attend_text_every_n_blocks
        
        self.DiT = DiTClass(**dit_kwargs)
        self.cfg_embeddings = nn.Parameter(torch.randn(1, self.config.num_vision_tokens, self.config.vision_dim))
        
        # 兼容两种时间调度：
        # - beta: 旧版 GR00T 风格
        # - jit_lognormal: JiT 风格（默认）
        if self.config.noise_schedule == "beta":
            self.beta_dist = torch.distributions.Beta(
                concentration1=self.config.noise_beta_alpha,
                concentration0=self.config.noise_beta_beta
            )
        elif self.config.noise_schedule == "jit_lognormal":
            self.beta_dist = None
        else:
            raise ValueError(
                f"Unknown `noise_schedule`: {self.config.noise_schedule}. "
                "Expected one of: ['beta', 'jit_lognormal']."
            )

    def _compute_dtype(self) -> torch.dtype:
        return self.action_encoder.W1.W.dtype

    @staticmethod
    def _cast_if_needed(x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        return x if x.dtype == target_dtype else x.to(dtype=target_dtype)

    def sample_noise(
        self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # 与 JiT 对齐：e ~ N(0, noise_scale^2)
        return torch.randn(size=shape, dtype=dtype, device=device) * float(self.config.noise_scale)

    def sample_time(self, bsize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        采样时间步：
        - `jit_lognormal`: t = sigmoid(N(P_mean, P_std))（JiT）
        - `beta`: t = (noise_s - Beta(alpha,beta)) / noise_s（兼容旧版）
        """
        if self.config.noise_schedule == "jit_lognormal":
            z = torch.randn(bsize, device=device, dtype=torch.float32) * float(self.config.P_std) + float(self.config.P_mean)
            sample = torch.sigmoid(z)
        else:
            # Beta 采样在 float32 上更稳定；再 cast 回目标 dtype，避免 (bf16/half) 隐式升精度
            assert self.beta_dist is not None
            sample = self.beta_dist.sample([bsize]).to(device=device, dtype=torch.float32)
            sample = (self.config.noise_s - sample) / self.config.noise_s
        return sample.to(dtype=dtype)

    
    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor, # [B, D]
        actions: torch.Tensor, # [B, T, K]
        action_hz: torch.Tensor,  # [B]
        embodiment_id: torch.Tensor,  # [B]
        state_mask: torch.Tensor,  # [B, D]
        actions_mask: torch.Tensor,  # [B, T, K]
        attention_mask: Optional[torch.Tensor] = None,  # [B, vlm_seq_len] VLM 的 attention_mask
    ) -> torch.Tensor:
        model_dtype = self._compute_dtype()
        h_t = self._cast_if_needed(h_t, model_dtype)
        h_t1_star = self._cast_if_needed(h_t1_star, model_dtype)
        h_vlm = self._cast_if_needed(h_vlm, model_dtype)
        state = self._cast_if_needed(state, model_dtype)
        actions = self._cast_if_needed(actions, model_dtype)
        device = actions.device
        batch_size = h_t.shape[0]

        state_mask_f = state_mask.to(device=device, dtype=model_dtype)
        actions_mask_f = actions_mask.to(device=device, dtype=model_dtype)
        action_hz_f = action_hz.to(device=device, dtype=torch.float32)
        if action_hz_f.ndim != 1 or action_hz_f.shape[0] != batch_size:
            raise ValueError(
                f"`action_hz` must have shape [B], got {tuple(action_hz_f.shape)} for batch_size={batch_size}."
            )
        state = state * state_mask_f
        data_token_valid = actions_mask.to(device=device, dtype=torch.bool).any(dim=-1)
        
        # 采样噪声和时间
        noise = self.sample_noise(actions.shape, device, actions.dtype)
        time = self.sample_time(actions.shape[0], device, actions.dtype)
        time = time[:, None, None]

        # AML: 流匹配插值 A_t^tau = (1-t)*noise + t*actions（t=0 噪声，t=1 数据），保留原始值用于后续 v 计算
        A_t_tau = (1 - time) * noise + time * actions
        # 离散化时间步，并确保在有效范围内 [0, num_timestep_buckets-1]
        t_discretized = (time[:, 0, 0] * self.config.num_timestep_buckets).long()
        t_discretized = torch.clamp(t_discretized, 0, self.config.num_timestep_buckets - 1)

        # 连续时间编码：t_i = i / hz，token mask 由 horizon_sec * hz 决定
        t_grid, hz_token_valid, _ = build_time_grid(
            horizon_sec=float(self.config.horizon_sec),
            hz=action_hz_f,
            seq_len=int(actions.shape[1]),
        )
        expected_total = int(hz_token_valid.sum().item())
        actual_total = int(data_token_valid.sum().item())
        if actual_total != expected_total:
            raise ValueError(
                "Action mask/time-grid mismatch in training (batch-level count): "
                f"data_total={actual_total}, hz_total={expected_total}, "
                f"horizon_sec={self.config.horizon_sec}. "
                "Please ensure dataloader action padding matches floor(horizon_sec * action_hz)."
            )
        token_valid = data_token_valid

        # 编码动作特征（DiT 输入用 A_t_tau 的 embedding）
        noisy_trajectory_emb = self.action_encoder(A_t_tau, embodiment_id)
        time_emb = self.time_encoder(t_grid).to(dtype=noisy_trajectory_emb.dtype)
        noisy_trajectory_emb = noisy_trajectory_emb + time_emb
        noisy_trajectory_emb = noisy_trajectory_emb * token_valid.unsqueeze(-1).to(dtype=noisy_trajectory_emb.dtype)

        # 编码条件特征
        if self.config.use_state:
            cond_state = self.enc_state(state)
            if self.training and self.config.state_dropout_prob > 0.0:
                do_dropout = (
                    torch.rand(cond_state.shape[0], device=cond_state.device) < self.config.state_dropout_prob
                )
                do_dropout = do_dropout[:, None, None].to(dtype=cond_state.dtype)
                cond_state = (
                    cond_state * (1.0 - do_dropout)
                    + self.state_mask_token.to(dtype=cond_state.dtype) * do_dropout
                )
        else:
            cond_state = None
        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]

        # CFG drop（仅作用于 h_t1_star）
        if self.training and self.config.cfg_drop_prob > 0.0:
            bsz = h_t.shape[0]
            mask = (torch.rand(bsz, device=device) < self.config.cfg_drop_prob).view(bsz, 1, 1)
            cond_future = torch.where(mask, self.cfg_embeddings.expand(bsz, -1, -1), h_t1_star)
        else:
            cond_future = h_t1_star

        # 统一数据流：VLM 特征合并到 encoder_hidden_states
        encoder_hidden_states = torch.cat((h_t, cond_future, cond_vlm), dim=1)
        if self.config.use_state:
            hidden_states = torch.cat((cond_state, noisy_trajectory_emb), dim=1)
            state_token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            hidden_attention_mask = torch.cat([state_token_valid, token_valid], dim=1)
        else:
            hidden_states = noisy_trajectory_emb
            hidden_attention_mask = token_valid
        
        # 构造 encoder_attention_mask：视觉部分(h_t+h_t1)全关注 + VLM部分使用原始 attention_mask
        num_vision = h_t.shape[1] + cond_future.shape[1]  # 256 + 256 = 512
        num_vlm = cond_vlm.shape[1]
        if attention_mask is not None:
            # diffusers/SDPA 要求 mask dtype 为 bool 或 float（或与 query dtype 一致）
            # 这里统一用 bool mask：True=有效/可见，False=padding/不可见
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            # 无 mask 时全部关注
            encoder_attention_mask = None
        
        # 根据模式选择调用方式
        if self.config.use_alternate_vldit:
            # 构建 attention masks
            num_h_t = h_t.shape[1]
            num_h_t1 = cond_future.shape[1]
            num_vlm = cond_vlm.shape[1]
            
            # image_mask: 视觉部分为 True
            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            # vlm_mask: VLM 部分为 True
            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_discretized,
                hidden_attention_mask=hidden_attention_mask,
                image_mask=image_mask,
                vlm_mask=vlm_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            # 标准 DiT
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_discretized,
                hidden_attention_mask=hidden_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        
        # AML: 网络预测去噪动作 A_hat，再由 A_hat 推导 v_hat；损失仍在 v 上计算
        pred_denoised_action_all = self.action_decoder(dit_output, embodiment_id)
        A_hat = pred_denoised_action_all[:, -actions.shape[1] :, :]
        one_minus_t = (1 - time).clamp(min=float(self.config.t_eps))
        v_hat = (A_hat - A_t_tau) / one_minus_t
        v_target = (actions - A_t_tau) / one_minus_t
        loss_elem = F.mse_loss(v_hat, v_target, reduction="none")
        valid = actions_mask_f
        denom = valid.sum().clamp_min(1.0)
        losses = (loss_elem * valid).sum() / denom
        return losses

    @torch.inference_mode()
    def sample_actions_cfg(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor,
        action_hz: torch.Tensor,  # [B]
        embodiment_id: torch.Tensor,  # [B]
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, vlm_seq_len] VLM 的 attention_mask
        max_action_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        AML 推理：每步预测去噪动作 A_hat，换算 v_hat 后欧拉积分；从 t=0 噪声到 t=1 数据。
        CFG：cfg_scale = 1.0 或 None 时关闭，!= 1.0 时在动作空间做 CFG 融合。

        Args:
            h_t: 当前视觉特征 [B, num_vision_tokens, vision_dim]
            h_t1_star: 目标视觉特征 [B, num_vision_tokens, vision_dim] 
            h_vlm: VLM 特征 [B, vlm_dim]
            state: 本体状态特征 [B, state_dim]
            action_hz: 每个样本的控制频率 [B]
            cfg_scale: CFG 引导强度
            num_inference_steps: 推理步数
            max_action_horizon: 可选输出长度上限/对齐长度；若提供则输出会 pad 到该长度
            
        Returns:
            actions: 采样的动作序列 [B, T, action_dim]
        """
        device = h_t.device
        model_dtype = self._compute_dtype()
        h_t = self._cast_if_needed(h_t, model_dtype)
        h_t1_star = self._cast_if_needed(h_t1_star, model_dtype)
        h_vlm = self._cast_if_needed(h_vlm, model_dtype)
        state = self._cast_if_needed(state, model_dtype)
        batch_size = h_t.shape[0]
        action_hz_f = action_hz.to(device=device, dtype=torch.float32)
        if action_hz_f.ndim != 1 or action_hz_f.shape[0] != batch_size:
            raise ValueError(
                f"`action_hz` must have shape [B], got {tuple(action_hz_f.shape)} for batch_size={batch_size}."
            )
        n_from_hz = torch.floor(float(self.config.horizon_sec) * action_hz_f).to(dtype=torch.long)
        if torch.any(n_from_hz < 1):
            bad_idx = int(torch.nonzero(n_from_hz < 1, as_tuple=False)[0].item())
            raise ValueError(
                f"Invalid effective token count from horizon_sec*hz: sample={bad_idx}, "
                f"horizon_sec={self.config.horizon_sec}, hz={float(action_hz_f[bad_idx])}, "
                f"floor={int(n_from_hz[bad_idx])}. Increase `horizon_sec` or ensure hz>=1."
            )
        base_horizon = int(n_from_hz.max().item())
        if max_action_horizon is not None:
            max_action_horizon = int(max_action_horizon)
            if max_action_horizon <= 0:
                raise ValueError(f"`max_action_horizon` must be > 0, got {max_action_horizon}.")
            if base_horizon > max_action_horizon:
                raise ValueError(
                    f"Required horizon from hz ({base_horizon}) exceeds max_action_horizon ({max_action_horizon})."
                )
            action_horizon = max_action_horizon
        else:
            action_horizon = base_horizon
        t_grid, time_valid, _ = build_time_grid(
            horizon_sec=float(self.config.horizon_sec),
            hz=action_hz_f,
            seq_len=int(action_horizon),
        )
        # 默认从 config 读取（便于在 YAML 里通过 ConditionalFlowMatchingConfig 统一管理）
        if num_inference_steps is None:
            num_inference_steps = int(getattr(self.config, "num_steps", 50))
        if cfg_scale is None:
            cfg_scale = float(self.config.cfg_guidance_scale)
        # 初始化为纯噪声（t=0 的起点，即当前 A_t^tau）
        A_t_tau = float(self.config.noise_scale) * torch.randn(
            size=(batch_size, action_horizon, self.config.action_dim),
            dtype=model_dtype,
            device=device,
        )
        A_t_tau = A_t_tau * time_valid.unsqueeze(-1).to(dtype=A_t_tau.dtype)

        dt = 1.0 / float(num_inference_steps)

        # 编码条件特征（循环外，只需计算一次）
        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]
        if self.config.use_state:
            cond_state = self.enc_state(state)
        else:
            cond_state = None

        # 统一数据流：构建 encoder_hidden_states
        cond_encoder_hidden = torch.cat((h_t, h_t1_star, cond_vlm), dim=1)

        # 构造 encoder_attention_mask：视觉部分(h_t+h_t1)全关注 + VLM部分使用原始 attention_mask
        num_vision = h_t.shape[1] + h_t1_star.shape[1]  # 256 + 256 = 512
        if attention_mask is not None:
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            encoder_attention_mask = None

        # 修正CFG判断：只有当 cfg_scale 存在且 != 1.0 时才启用CFG
        # cfg_scale=1.0 时，CFG公式退化为纯条件预测，应避免计算无条件分支
        use_cfg = cfg_scale is not None and cfg_scale != 1.0
        if use_cfg:
            # 无条件分支：仅替换 h_t1_star
            uncond_encoder_hidden = torch.cat((
                h_t,
                self.cfg_embeddings.expand(batch_size, -1, -1),
                cond_vlm
            ), dim=1)

        # 如果使用 AlternateVLDiT，预先构建 masks
        if self.config.use_alternate_vldit:
            num_h_t = h_t.shape[1]
            num_h_t1 = h_t1_star.shape[1]
            num_vlm = cond_vlm.shape[1]

            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

        # AML 降噪循环：从 t=0 正向积分到 t=1（噪声→数据）；每步预测去噪 action，再换算 v_hat 做 Euler 更新
        for step in range(num_inference_steps):
            t_cont = step / float(num_inference_steps)  # 当前 tau
            # 离散化时间步，与训练时保持一致的方式
            t_discretized = int(t_cont * self.config.num_timestep_buckets)
            t_discretized = min(self.config.num_timestep_buckets - 1, max(0, t_discretized))

            # 编码当前噪声轨迹 A_t^tau
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
            )
            action_features = self.action_encoder(A_t_tau, embodiment_id)
            time_emb = self.time_encoder(t_grid).to(dtype=action_features.dtype)
            time_emb = time_emb * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)
            action_features = action_features + time_emb
            action_features = action_features * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)

            # 构建 hidden_states
            if self.config.use_state:
                hidden_states = torch.cat((cond_state, action_features), dim=1)
                state_token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                hidden_attention_mask = torch.cat([state_token_valid, time_valid], dim=1)
            else:
                hidden_states = action_features
                hidden_attention_mask = time_valid

            # 根据模式调用 DiT
            if self.config.use_alternate_vldit:
                # 条件预测：AML 下 decoder 输出为去噪动作 A_hat
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    hidden_attention_mask=hidden_attention_mask,
                    image_mask=image_mask,
                    vlm_mask=vlm_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                A_hat_cond_all = self.action_decoder(model_output_cond, embodiment_id)
                A_hat_cond = A_hat_cond_all[:, -action_horizon:, :]

                if use_cfg:
                    # 无条件预测
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        hidden_attention_mask=hidden_attention_mask,
                        image_mask=image_mask,
                        vlm_mask=vlm_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    A_hat_uncond_all = self.action_decoder(model_output_uncond, embodiment_id)
                    A_hat_uncond = A_hat_uncond_all[:, -action_horizon:, :]
                    A_hat = A_hat_uncond + cfg_scale * (A_hat_cond - A_hat_uncond)
                else:
                    A_hat = A_hat_cond
            else:
                # 标准 DiT
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    hidden_attention_mask=hidden_attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                A_hat_cond_all = self.action_decoder(model_output_cond, embodiment_id)
                A_hat_cond = A_hat_cond_all[:, -action_horizon:, :]

                if use_cfg:
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        hidden_attention_mask=hidden_attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    A_hat_uncond_all = self.action_decoder(model_output_uncond, embodiment_id)
                    A_hat_uncond = A_hat_uncond_all[:, -action_horizon:, :]
                    A_hat = A_hat_uncond + cfg_scale * (A_hat_cond - A_hat_uncond)
                else:
                    A_hat = A_hat_cond

            # 采样阶段使用更小 eps（而非训练 t_eps），避免最后一步残留噪声。（只有推理步数很多时，>20，才有意义）
            one_minus_tau = max(1.0 - t_cont, float(self.config.sample_eps))
            v_hat = (A_hat - A_t_tau) / one_minus_tau
            v_hat = v_hat * time_valid.unsqueeze(-1).to(dtype=v_hat.dtype)
            A_t_tau = A_t_tau + dt * v_hat
            A_t_tau = A_t_tau * time_valid.unsqueeze(-1).to(dtype=A_t_tau.dtype)

        return A_t_tau
