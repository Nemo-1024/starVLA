import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torchvision import transforms

from einops import repeat
from .pos_embs import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, Fixed3DPositionalEncoding, PositionalEncoding
from .dpt_head import DPTHead
from .modules import QFormer, QFormer_att, CategorySpecificMLP


class LAMEncoder(nn.Module):
    def __init__(self, context_dim: int, input_dim: int=1024,
                 num_layers: int=4, num_heads: int=16, ffn_expansion_factor=4,
                 dropout: float = 0.0,  num_frames: int=5, num_queries: int=1, grid_hw: Tuple[int, int] = (16, 20), patch_size: int = 16, add_state: bool = False, modal_mask: bool = False, max_state_dim: int = 32, num_embodiments: int = 32, code_dim: Optional[int] = None):
        super().__init__()
        self.num_frames = num_frames
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
        self.patch_size = patch_size
        self.context_dim = context_dim
        self.max_state_dim = max_state_dim
        self.num_embodiments = int(num_embodiments)
        if input_dim != context_dim:
            self.project_in = nn.Linear(input_dim, context_dim)
        else:
            self.project_in = nn.Identity()
        if code_dim != context_dim:
            self.out_proj = nn.Linear(context_dim, code_dim)
        else:
            self.out_proj = nn.Identity()

        self.pos_embed = Fixed3DPositionalEncoding(context_dim, num_frames, self.grid_height, self.grid_width)
        if add_state:
            self.pos_state_embed = PositionalEncoding(context_dim)
            self.state_project = CategorySpecificMLP(
                num_categories=self.num_embodiments,
                input_dim=max_state_dim,
                hidden_dim=context_dim,
                output_dim=context_dim,
            )
        else:
            self.pos_state_embed = None
            self.state_project = None
        self.add_state = add_state
        # 3) QFormer：从上下文中提取 latent actions
        self.QFormer = QFormer_att(
            query_dim=context_dim,
            context_dim=context_dim,
            num_frames=num_frames,
            num_queries=num_queries,
            grid_hw=(self.grid_height, self.grid_width),
            add_tokens=1 if add_state else 0,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout=dropout,
            use_mask=modal_mask   #启用mask
        )
        
    def forward(
        self,
        features: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入:
          - features: DINO 隐状态序列（Tensor）
              支持形状 [B, T, K, D]（K=grid_size^2）
          - states:   [B, T, 1, 8] 或 [B, T, 8]；当 add_state=False 时可为 None
        输出:
          - latents:  [B, num_queries, context_dim]
        """
        if features.dim() != 4:
            raise ValueError(f"features 期望为 [B,T,K,D]，得到: {tuple(features.shape)}")
        B, T = int(features.shape[0]), int(features.shape[1])
        expected_tokens = int(self.grid_height * self.grid_width)
        if features.shape[-2] != expected_tokens:
            raise ValueError(
                f"features token count mismatch: got {features.shape[-2]}, expected {expected_tokens} "
                f"for grid_hw=({self.grid_height},{self.grid_width})."
            )
        x_ctx = self.project_in(features)
        # print(x_ctx.shape)
        x_ctx = self.pos_embed(x_ctx)  #[B, T, hw, D]
        # breakpoint()
        if self.add_state:
            if states is None:
                raise ValueError("states 不能为空：LAMEncoder(add_state=True) 需要状态输入。")
            # 统一 states 形状到 [B, T, max_state_dim]，以匹配 AC predictor 的输入
            if states.dim() == 4 and states.size(-2) == 1 and states.size(-1) == self.max_state_dim:
                states = states.squeeze(-2)
            elif states.dim() == 3 and states.size(-1) == self.max_state_dim:
                pass
            else:
                raise ValueError(
                    f"states 期望为 [B,T,max_state_dim] 或 [B,T,1,max_state_dim]，得到: {tuple(states.shape)}"
                )
            if int(states.shape[0]) != B or int(states.shape[1]) != T:
                raise ValueError(
                    f"states 与 features 的批次/时间维不一致: states={tuple(states.shape)} vs features={tuple(features.shape)}"
                )
            if embodiment_id is None:
                raise ValueError("LAMEncoder(add_state=True) requires `embodiment_id`.")
            if not isinstance(embodiment_id, torch.Tensor):
                raise TypeError(
                    f"LAMEncoder expects `embodiment_id` as torch.Tensor, got {type(embodiment_id).__name__}."
                )
            emb = embodiment_id
            if emb.ndim == 2 and emb.size(1) == 1:
                emb = emb.squeeze(1)
            elif emb.ndim != 1:
                raise ValueError(
                    f"`embodiment_id` must be [B] or [B,1], got {tuple(emb.shape)}"
                )
            if emb.shape[0] != B:
                raise ValueError(
                    f"embodiment_id batch mismatch: got {emb.shape[0]}, expected {B}"
                )
            emb = emb.to(device=states.device, dtype=torch.long)
            states = self.pos_state_embed(self.state_project(states, emb))
        # 约定后置拼接：每帧为 [image..., state]
            x_ctx = torch.cat([x_ctx, states.unsqueeze(-2)], dim=-2) #[B, T, (hw+1), D]
        # 3) QFormer: 从上下文中提取 latent actions
        latents = self.QFormer(x_ctx)       # [B, num_queries, context_dim]

        return self.out_proj(latents)
