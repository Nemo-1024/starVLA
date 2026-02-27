import torch.nn as nn
import torch
from typing import Optional, Tuple
from .modules import CategorySpecificMLP
from .pos_embs import Fixed2DPositionalEncoding


class LAMDecoder_v2(nn.Module):
    """
    通过堆叠多个 DecoderBlock，将动作应用到状态上，以重建下一帧的特征。
    """
    def __init__(self, context_dim, input_dim: int=1024, num_queries: int=1, num_layers=6, num_heads=16, dropout=0.1, grid_hw: Tuple[int, int] = (16, 20), train_in_latent: bool = True, ffn_expansion_factor=2, num_embodiments: int = 32, code_dim: Optional[int] = None):
        """
        初始化 LAMDecoder_v2。
        
        参数:
            context_dim (int): 状态/动作特征维度 (D_feat)。
            input_dim (int): 输入特征的维度 (D_feat)。
            num_layers (int): DecoderBlock 的堆叠层数。
            num_heads (int): 每个注意力模块的头数。
        """
        super().__init__()
        self.feature_dim = context_dim
        self.input_dim = input_dim
        self.train_in_latent = train_in_latent
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
            
        # 解码层
        # self.dec_layers = nn.ModuleList([
        #     CrossAttentionBlock(feature_dim, num_heads=num_heads, ffn_ratio=4, dropout=dropout) 
        #     for _ in range(num_layers)
        # ])
        self.num_queries = num_queries
        # if self.num_queries == 1:
        self.dec_layers = nn.ModuleList([nn.TransformerEncoderLayer(context_dim, num_heads, dim_feedforward=int(context_dim*ffn_expansion_factor), dropout=dropout, batch_first=True, norm_first=True) for _ in range(num_layers)])
        # else:
            # self.dec_layers = nn.ModuleList([Attn_Crossn_Block(feature_dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, dropout=dropout) for _ in range(num_layers)])
        self.pos_embed = Fixed2DPositionalEncoding(context_dim, self.grid_height, self.grid_width)
        # # Query self-attention增强
        # self.query_attn = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        # self.last_ln = nn.LayerNorm(input_dim)
        # 简化输入输出投影，避免冗余
        if input_dim == context_dim:
            self.project_input = nn.Identity()
            self.project_output = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, context_dim)
            self.project_output = nn.Linear(context_dim, input_dim)
        if code_dim is not None and code_dim != context_dim:
            self.action_in_proj = nn.Sequential(
                nn.Linear(code_dim, context_dim),
                nn.LayerNorm(context_dim),
            )
        else:
            # code_dim == context_dim 时也进行归一化，降低异常 latent 的注入冲击。
            self.action_in_proj = nn.LayerNorm(context_dim)
        if not train_in_latent:
            self.to_pixel = nn.ConvTranspose2d(input_dim, 3, kernel_size=16, stride=16)
        
    def forward(self, features, actions):
        """
        前向传播。
        
        参数:
            features (torch.Tensor): 初始帧的特征，形状 [B, 1, K, input_dim]。
            actions (torch.Tensor): VQ量化后的动作code，形状 [B, 1, node_dim]。
            states (torch.Tensor): 初始状态，形状 [B, 1, 8]。
            embodiment_id (torch.Tensor): [B] 或 [B,1]，选择 embodiment 特定参数（本模块未使用）。
            
        返回:
            torch.Tensor: 重建的最后一帧特征 f_hat_T，形状 [B, 1, K, input_dim]。
        """
        # 投影query并使用self-attention增强
        actions_tokens = self.action_in_proj(actions)  # [B, 1, feature_dim]
        
        # 投影输入特征（只调用一次，避免冗余）
        features_tokens = self.project_input(features)  # [B, 1, K, feature_dim] 或 [B, K, feature_dim]

        if features_tokens.dim() == 4:
            # 将单帧时间维压缩，Transformer 期望 3D: [B, S, E]
            features_tokens = features_tokens.squeeze(1)  # [B, K, feature_dim]
        expected_tokens = int(self.grid_height * self.grid_width)
        if features_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Decoder token mismatch: got K={features_tokens.shape[1]}, expected K={expected_tokens} "
                f"for grid_hw=({self.grid_height},{self.grid_width})."
            )
        features_tokens = self.pos_embed(features_tokens)
        if actions_tokens.shape[1] == 1:
            # 将动作条件加到每个空间token上，自动在K维广播
            x = features_tokens + actions_tokens  # [B, K, feature_dim]
            # x = torch.cat([features_tokens, actions_tokens], dim=1)
            # 通过解码层堆叠
            for layer in self.dec_layers:
                x = layer(x)
        else:
            x = torch.cat([features_tokens, actions_tokens], dim=1)
            # x = features_tokens
            for layer in self.dec_layers:
                x = layer(x)
        # 输出投影
        reconstructed_features = self.project_output(x[:, :features_tokens.shape[1]])  # [B, K, input_dim]
        # reconstructed_features = self.last_ln(reconstructed_features)
        # 统一返回形状为 [B, 1, K, *]
        if not self.train_in_latent:
            B, K, D = reconstructed_features.shape
            if K != expected_tokens:
                raise ValueError(
                    f"Decoder output token mismatch: got K={K}, expected K={expected_tokens} "
                    f"for grid_hw=({self.grid_height},{self.grid_width})."
                )
            rec_img = self.to_pixel(
                reconstructed_features.transpose(1, 2).reshape(B, D, self.grid_height, self.grid_width)
            )  # [B, 3, H, W]
            return rec_img.unsqueeze(1)  # [B, 1, 3, H, W]
        else:
            return reconstructed_features.unsqueeze(1)  # [B, 1, K, input_dim]


class StatePredictor(nn.Module):
    """
    物理接地末端执行器状态预测器 (Physical Grounding State Predictor)

    使用 embodiment 条件化的 CategorySpecific MLP 对状态进行编码与解码。

    输入:
        z_t: [B, num_queries, latent_dim]
        state_0: [B, 1, 8]
    输出:
        s_pred: [B, 8]  (与目标状态对应)
    """

    def __init__(
        self,
        latent_dim: int,
        dropout: float = 0.1,
        num_embodiments: int = 32,
        num_queries: int = 1,
        max_state_dim: int = 32,
        code_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_embodiments = int(num_embodiments)
        self.max_state_dim = int(max_state_dim)
        z_input_dim = int(code_dim) if code_dim is not None else int(latent_dim)
        self.z_encoder = nn.Linear(z_input_dim, latent_dim)
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.num_embodiments,
            input_dim=self.max_state_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
        )
        self.state_decoder = CategorySpecificMLP(
            num_categories=self.num_embodiments,
            input_dim=2 * latent_dim,
            hidden_dim=latent_dim,
            output_dim=self.max_state_dim,
        )

    def forward(self, z_t: torch.Tensor, state_0: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z_t: [B, num_queries, latent_dim]
            state_0: [B, 1, max_state_dim]
            embodiment_id: [B] 或 [B,1] 的 torch.Tensor
        Returns:
            s_pred: [B, max_state_dim]
        """
        B = state_0.size(0)
        if not isinstance(embodiment_id, torch.Tensor):
            raise TypeError(
                f"StatePredictor expects `embodiment_id` as torch.Tensor, got {type(embodiment_id).__name__}."
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
        emb = emb.to(device=state_0.device, dtype=torch.long)
        single_timestep = state_0.dim() == 3 and state_0.size(1) == 1
        if single_timestep:
            state_0 = state_0.squeeze(1)
        z_mean = z_t.mean(dim=1)
        expected_z_dim = int(self.z_encoder.in_features)
        if z_mean.size(-1) != expected_z_dim:
            raise ValueError(
                f"Unexpected z_t feature dim {z_mean.size(-1)}. Expected {expected_z_dim}."
            )
        z_embed = self.z_encoder(z_mean)
        state_embed = self.state_encoder(state_0.contiguous(), emb)
        if state_embed.dim() == 3 and z_embed.dim() == 2:
            z_embed = z_embed.unsqueeze(1).expand(-1, state_embed.size(1), -1)
        fused = torch.cat([z_embed, state_embed], dim=-1)
        s_pred = torch.tanh(self.state_decoder(fused, emb))
        if single_timestep and s_pred.dim() == 3 and s_pred.size(1) == 1:
            s_pred = s_pred.squeeze(1)
        return s_pred
