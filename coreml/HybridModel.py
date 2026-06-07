import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from model import GPTModel, LayerNorm, FeedForward
from PositionalVariants import SinusoidalEncoding
from AttentionVariants import (
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadAttentionWithSWA,
    LinearAttention,
    CausalConvModule,          
)

class HybridTransformerBlock(nn.Module):
    def __init__(self, cfg: dict, hybrid_type: str = "interleaved"):
        super().__init__()

        if hybrid_type not in ("interleaved", "gated_ffn"):
            raise ValueError(
                f"hybrid_type must be 'interleaved' or 'gated_ffn', got '{hybrid_type}'"
            )

        self.hybrid_type = hybrid_type

        emb_dim     = cfg["emb_dim"]
        num_heads   = cfg["n_heads"]
        dropout     = cfg["drop_rate"]
        qkv_bias    = cfg["qkv_bias"]
        attn_type   = cfg.get("attn_type", "mha")
        kernel_size = cfg.get("conv_kernel_size", 7)

        if attn_type == "mha":
            self.att = MultiHeadAttention(
                cfg, emb_dim, emb_dim, dropout, num_heads, qkv_bias)
        elif attn_type == "gqa":
            self.att = GroupedQueryAttention(
                cfg, emb_dim, emb_dim, dropout, num_heads, qkv_bias=qkv_bias)
        elif attn_type == "swa":
            self.att = MultiHeadAttentionWithSWA(
                cfg, emb_dim, emb_dim, dropout, num_heads, qkv_bias)
        elif attn_type == "linear":
            self.att = LinearAttention(
                cfg, emb_dim, emb_dim, dropout, num_heads, qkv_bias)
        else:
            raise ValueError(f"Unsupported attn_type in config: '{attn_type}'")

        self.norm1 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(dropout)
        self.causal_conv = CausalConvModule(
            emb_dim=emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        if hybrid_type == "interleaved":
            self.ff    = FeedForward(cfg)
            self.norm2 = LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if self.hybrid_type == "interleaved":
            return self._forward_interleaved(x, use_cache)
        else:
            return self._forward_gated_ffn(x, use_cache)

    def _forward_interleaved(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x) + shortcut
        x = self.causal_conv(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x) + shortcut

        return x

    def _forward_gated_ffn(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x) + shortcut
        x = self.causal_conv(x)

        return x

    def reset_cache(self):
        self.att.reset_cache()


class HybridGPTModel(GPTModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        hybrid_type = cfg.get("hybrid_type", "interleaved")
        self.trf_blocks = nn.ModuleList(
            [HybridTransformerBlock(cfg, hybrid_type=hybrid_type)
             for _ in range(cfg["n_layers"])]
        )

    def forward(self, in_idx: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx)
        if self.pos_type == "sinusoidal":
            x = self.pos_emb(x, start_pos=0)

        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x, use_cache=False)
                
        self.current_pos = 0

        x = self.final_norm(x)
        return self.out_head(x)