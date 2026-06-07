import torch
import torch.nn as nn
import torch.nn.functional as F
from PositionalVariants import RotaryPositionalEmbedding, ALiBiPositionalBias, RelativePositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.pos_type = cfg.get("pos_type", "sinusoidal")
        
        if self.pos_type == "rope":
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096)
        elif self.pos_type == "alibi":
            self.alibi = ALiBiPositionalBias(self.num_heads)
        elif self.pos_type == "rel_pos":
            self.rel_pos = RelativePositionalEncoding(self.head_dim, max_clip=cfg.get("rel_clip", 16))

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x,  use_cache=False):
        b, num_tokens, d_in = x.shape
        device = x.device

        queries = self.W_query(x)
        keys_new = self.W_key(x)  
        values_new = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.pos_type == "rope":
            start_pos = self.ptr_current_pos if use_cache else 0
            queries, keys_new = self.rope(queries, keys_new, start_pos=start_pos)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys, values = self.cache_k, self.cache_v
            
            q_start_pos_abs = self.ptr_current_pos
            k_start_pos = 0
            self.ptr_current_pos += num_tokens
        else:
            keys, values = keys_new, values_new
            q_start_pos_abs = 0
            k_start_pos = 0
            self.ptr_current_pos = 0
            self.cache_k, self.cache_v = None, None

        num_tokens_Q = queries.shape[2]
        num_tokens_K = keys.shape[2]

        attn_scores = queries @ keys.transpose(2, 3)  

        if self.pos_type == "alibi":
            bias = self.alibi(q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos, device=device)
            attn_scores = attn_scores + bias

        elif self.pos_type == "rel_pos":
            bias = self.rel_pos(queries, q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos)
            attn_scores = attn_scores + bias

        q_positions = torch.arange(q_start_pos_abs, q_start_pos_abs + num_tokens_Q, device=device).unsqueeze(-1)
        k_positions = torch.arange(k_start_pos, k_start_pos + num_tokens_K, device=device).unsqueeze(0)
        mask_bool = q_positions < k_positions

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, cfg, d_in, d_out, dropout, num_heads, dtype=None, qkv_bias=False
    ):
        super().__init__()
        num_kv_groups = cfg.get("num_kv_groups", 4)
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.pos_type = cfg.get("pos_type", "sinusoidal")
        
        if self.pos_type == "rope":
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096)
        elif self.pos_type == "alibi":
            self.alibi = ALiBiPositionalBias(self.num_heads)
        elif self.pos_type == "rel_pos":
            self.rel_pos = RelativePositionalEncoding(self.head_dim, max_clip=cfg.get("rel_clip", 16))

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        
    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape
        device = x.device

        queries = self.W_query(x)
        keys = self.W_key(x)      
        values = self.W_value(x)   

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.pos_type == "rope":
            start_pos = self.ptr_current_pos if use_cache else 0
            queries, keys_new = self.rope(queries, keys_new, start_pos=start_pos)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys_base, values_base = self.cache_k, self.cache_v
            
            q_start_pos_abs = self.ptr_current_pos
            k_start_pos = 0
            self.ptr_current_pos += num_tokens
        else:
            keys_base, values_base = keys_new, values_new
            q_start_pos_abs = 0
            k_start_pos = 0
            self.ptr_current_pos = 0
            self.cache_k, self.cache_v = None, None

        keys = keys_base.repeat_interleave(self.group_size, dim=1)
        values = values_base.repeat_interleave(self.group_size, dim=1)

        num_tokens_Q = queries.shape[2]
        num_tokens_K = keys.shape[2]

        attn_scores = queries @ keys.transpose(2, 3)

        if self.pos_type == "alibi":
            bias = self.alibi(q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos, device=device)
            attn_scores = attn_scores + bias

        elif self.pos_type == "rel_pos":
            bias = self.rel_pos(queries, q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos)
            attn_scores = attn_scores + bias

        q_positions = torch.arange(q_start_pos_abs, q_start_pos_abs + num_tokens_Q, device=device).unsqueeze(-1)
        k_positions = torch.arange(k_start_pos, k_start_pos + num_tokens_K, device=device).unsqueeze(0)
        mask = q_positions < k_positions

        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(self, cfg, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.pos_type = cfg.get("pos_type", "sinusoidal")
        
        if self.pos_type == "rope":
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096)
        elif self.pos_type == "alibi":
            self.alibi = ALiBiPositionalBias(self.num_heads)
        elif self.pos_type == "rel_pos":
            self.rel_pos = RelativePositionalEncoding(self.head_dim, max_clip=cfg.get("rel_clip", 16))

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.sliding_window_size = cfg.get("sliding_window_size", 128)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape
        device = x.device

        queries = self.W_query(x)
        keys_new = self.W_key(x)  
        values_new = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.pos_type == "rope":
            start_pos = self.ptr_current_pos if use_cache else 0
            queries, keys_new = self.rope(queries, keys_new, start_pos=start_pos)

        if use_cache:
            old_len = 0 if self.cache_k is None else self.cache_k.size(2)
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)

            if self.sliding_window_size is not None:
                if self.cache_k.size(2) > self.sliding_window_size:
                    self.cache_k = self.cache_k[:, :, -self.sliding_window_size:, :]
                    self.cache_v = self.cache_v[:, :, -self.sliding_window_size:, :]

            keys, values = self.cache_k, self.cache_v

            total_len = old_len + num_tokens
            k_len_now = self.cache_k.size(2)
            dropped = max(0, total_len - k_len_now)
            
            k_start_pos = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
            self.ptr_current_pos += num_tokens
        else:
            keys, values = keys_new, values_new
            q_start_pos_abs = 0
            k_start_pos = 0
            self.ptr_current_pos = 0
            self.cache_k, self.cache_v = None, None

        num_tokens_Q = queries.shape[2]
        num_tokens_K = keys.shape[2]

        attn_scores = queries @ keys.transpose(2, 3)

        if self.pos_type == "alibi":
            bias = self.alibi(q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos, device=device)
            attn_scores = attn_scores + bias

        elif self.pos_type == "rel_pos":
            bias = self.rel_pos(queries, q_seq_len=num_tokens_Q, k_seq_len=num_tokens_K, start_pos=q_start_pos_abs, k_start_pos=k_start_pos)
            attn_scores = attn_scores + bias

        q_positions = torch.arange(q_start_pos_abs, q_start_pos_abs + num_tokens_Q, device=device).unsqueeze(-1)
        k_positions = torch.arange(k_start_pos, k_start_pos + num_tokens_K, device=device).unsqueeze(0)

        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)

        diff = q_positions - k_positions
        mask_bool = (diff < 0) | (diff >= W)

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
        
        
        
class LinearAttention(nn.Module):
    def __init__(self, cfg, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.pos_type = cfg.get("pos_type", "sinusoidal")

        if self.pos_type in ["alibi", "rel_pos"]:
            raise ValueError("Linear Attention cannot be used with ALiBi or Relative Positional Encodings.")

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        if self.pos_type == "rope":
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096)

        self.register_buffer("state_s", None, persistent=False)
        self.register_buffer("state_z", None, persistent=False)
        self.ptr_current_pos = 0

    def feature_map(self, x):
        return torch.clamp_min(torch.nn.functional.elu(x) + 1.0, 1e-4)

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape
        device = x.device

        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if self.pos_type == "rope":
            start_pos = self.ptr_current_pos if use_cache else 0
            queries, keys = self.rope(queries, keys, start_pos=start_pos)

        original_dtype = queries.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            queries = queries.to(torch.float32)
            keys = keys.to(torch.float32)
            values = values.to(torch.float32)

            scale_factor = self.head_dim ** -0.25
            queries = queries * scale_factor
            keys = keys * scale_factor

            Q_phi = self.feature_map(queries)
            K_phi = self.feature_map(keys)

            KV_chunk = torch.einsum('bhtd,bhte->bhtde', K_phi, values)

            if use_cache:
                S_chunk = torch.cumsum(KV_chunk, dim=2)
                Z_chunk = torch.cumsum(K_phi, dim=2)

                if self.state_s is None:
                    S_total = S_chunk
                    Z_total = Z_chunk
                else:
                    S_total = self.state_s.unsqueeze(2) + S_chunk
                    Z_total = self.state_z.unsqueeze(2) + Z_chunk

                self.state_s = S_total[:, :, -1, :, :].clone()
                self.state_z = Z_total[:, :, -1, :].clone()
                self.ptr_current_pos += num_tokens

                num = torch.einsum('bhtd,bhtde->bhte', Q_phi, S_total)
                den = torch.einsum('bhtd,bhtd->bht', Q_phi, Z_total).unsqueeze(-1)

            else:
                S_total = torch.cumsum(KV_chunk, dim=2)
                Z_total = torch.cumsum(K_phi, dim=2)
                num = torch.einsum('bhtd,bhtde->bhte', Q_phi, S_total)
                den = torch.einsum('bhtd,bhtd->bht', Q_phi, Z_total).unsqueeze(-1)
                self.ptr_current_pos = 0
                self.state_s, self.state_z = None, None

            den = den + 1e-6
            attn_out = num / den 
            attn_out = torch.clamp(attn_out, min=-65000.0, max=65000.0)

        attn_out = attn_out.to(original_dtype)
        attn_out = self.dropout(attn_out)

        context_vec = attn_out.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.state_s, self.state_z = None, None
        self.ptr_current_pos = 0
        
class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,   
            bias=False,
            padding=0,
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)

        return x.transpose(1, 2)
 
 
class CausalConvModule(nn.Module):
    def __init__(self, emb_dim: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(emb_dim)
        self.pointwise_expand = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.causal_dw_conv = CausalDepthwiseConv1d(emb_dim, kernel_size)
        self.conv_norm = nn.LayerNorm(emb_dim)
        self.activation = nn.SiLU()
        self.pointwise_project = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pointwise_expand(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = self.causal_dw_conv(x)
        x = self.conv_norm(x)
        x = self.activation(x)
        x = self.pointwise_project(x)
        return residual + self.dropout(x)

