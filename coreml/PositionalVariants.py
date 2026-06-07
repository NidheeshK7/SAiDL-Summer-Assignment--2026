import torch
import torch.nn as nn
import math
class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
    
        self.register_buffer('pe', pe)

    def forward(self, x, start_pos = 0):
        seq_len = x.size(1)
        x = x + self.pe[:, start_pos : start_pos + seq_len, :].requires_grad_(False)
        return self.dropout(x)
    
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        seq_idx = torch.arange(max_seq_len, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        cos_cached = idx_theta2.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = idx_theta2.sin().unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.head_dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        num_tokens = q.size(2)

        cos = self.cos_cached[:, :, start_pos : start_pos + num_tokens, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + num_tokens, :]

        q_rope = (q * cos) + (self._neg_half(q) * sin)
        k_rope = (k * cos) + (self._neg_half(k) * sin)

        return q_rope, k_rope
    
    
class ALiBiPositionalBias(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        slopes = self._get_slopes(num_heads)
        
        self.register_buffer("slopes", slopes.view(1, num_heads, 1, 1), persistent=False)

    def _get_slopes(self, n: int):

        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            
            slopes = get_slopes_power_of_2(closest_power_of_2) + \
                     self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2].tolist()
                     
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, q_seq_len: int, k_seq_len: int, start_pos: int = 0, k_start_pos=0, device=None):

        q_indices = torch.arange(start_pos, start_pos + q_seq_len, device=device).unsqueeze(1)
        
        k_indices = torch.arange(k_start_pos, k_start_pos + k_seq_len, device=device).unsqueeze(0)
        
        relative_distances = k_indices - q_indices
        
        bias = self.slopes * relative_distances
        
        return bias
    
    
class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim: int, max_clip: int = 16):
        super().__init__()
        self.head_dim = head_dim
        self.max_clip = max_clip
        
        self.relative_key_embeddings = nn.Embedding(2 * self.max_clip + 1, head_dim)

    def forward(self, queries: torch.Tensor, q_seq_len: int, k_seq_len: int, start_pos: int = 0, k_start_pos: int=0):

        device = queries.device
        q_indices = torch.arange(start_pos, start_pos + q_seq_len, device=device).unsqueeze(1)
        k_indices = torch.arange(k_start_pos, k_start_pos + k_seq_len, device=device).unsqueeze(0)
        
        distance_matrix = k_indices - q_indices
        
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.max_clip, self.max_clip)
        
        distance_matrix_shifted = distance_matrix_clipped + self.max_clip
        
        rel_keys = self.relative_key_embeddings(distance_matrix_shifted)
        
        rel_attn_scores = torch.einsum('bhqd,qkd->bhqk', queries, rel_keys)
        
        return rel_attn_scores