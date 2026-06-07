import torch
import torch.nn as nn
import torch.nn.functional as F

class AFTSimple(nn.Module):
    def __init__(self, cfg: dict, d_in: int, d_out: int, dropout: float,
                 num_heads: int = 1, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        original_dtype = Q.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)

            max_K = K.max(dim=1, keepdim=True).values
            exp_K = torch.exp(K - max_K)
            exp_KV = exp_K * V

            num = torch.cumsum(exp_KV, dim=1)
            den = torch.cumsum(exp_K, dim=1)

            Y = torch.sigmoid(Q) * (num / (den + 1e-9))

        Y = Y.to(original_dtype)

        return self.dropout(self.out_proj(Y))

    def reset_cache(self):
        pass

class AFTFull(nn.Module):
    def __init__(self, cfg: dict, d_in: int, d_out: int, dropout: float,
                 num_heads: int = 1, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.max_seq_len = cfg.get("train_context_length", 512)
        self.bias_dim = cfg.get("aft_bias_dim", 128)

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.pos_bias_u = nn.Parameter(torch.zeros(self.max_seq_len, self.bias_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.max_seq_len, self.bias_dim))
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

    def _get_causal_exp_bias(self, T: int, device: torch.device) -> torch.Tensor:
        U = self.pos_bias_u[:T]
        V = self.pos_bias_v[:T]
        w = U @ V.T

        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        w = w.masked_fill(causal_mask, float('-inf'))

        return torch.exp(w)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        original_dtype = Q.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)

            exp_w = self._get_causal_exp_bias(T, x.device).to(torch.float32)

            max_K = K.max(dim=1, keepdim=True).values
            exp_K = torch.exp(K - max_K)
            exp_KV = exp_K * V

            num = torch.einsum('ij,bjd->bid', exp_w, exp_KV)
            den = torch.einsum('ij,bjd->bid', exp_w, exp_K)

            Y = torch.sigmoid(Q) * (num / (den + 1e-9))

        Y = Y.to(original_dtype)

        return self.dropout(self.out_proj(Y))

    def reset_cache(self):
        pass

class AFTLocal(nn.Module):
    def __init__(self, cfg: dict, d_in: int, d_out: int, dropout: float,
                 num_heads: int = 1, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.max_seq_len = cfg.get("train_context_length", 512)
        self.bias_dim = cfg.get("aft_bias_dim", 128)
        self.window_size = cfg.get("aft_local_window", 32)

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.pos_bias_u = nn.Parameter(torch.zeros(self.max_seq_len, self.bias_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.max_seq_len, self.bias_dim))
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

    def _get_causal_exp_bias(self, T: int, device: torch.device) -> torch.Tensor:
        U = self.pos_bias_u[:T]
        V = self.pos_bias_v[:T]
        w = U @ V.T

        idx = torch.arange(T, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

        outside_window = dist >= self.window_size
        w = w.masked_fill(outside_window, 0.0)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        w = w.masked_fill(causal_mask, float('-inf'))

        return torch.exp(w)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        original_dtype = Q.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)

            exp_w = self._get_causal_exp_bias(T, x.device).to(torch.float32)

            max_K = K.max(dim=1, keepdim=True).values
            exp_K = torch.exp(K - max_K)
            exp_KV = exp_K * V

            num = torch.einsum('ij,bjd->bid', exp_w, exp_KV)
            den = torch.einsum('ij,bjd->bid', exp_w, exp_K)

            Y = torch.sigmoid(Q) * (num / (den + 1e-9))

        Y = Y.to(original_dtype)

        return self.dropout(self.out_proj(Y))

    def reset_cache(self):
        pass

class AFTConv(nn.Module):
    def __init__(self, cfg: dict, d_in: int, d_out: int, dropout: float,
                 num_heads: int = 1, qkv_bias: bool = False):
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
            )

        self.d_out       = d_out
        self.num_heads   = num_heads
        self.head_dim    = d_out // num_heads
        self.kernel_size = cfg.get("aft_conv_kernel", 16)
        self.causal_pad  = self.kernel_size - 1

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.W_k = nn.Linear(d_in, num_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.w = nn.Parameter(torch.zeros(num_heads, self.kernel_size))

        self.gamma = nn.Parameter(torch.zeros(num_heads))
        self.beta = nn.Parameter(torch.zeros(num_heads))

    def _get_conv_filter(self) -> torch.Tensor:
        w_mean = self.w.mean(dim=-1, keepdim=True)
        w_std = self.w.std(dim=-1, keepdim=True) + 1e-6
        w_norm = (self.w - w_mean) / w_std

        gamma = self.gamma.unsqueeze(-1)
        beta = self.beta.unsqueeze(-1)
        w_reparam = gamma * w_norm + beta

        return torch.exp(w_reparam) - 1.0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        h = self.num_heads
        d_head = self.head_dim
        s = self.kernel_size

        Q = self.W_q(x).view(B, T, h, d_head)
        V = self.W_v(x).view(B, T, h, d_head)
        K = self.W_k(x)

        original_dtype = Q.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)

            filt = self._get_conv_filter().to(torch.float32)

            max_K = K.max(dim=1, keepdim=True).values
            exp_K = torch.exp(K - max_K)

            exp_KV = exp_K.unsqueeze(-1) * V

            exp_KV_conv = exp_KV.permute(0, 2, 3, 1).reshape(B, h * d_head, T)
            exp_KV_padded = F.pad(exp_KV_conv, (self.causal_pad, 0))

            filt_KV = filt.unsqueeze(1).expand(h, d_head, s).contiguous().reshape(h * d_head, 1, s)

            num_local_flat = F.conv1d(exp_KV_padded, filt_KV, groups=h * d_head)
            num_local = num_local_flat.view(B, h, d_head, T).permute(0, 3, 1, 2)

            exp_K_conv = exp_K.permute(0, 2, 1)
            exp_K_padded = F.pad(exp_K_conv, (self.causal_pad, 0))

            filt_K = filt.unsqueeze(1).contiguous()
            den_local_flat = F.conv1d(exp_K_padded, filt_K, groups=h)
            den_local = den_local_flat.permute(0, 2, 1).unsqueeze(-1)

            num_global = torch.cumsum(exp_KV, dim=1)
            den_global = torch.cumsum(exp_K, dim=1).unsqueeze(-1)

            num = num_local + num_global
            den = den_local + den_global

            Y = torch.sigmoid(Q) * (num / (den + 1e-9))
            Y = Y.reshape(B, T, D)

        Y = Y.to(original_dtype)

        return self.dropout(self.out_proj(Y))

    def reset_cache(self):
        pass