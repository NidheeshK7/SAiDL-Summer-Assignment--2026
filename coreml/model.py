import torch
import torch.nn as nn
from PositionalVariants import SinusoidalEncoding
from AttentionVariants import MultiHeadAttention, GroupedQueryAttention, MultiHeadAttentionWithSWA, LinearAttention
from AFTvariants import AFTSimple, AFTFull, AFTLocal, AFTConv
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        attn_type = cfg.get("attn_type", "mha")
        d_in = cfg["emb_dim"]
        d_out = cfg["emb_dim"]
        num_heads = cfg["n_heads"]
        dropout = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        
        if attn_type == "mha":
            self.att = MultiHeadAttention(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        elif attn_type == "gqa":
            self.att = GroupedQueryAttention(cfg, d_in, d_out, dropout, num_heads, qkv_bias=qkv_bias)
        elif attn_type == "swa":
            self.att = MultiHeadAttentionWithSWA(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        elif attn_type == "linear":
            self.att = LinearAttention(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
            
        elif attn_type == "aft_simple":
            self.att = AFTSimple(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        elif attn_type == "aft_full":
            self.att = AFTFull(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        elif attn_type == "aft_local":
            self.att = AFTLocal(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        elif attn_type == "aft_conv":
            self.att = AFTConv(cfg, d_in, d_out, dropout, num_heads, qkv_bias)
        else:
            raise ValueError(f"Unsupported attn_type in config: {attn_type}")
           
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut  

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        self.pos_type = cfg.get("pos_type", "sinusoidal")
        if self.pos_type == "sinusoidal":
            self.pos_emb = SinusoidalEncoding(cfg["emb_dim"], seq_len=4096, dropout=0.0)
        else:
            self.pos_emb = None

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.current_pos = 0

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx)

        if self.pos_type == "sinusoidal":
            start_pos = self.current_pos if use_cache else 0
            x = self.pos_emb(x, start_pos=start_pos)
            
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x, use_cache=use_cache)
        if use_cache:
            self.current_pos += seq_len
        else:
            self.current_pos = 0

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0



def generate_text_simple_cached(model, idx, max_new_tokens, context_size, use_cache=True):
    model.eval()

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -context_size:], use_cache=True)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -context_size:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


