"""Llama-style decoder-only transformer, written out by hand.

Architecture: pre-norm RMSNorm, rotary position embeddings (RoPE), grouped-query
attention (GQA), SwiGLU MLP, tied input/output embeddings, no biases.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1536
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability, then cast back to the activation dtype.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) by angle pos * inv_freq[i].

    x: (B, H, T, D); cos/sin: (T, D/2), or (B, 1, T, D/2) for per-sequence positions.
    """
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask=None, cache=None, layer=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)  # all keys/values so far, incl. this chunk
        # GQA: each group of n_heads / n_kv_heads query heads shares one K/V head.
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if attn_mask is None:
            assert cache is None, "cached attention needs an explicit mask"
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None, cache=None, layer=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, cache, layer)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # GPT-2 style: N(0, 0.02) everywhere, residual output projections scaled
        # by 1/sqrt(2 * n_layers) so the residual stream variance doesn't grow with depth.
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue  # RMSNorm gains stay at 1
            s = std / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("wo.weight", "w_down.weight")) else std
            nn.init.normal_(p, mean=0.0, std=s)

    def hidden(self, idx, pos=None, attn_mask=None, cache=None):
        """Final normed hidden states (B, T, d_model).

        Default (pretraining) path: positions 0..T-1, causal attention via the fused kernel.
        For padded batches / incremental decoding pass
          pos:       (B, T) absolute position of each token (for RoPE),
          attn_mask: (B, 1, T, S) bool, True where query t may attend key s (S = cached + T),
          cache:     KVCache, appended to in place.
        """
        T = idx.shape[1]
        if pos is None:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        else:
            cos, sin = self.rope_cos[pos].unsqueeze(1), self.rope_sin[pos].unsqueeze(1)
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, cache, i)
        if cache is not None:
            cache.length += T
        return self.norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, **kw):
        logits = self.lm_head(self.hidden(idx, **kw))
        if targets is None:
            return logits
        # Cross-entropy in fp32: bf16 logits lose too much precision over a 32k vocab.
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs per token (fwd + bwd ~= 3x fwd), PaLM appendix B formula."""
        c = self.cfg
        return 6 * self.num_params(non_embedding=True) + 6 * c.vocab_size * c.d_model + 12 * c.n_layers * c.d_model * seq_len


class KVCache:
    """Preallocated per-layer keys/values for incremental decoding."""

    def __init__(self, cfg: ModelConfig, batch: int, max_len: int, device, dtype):
        shape = (cfg.n_layers, batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # advanced by Transformer.hidden after all layers have written

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        end = self.length + k.shape[2]
        self.k[layer, :, :, self.length : end] = k
        self.v[layer, :, :, self.length : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]
