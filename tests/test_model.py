import math

import torch
import torch.nn.functional as F

from llm.model import ModelConfig, Transformer, apply_rope, rope_cache

torch.manual_seed(0)
TINY = ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96, max_seq_len=32)


def test_rope_matches_complex_reference():
    # RoPE rotates each (even, odd) pair as a complex number by e^{i * pos * freq}.
    B, H, T, D = 2, 3, 16, 8
    x = torch.randn(B, H, T, D)
    cos, sin = rope_cache(T, D, 10000.0)
    freqs = torch.polar(torch.ones_like(cos), torch.atan2(sin, cos))
    ref = torch.view_as_real(torch.view_as_complex(x.view(B, H, T, D // 2, 2)) * freqs).flatten(-2)
    torch.testing.assert_close(apply_rope(x, cos, sin), ref)


def test_rope_is_relative():
    # <rope(q, m), rope(k, n)> depends only on m - n.
    D = 8
    cos, sin = rope_cache(32, D, 10000.0)
    q, k = torch.randn(D), torch.randn(D)
    def at(v, pos):
        seq = torch.zeros(1, 1, 32, D)
        seq[0, 0, pos] = v
        return apply_rope(seq, cos, sin)[0, 0, pos]
    torch.testing.assert_close(at(q, 5) @ at(k, 2), at(q, 20) @ at(k, 17), rtol=1e-4, atol=1e-5)


def test_attention_matches_naive():
    m = Transformer(TINY)
    attn = m.blocks[0].attn
    x = torch.randn(2, 10, TINY.d_model)
    B, T, _ = x.shape
    q = attn.wq(x).view(B, T, attn.n_heads, -1).transpose(1, 2)
    k = attn.wk(x).view(B, T, attn.n_kv_heads, -1).transpose(1, 2)
    v = attn.wv(x).view(B, T, attn.n_kv_heads, -1).transpose(1, 2)
    q, k = apply_rope(q, m.rope_cos, m.rope_sin), apply_rope(k, m.rope_cos, m.rope_sin)
    rep = attn.n_heads // attn.n_kv_heads
    out = []
    for h in range(attn.n_heads):
        kh, vh = k[:, h // rep], v[:, h // rep]  # query head h reads kv head h // rep
        s = q[:, h] @ kh.transpose(-1, -2) / math.sqrt(attn.head_dim)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        out.append(F.softmax(s, -1) @ vh)
    ref = attn.wo(torch.cat(out, -1))
    torch.testing.assert_close(attn(x, m.rope_cos, m.rope_sin), ref, rtol=1e-5, atol=1e-5)


def test_causal():
    m = Transformer(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 20))
    idx2 = idx.clone()
    idx2[0, 12:] = torch.randint(0, TINY.vocab_size, (8,))
    torch.testing.assert_close(m(idx)[:, :12], m(idx2)[:, :12])


def test_initial_loss_near_uniform():
    m = Transformer(TINY)
    idx, tgt = torch.randint(0, TINY.vocab_size, (2, 4, 32))
    _, loss = m(idx, tgt)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.1


def test_can_overfit_one_batch():
    m = Transformer(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, TINY.vocab_size, (2, 33))
    for _ in range(200):
        _, loss = m(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_tied_embeddings():
    m = Transformer(TINY)
    assert m.lm_head.weight is m.embed.weight
