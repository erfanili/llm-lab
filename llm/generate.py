"""Batched sampling with a KV cache.

Prompts of different lengths are left-padded so every sequence's last prompt token sits in the
same column. Padding is excluded from attention, and RoPE positions count only real tokens, so
each row produces exactly what it would if run alone.
"""

import torch

from llm.model import KVCache, Transformer


def attention_mask(valid: torch.Tensor, q_start: int, n_q: int) -> torch.Tensor:
    """(B, S) key validity -> (B, 1, n_q, S) bool mask for queries at columns q_start..q_start+n_q-1."""
    S = valid.shape[1]
    q = torch.arange(q_start, q_start + n_q, device=valid.device)[:, None]
    k = torch.arange(S, device=valid.device)[None, :]
    allowed = (k <= q)[None] & valid[:, None, :]
    allowed |= (k == q)[None]  # a padding query attends to itself only, so no row is fully masked (NaN)
    return allowed[:, None]


def sample_next(logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
                generator: torch.Generator | None) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = logits.sort(-1, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        drop = cum - sorted_logits.softmax(-1) > top_p  # keep the smallest prefix with mass >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator)[:, 0]


@torch.no_grad()
def generate(model: Transformer, prompts: list[list[int]], max_new_tokens: int, *, temperature: float = 1.0,
             top_p: float = 1.0, top_k: int = 0, stop_ids: tuple[int, ...] = (0,), seed: int | None = None,
             device=None, amp: bool = True) -> list[dict]:
    """Returns per prompt {"tokens": generated ids (stop token excluded), "finish": "stop" | "length"}.
    amp=True runs bf16 autocast on CUDA (fast, for training loops); evals pass amp=False for fp32."""
    device = device or next(model.parameters()).device
    max_prompt = model.cfg.max_seq_len - max_new_tokens
    assert max_prompt > 0
    prompts = [p[-max_prompt:] for p in prompts]
    B, P = len(prompts), max(len(p) for p in prompts)
    idx = torch.zeros(B, P, dtype=torch.long, device=device)
    valid = torch.zeros(B, P, dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        idx[i, P - len(p):] = torch.tensor(p, device=device)
        valid[i, P - len(p):] = True
    pos = (valid.long().cumsum(1) - 1).clamp(min=0)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    use_amp = amp and idx.is_cuda
    cache = KVCache(model.cfg, B, P + max_new_tokens, device, torch.bfloat16 if use_amp else torch.float32)
    stop = torch.tensor(stop_ids, device=device)
    out = torch.zeros(B, max_new_tokens, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), max_new_tokens, device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(idx, pos=pos, attn_mask=attention_mask(valid, 0, P), cache=cache)[:, -1]
        next_pos = pos[:, -1] + 1
        for t in range(max_new_tokens):
            nxt = sample_next(logits, temperature, top_p, top_k, gen)
            out[:, t] = nxt
            just_stopped = torch.isin(nxt, stop) & ~done
            lengths[just_stopped] = t
            done |= just_stopped
            if done.all() or t == max_new_tokens - 1:
                break
            valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=device)], 1)
            logits = model(nxt[:, None], pos=next_pos[:, None], attn_mask=attention_mask(valid, cache.length, 1),
                           cache=cache)[:, -1]
            next_pos = next_pos + 1

    out, lengths, done = out.tolist(), lengths.tolist(), done.tolist()
    return [dict(tokens=o[:n], finish="stop" if d else "length") for o, n, d in zip(out, lengths, done)]
