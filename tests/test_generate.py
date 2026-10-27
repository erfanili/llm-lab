import torch

from llm.evals import score
from llm.generate import generate
from llm.model import ModelConfig, Transformer

torch.manual_seed(0)
TINY = ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96, max_seq_len=64)


def greedy_reference(model, prompt, n):
    """No cache, no padding: rerun the full sequence each step."""
    seq = list(prompt)
    for _ in range(n):
        seq.append(model(torch.tensor([seq]))[0, -1].argmax().item())
    return seq[len(prompt):]


def test_cached_padded_batch_matches_reference():
    model = Transformer(TINY).eval()
    prompts = [[5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9], [42]]
    outs = generate(model, prompts, 12, temperature=0.0, stop_ids=(-1,))
    for p, o in zip(prompts, outs):
        assert o["tokens"] == greedy_reference(model, p, 12)
        assert o["finish"] == "length"


def test_stops_at_stop_token():
    model = Transformer(TINY).eval()
    ref = greedy_reference(model, [3, 4], 10)
    stop = ref[4]
    out = generate(model, [[3, 4]], 10, temperature=0.0, stop_ids=(stop,))[0]
    assert out["finish"] == "stop" and out["tokens"] == ref[: ref.index(stop)]


def test_seeded_sampling_reproducible():
    model = Transformer(TINY).eval()
    a = generate(model, [[1, 2]] * 4, 10, temperature=1.0, top_p=0.9, seed=3, stop_ids=(-1,))
    b = generate(model, [[1, 2]] * 4, 10, temperature=1.0, top_p=0.9, seed=3, stop_ids=(-1,))
    assert a == b


def test_score_matches_full_forward():
    model = Transformer(TINY).eval()
    reqs = [([0, 5, 6], [7, 8]), ([0, 1], [2, 3, 4, 5]), ([0], [9])]
    got = score(model, reqs, torch.device("cpu"), max_len=63)
    for (ctx, cont), (ll, _) in zip(reqs, got):
        seq = torch.tensor([ctx + cont])
        logp = model(seq[:, :-1]).log_softmax(-1)[0]
        ref = sum(logp[len(ctx) - 1 + i, t].item() for i, t in enumerate(cont))
        assert abs(ll - ref) < 1e-4
