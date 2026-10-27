import torch

from rlhf import gae


def gae_reference(r, v, gamma, lam):
    """Textbook GAE on a single trajectory (lists), terminal after the last step."""
    adv, last = [0.0] * len(r), 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(r) else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def test_gae_matches_reference_with_padding():
    torch.manual_seed(0)
    B, T = 3, 12
    rewards, values = torch.randn(B, T), torch.randn(B, T)
    spans = [(2, 7), (0, 12), (5, 6)]  # prompt offset / response end per row
    mask = torch.zeros(B, T)
    for i, (s, e) in enumerate(spans):
        mask[i, s:e] = 1
    adv = gae(rewards * mask, values, mask, gamma=0.99, lam=0.9)
    for i, (s, e) in enumerate(spans):
        ref = gae_reference(rewards[i, s:e].tolist(), values[i, s:e].tolist(), 0.99, 0.9)
        torch.testing.assert_close(adv[i, s:e], torch.tensor(ref))
        assert adv[i, :s].abs().sum() == 0 and adv[i, e:].abs().sum() == 0


def test_judge_order_swap_aggregation(tmp_path, monkeypatch):
    from llm import judge as J

    monkeypatch.setattr(J.anthropic, "Anthropic", lambda: None)
    j = J.Judge(tmp_path / "cache.jsonl")
    # Fake judge that always prefers whichever response contains "good", else says "A" (position bias).
    def fake(prompt, a, b, ref):
        return {"winner": "A" if "good" in a else ("B" if "good" in b else "A")}
    monkeypatch.setattr(j, "_pairwise_one", fake)
    assert j.pairwise("p", "bad", "good")["outcome"] == "B"
    assert j.pairwise("p", "good", "bad")["outcome"] == "A"
    assert j.pairwise("p", "meh", "bad")["outcome"] == "tie"  # pure position bias -> tie
    summary, _ = j.win_rate([dict(id=1, prompt="p", a="bad", b="good"), dict(id=2, prompt="p", a="x", b="y")])
    assert summary["b_wins"] == 1 and summary["ties"] == 1 and summary["win_rate_b"]["mean"] == 0.75
