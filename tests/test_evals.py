import torch

from llm.evals import doc_bpb, encode_pair
from llm.model import ModelConfig, Transformer


class CharLM:
    """Character-level LM wrapper around a tiny random Transformer: 1 token == 1 byte (ASCII)."""

    def __init__(self, max_len=16):
        self.model = Transformer(ModelConfig(vocab_size=128, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                                             d_ff=64, max_seq_len=max_len)).eval()
        self.prefix, self.empty_ctx, self.max_len, self.device = [0], [0], max_len, torch.device("cpu")
        self.amp = False

    def encode(self, s):
        return [ord(c) for c in s]

    def hidden(self, x):
        return self.model.hidden(x)

    def lm_head(self, h):
        return self.model.lm_head(h)


def test_encode_pair_moves_trailing_space():
    lm = CharLM()
    ctx, cont = encode_pair(lm, "Answer: ", "yes")
    assert ctx == [0] + lm.encode("Answer:") and cont == lm.encode(" yes")
    assert encode_pair(lm, "", "hi") == ([0], lm.encode("hi"))


def test_doc_bpb_scores_every_token_once_with_full_context_when_short():
    lm = CharLM(max_len=16)
    docs = ["short", "a much longer document that needs several rolling windows to cover"]
    r = doc_bpb(lm, docs)
    assert r["tokens"] == sum(len(d) for d in docs) and r["bytes"] == r["tokens"]
    # short doc fits in one window: must equal a plain full forward pass
    x = torch.tensor([[0] + lm.encode(docs[0])])
    with torch.no_grad():
        lp = lm.model(x[:, :-1]).log_softmax(-1)[0].gather(1, x[0, 1:, None]).sum().item()
    short = doc_bpb(lm, docs[:1])
    assert abs(short["bpb"] - (-lp / torch.log(torch.tensor(2.0)).item() / len(docs[0]))) < 1e-4
