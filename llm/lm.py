"""A common scoring interface over our models and Hugging Face reference models.

The eval harness only needs: encode(text), the token(s) that start a context, max context length,
and hidden(x) / lm_head(h) so log-probs are computed only where needed. transformers is used
here purely to *evaluate* external reference checkpoints (Pythia, OPT, ...), never for training.
"""

import torch
from tokenizers import Tokenizer

from llm.checkpoint import load_model


class OurLM:
    def __init__(self, ckpt: str, tokenizer: str, device):
        self.model, self.ckpt = load_model(ckpt, device)
        self.tok = Tokenizer.from_file(tokenizer)
        self.eot = self.tok.token_to_id("<|endoftext|>")
        self.prefix = [self.eot]  # our docs always follow <|endoftext|> in pretraining
        self.empty_ctx = [self.eot]
        self.max_len = self.model.cfg.max_seq_len
        self.device = device
        self.name = ckpt
        self.amp = False  # bf16 autocast; fp32 by default (precision-sensitive tasks)

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s).ids if s else []

    def hidden(self, x):
        return self.model.hidden(x)

    def lm_head(self, h):
        return self.model.lm_head(h)


class HFLM:
    """Mirrors lm-evaluation-harness HFLM defaults: no BOS added to contexts, BOS/EOS as the
    context of empty-context requests, fp32 weights (we autocast to bf16 inside the harness)."""

    def __init__(self, name: str, device, revision: str | None = None, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 2048)
        start = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.prefix, self.empty_ctx = [], [start]
        self.device = device
        self.name = name if revision is None else f"{name}@{revision}"
        self._head = self.model.get_output_embeddings()
        self.amp = False
        self._check_head()

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False) if s else []

    def hidden(self, x):
        return self.model.base_model(input_ids=x).last_hidden_state

    def lm_head(self, h):
        return self._head(h)

    @torch.no_grad()
    def _check_head(self):
        """hidden() + lm_head() must reproduce the model's own logits (catches e.g. logit soft-capping
        or an extra projection that our split would silently skip)."""
        x = torch.tensor([self.encode("The quick brown fox jumps over the lazy dog.")], device=self.device)
        ref = self.model(input_ids=x).logits.float()
        ours = self.lm_head(self.hidden(x)).float()
        err = (ref - ours).abs().max().item()
        assert err < 1e-3 * max(ref.abs().max().item(), 1.0), f"hidden/lm_head split mismatch for {self.name}: {err}"
