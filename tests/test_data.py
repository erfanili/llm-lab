from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from llm.data import TrainLoader, token_byte_lengths

TOKENIZER = "/data/erfan/llm-lab/tokenizer/bpe32k.json"


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason="tokenizer not trained")
def test_token_byte_lengths_exact():
    tok = Tokenizer.from_file(TOKENIZER)
    lens = token_byte_lengths(TOKENIZER)
    for text in ["Hello world!", "ünïcödé 日本語 🙂", "  \n\tcode(x) {return 2**10;}", "1234567890"]:
        assert lens[tok.encode(text).ids].sum() == len(text.encode("utf-8"))
    assert lens[tok.token_to_id("<|endoftext|>")] == 0


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "toy.bin"
    np.arange(10_000, dtype=np.uint16).tofile(path)
    return str(path)


def make(path, rank, world=2, seed=0):
    return TrainLoader(path, seq_len=16, micro_batch=4, grad_accum=2, rank=rank, world_size=world,
                       seed=seed, device=torch.device("cpu"))


def test_targets_are_shifted_inputs(token_file):
    x, y = make(token_file, 0).get(0, 0)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y - x, torch.ones_like(x))  # toy data is arange


def test_deterministic_and_resumable(token_file):
    a, b = make(token_file, 0), make(token_file, 0)
    b.get(7, 1)  # touch other state first
    for step in (3, 0, 5):
        assert torch.equal(a.get(step, 1)[0], b.get(step, 1)[0])


def test_ranks_and_steps_disjoint_within_epoch(token_file):
    loaders = [make(token_file, r) for r in range(2)]
    n_chunks = loaders[0].data.n_chunks
    seen = []
    steps = n_chunks // (4 * 2 * 2)
    for step in range(steps):
        for micro in range(2):
            for ld in loaders:
                seen += (ld.get(step, micro)[0][:, 0] // 16).tolist()
    assert len(seen) == len(set(seen))  # no chunk repeated within an epoch


def test_seed_changes_order(token_file):
    assert not torch.equal(make(token_file, 0, seed=0).get(0, 0)[0], make(token_file, 0, seed=1).get(0, 0)[0])
