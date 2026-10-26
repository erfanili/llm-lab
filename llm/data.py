"""Token data loading.

A token file is a flat uint16 array (docs separated by <|endoftext|>). It is cut into
non-overlapping chunks of seq_len + 1 tokens (inputs + shifted targets). Training order
is a seeded permutation of chunks, and the rows for (step, micro_step, rank) are a pure
function of those indices, so resuming at step s reproduces the exact same data stream.
"""

import json

import numpy as np
import torch
from tokenizers import Tokenizer


class TokenFile:
    def __init__(self, path: str, seq_len: int):
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        # Chunk i covers tokens[i*T : i*T + T + 1]; neighbours share one token (target of the last position).
        self.n_chunks = (len(self.tokens) - 1) // seq_len

    def chunks(self, idx: np.ndarray) -> np.ndarray:
        T = self.seq_len
        return np.stack([self.tokens[i * T : i * T + T + 1] for i in idx]).astype(np.int64)


class TrainLoader:
    def __init__(self, path: str, seq_len: int, micro_batch: int, grad_accum: int,
                 rank: int, world_size: int, seed: int, device: torch.device):
        self.data = TokenFile(path, seq_len)
        self.B, self.accum, self.rank, self.world = micro_batch, grad_accum, rank, world_size
        self.seed, self.device = seed, device
        self._perm_epoch, self._perm = -1, None

    @property
    def tokens_per_step(self) -> int:
        return self.B * self.data.seq_len * self.accum * self.world

    def _chunk_ids(self, global_ids: np.ndarray) -> np.ndarray:
        # Each epoch is a fresh seeded permutation; a batch may straddle an epoch boundary.
        n = self.data.n_chunks
        out = np.empty_like(global_ids)
        for e in np.unique(global_ids // n):
            if e != self._perm_epoch:
                self._perm = np.random.default_rng([self.seed, int(e)]).permutation(n)
                self._perm_epoch = int(e)
            m = global_ids // n == e
            out[m] = self._perm[global_ids[m] % n]
        return out

    def get(self, step: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Global sample index layout: step -> micro_step -> rank -> row.
        first = ((step * self.accum + micro) * self.world + self.rank) * self.B
        ids = self._chunk_ids(np.arange(first, first + self.B))
        x = torch.from_numpy(self.data.chunks(ids)).pin_memory().to(self.device, non_blocking=True)
        return x[:, :-1], x[:, 1:]


def token_byte_lengths(tokenizer_path: str) -> np.ndarray:
    """UTF-8 byte length of every token id (0 for special tokens).

    With byte-level BPE every token decodes to a fixed byte string, so the number of
    bytes covered by any token span is exact. This makes bits-per-byte exact on any
    eval subset, and comparable across tokenizers.
    """
    tok = Tokenizer.from_file(tokenizer_path)
    # GPT-2 byte <-> printable unicode char mapping used by the ByteLevel pre-tokenizer.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    lens = np.zeros(tok.get_vocab_size(), dtype=np.int64)
    for s, i in tok.get_vocab().items():
        lens[i] = 0 if s in special else len(bytes(char_to_byte[ch] for ch in s))
    return lens


def load_meta(bin_path: str) -> dict:
    with open(bin_path.removesuffix(".bin") + ".json") as f:
        return json.load(f)
