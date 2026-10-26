"""Held-out loss and bits-per-byte on a fixed prefix of a token file."""

import math

import torch
import torch.nn.functional as F

from llm.data import TokenFile
from llm.dist import all_reduce_sum


@torch.no_grad()
def eval_loss(model, data: TokenFile, token_bytes: torch.Tensor, n_tokens: int, micro_batch: int,
              rank: int, world: int, device: torch.device) -> dict:
    """Always the same first n_tokens of the file, so every eval is on identical data.

    loss: mean next-token cross-entropy (nats) over all targets, including <|endoftext|>.
    bpb:  total nats / ln2 / total UTF-8 bytes, over targets that are real text (special tokens
          have 0 bytes and are excluded from both sums). Tokenizer-independent.
    """
    was_training = model.training
    model.eval()
    # Equal, full batches on every rank (avoids recompiles on a ragged last batch).
    n_chunks = min(n_tokens // data.seq_len, data.n_chunks) // (world * micro_batch) * (world * micro_batch)
    ids = torch.arange(rank, n_chunks, world)
    sums = torch.zeros(4, dtype=torch.float64, device=device)  # loss_all, n_all, loss_text, bytes_text
    for i in range(0, len(ids), micro_batch):
        x = torch.from_numpy(data.chunks(ids[i : i + micro_batch].numpy())).to(device, non_blocking=True)
        inp, tgt = x[:, :-1], x[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inp)
        nll = F.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(), reduction="none")
        nbytes = token_bytes[tgt.flatten()]
        text = nbytes > 0
        sums += torch.stack([nll.sum(), torch.tensor(nll.numel(), device=device),
                             nll[text].sum(), nbytes.sum()]).double()
    all_reduce_sum(sums)
    model.train(was_training)
    loss_all, n_all, loss_text, n_bytes = sums.tolist()
    return dict(loss=loss_all / n_all, bpb=loss_text / math.log(2) / n_bytes, tokens=int(n_all))
