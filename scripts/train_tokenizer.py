"""Train a byte-level BPE tokenizer on a sample of the training parquet files.

The same tokenizer is used for every model in the scaling ladder, so losses are comparable.
"""

import argparse
import itertools
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# GPT-4 (cl100k) pre-tokenization regex: splits contractions, words, 1-3 digit groups, punctuation, whitespace.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
EOT = "<|endoftext|>"


def iter_texts(files, max_docs):
    def gen():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=["text"]):
                yield from batch.column("text").to_pylist()
    return itertools.islice(gen(), max_docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--max-docs", type=int, default=500_000)
    args = ap.parse_args()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOT],  # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes: no unknown tokens ever
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(args.files, args.max_docs), trainer=trainer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    assert tok.token_to_id(EOT) == 0 and tok.get_vocab_size() == args.vocab_size
    sample = "Hello world! The year 2024 had 365 days.\n\tdef f(x): return x**2  # ünïcödé 日本語"
    assert tok.decode(tok.encode(sample).ids) == sample
    print(f"saved {args.out}, vocab={tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
