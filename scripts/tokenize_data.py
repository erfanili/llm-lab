"""Tokenize parquet text into a flat uint16 token file: doc0 <eot> doc1 <eot> ...

Writes <out>.bin plus <out>.json with token/byte/doc counts (bytes are needed for bits-per-byte).
Use --doc-range to take a contiguous slice of documents (e.g. split a held-out file into val/test).
"""

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--doc-range", type=float, nargs=2, default=(0.0, 1.0),
                    help="fraction [start, end) of each file's documents to keep")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    assert tok.get_vocab_size() <= 2**16
    n_tokens = n_bytes = n_docs = 0
    with open(args.out + ".bin", "wb") as fout:
        for path in args.files:
            pf = pq.ParquetFile(path)
            total = pf.metadata.num_rows
            lo, hi = int(args.doc_range[0] * total), int(args.doc_range[1] * total)
            row = 0
            for batch in tqdm(pf.iter_batches(batch_size=8192, columns=["text"]), total=-(-total // 8192), desc=path):
                texts = batch.column("text").to_pylist()
                start, end = max(lo - row, 0), min(hi - row, len(texts))
                row += len(texts)
                texts = texts[start:end]
                if not texts:
                    continue
                encs = tok.encode_batch_fast(texts)
                ids = np.concatenate([np.append(np.asarray(e.ids, dtype=np.uint16), np.uint16(eot)) for e in encs])
                fout.write(ids.tobytes())
                n_tokens += len(ids)
                n_bytes += sum(len(t.encode("utf-8")) for t in texts)
                n_docs += len(texts)
    meta = dict(n_tokens=n_tokens, n_bytes=n_bytes, n_docs=n_docs, files=args.files,
                doc_range=list(args.doc_range), tokenizer=args.tokenizer, dtype="uint16")
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), f"\nbytes/token = {n_bytes / n_tokens:.3f}")


if __name__ == "__main__":
    main()
