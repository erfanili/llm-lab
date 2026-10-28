#!/usr/bin/env bash
# Build tokens/train10b.bin: the existing 1B-stage train.bin (shards 000-002) + shards 003-012 of
# FineWeb-Edu sample/10BT. Shard 013 stays held out (val/test), so the val set is unchanged.
# /data is nearly full, so each raw shard is downloaded, tokenized, appended, then deleted.
set -euo pipefail
cd "$(dirname "$0")/.."

D=/data/erfan/llm-lab
OUT=$D/tokens/train10b.bin
TMP=$D/tokens/_shard
PY="uv run --no-sync python"

[ -f "$OUT" ] || cp "$D/tokens/train.bin" "$OUT"
for i in $(seq -w 3 12); do
    f=sample/10BT/0${i}_00000.parquet
    [ -f "$D/tokens/.done_0$i" ] && continue
    $PY -c "from huggingface_hub import hf_hub_download as h; print(h('HuggingFaceFW/fineweb-edu', '$f', repo_type='dataset', local_dir='$D/raw/fineweb-edu'))"
    $PY scripts/tokenize_data.py --tokenizer "$D/tokenizer/bpe32k.json" --files "$D/raw/fineweb-edu/$f" --out "$TMP"
    cat "$TMP.bin" >> "$OUT"
    mv "$TMP.json" "$D/tokens/.done_0$i"   # per-shard meta doubles as the resume marker
    rm "$TMP.bin" "$D/raw/fineweb-edu/$f"
done

# Combined meta: sum of the 1B-stage meta and the per-shard metas.
$PY - <<EOF
import json, glob
metas = [json.load(open("$D/tokens/train.json"))] + [json.load(open(p)) for p in sorted(glob.glob("$D/tokens/.done_0*"))]
out = dict(n_tokens=sum(m["n_tokens"] for m in metas), n_bytes=sum(m["n_bytes"] for m in metas),
           n_docs=sum(m["n_docs"] for m in metas), files=[f for m in metas for f in m["files"]],
           tokenizer=metas[0]["tokenizer"], dtype="uint16")
import os; assert os.path.getsize("$OUT") == 2 * out["n_tokens"], "bin size != meta token count"
json.dump(out, open("$D/tokens/train10b.json", "w"), indent=2)
print(out["n_tokens"], "tokens")
EOF
