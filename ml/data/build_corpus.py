#!/usr/bin/env python3
"""Clean Albanian corpus for LM training → ml/data/out/sq-corpus/{train,validation}.parquet.

Source: Albanian Wikipedia (wikimedia/wikipedia 20231101.sq, CC BY-SA 4.0), the same file
dictionary/build_ngrams.py downloads. Keeps prose paragraphs that contain ë/ç (correct
spelling), removes exact duplicates and holds out 1% for validation.

Usage (from the repo root):  services/ai/.venv/bin/python ml/data/build_corpus.py
Publish with:                services/ai/.venv/bin/python ml/hf/push_datasets.py corpus
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "out" / "sq-corpus"
SOURCE = "wikipedia-sq-20231101"
LICENSE = "CC-BY-SA-4.0"
ALB = re.compile(r"[ëçËÇ]")


def wikipedia() -> Path:
    import sys

    sys.path.insert(0, str(ROOT / "dictionary"))
    from build_ngrams import (
        fetch_wikipedia,  # same download, cached in dictionary/sources
    )

    return fetch_wikipedia()


def paragraphs(path: Path):
    table = pq.read_table(path, columns=["text"])
    for batch in table.to_batches(max_chunksize=2000):
        for article in batch.column(0).to_pylist():
            for p in article.split("\n"):
                p = " ".join(p.split())
                if (
                    len(p) < 120
                    or not ALB.search(p)
                    or p.count("|") > 2
                    or p.startswith(("*", "{", "<"))
                ):
                    continue
                yield p


def main() -> None:
    seen: set[bytes] = set()
    rows = []
    for p in paragraphs(wikipedia()):
        h = hashlib.md5(p.encode()).digest()
        if h not in seen:
            seen.add(h)
            rows.append(p)
    random.Random(42).shuffle(rows)
    n_val = max(1, len(rows) // 100)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, texts in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        table = pa.table(
            {
                "text": texts,
                "source": [SOURCE] * len(texts),
                "license": [LICENSE] * len(texts),
            }
        )
        pq.write_table(table, OUT / f"{split}.parquet", compression="zstd")
        words = sum(len(t.split()) for t in texts)
        print(f"{split}: {len(texts)} paragraphs, {words / 1e6:.1f}M words")


if __name__ == "__main__":
    main()
