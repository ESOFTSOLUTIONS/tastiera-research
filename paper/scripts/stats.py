#!/usr/bin/env python3
"""Recompute the resource / corpus numbers quoted in the paper.

Usage:  python paper/scripts/stats.py [path/to/this/repo]
Writes a Markdown table to stdout.
"""

import gzip
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
data = repo / "dictionary" / "data"


def rows(p: Path) -> int:
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))


commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = [("Repository commit", commit)]
out += [
    ("Lexicon words (`lexicon.tsv`)", rows(data / "lexicon.tsv")),
    ("Autocorrect rules (`autocorrect.tsv`)", rows(data / "autocorrect.tsv")),
    ("n-gram context lines (`ngrams.tsv`)", rows(data / "ngrams.tsv")),
    ("n-gram file size (MB)", round((data / "ngrams.tsv").stat().st_size / 1e6, 2)),
]
with gzip.open(data / "fjalor.jsonl.gz", "rt", encoding="utf-8") as fh:
    out.append(("Fjalori entries", sum(1 for _ in fh)))

try:
    import pyarrow.parquet as pq

    for split in ("train", "validation"):
        texts = pq.read_table(repo / "ml/data/out/sq-corpus" / f"{split}.parquet", columns=["text"]).column(0).to_pylist()
        out.append((f"sq-corpus {split} paragraphs", len(texts)))
        out.append((f"sq-corpus {split} words", sum(len(t.split()) for t in texts)))
except Exception as e:  # pyarrow missing or corpus not built
    out.append(("sq-corpus", f"unavailable: {e}"))

print("| Quantity | Value |\n|---|---|")
for k, v in out:
    print(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
