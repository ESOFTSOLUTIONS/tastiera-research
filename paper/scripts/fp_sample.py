#!/usr/bin/env python3
"""Dump false positives of the lexicon system (no LM) on T1 for manual inspection.

A false positive here is a word the user typed correctly (input == reference) that the
system changed. Writes a seeded random sample of at most --n of them (on T1 auto there
are only 99, so all of them) as JSONL with the sentence context, so that each can be
labelled by hand. The labels in results/fp_t1_lexicon_auto.jsonl were added manually.

Usage:
  python paper/scripts/fp_sample.py \
      --repo . --mode auto --out paper/results/fp_t1_lexicon_auto.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--set", default="diacritics")
ap.add_argument("--mode", default="auto", choices=("auto", "all"))
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
repo = args.repo.expanduser().resolve()

os.environ["TASTIERA_LM_DISABLED"] = "1"
os.environ.setdefault("TASTIERA_DICT_DIR", str(repo / "dictionary" / "data"))
sys.path.insert(0, str(repo / "services" / "ai"))

spec = importlib.util.spec_from_file_location("eval_lm", repo / "ml" / "eval" / "eval_lm.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

from tastiera_ai.pipeline import lexicon as lexmod  # noqa: E402

lex = lexmod.get()
fn = ev.correction_systems(["lexicon"], lex, None)[f"lexicon:{args.mode}"]
rows = ev.load_jsonl(args.set, args.limit)

fps = []
for i, r in enumerate(rows):
    out = fn(r["input"])
    if isinstance(out, tuple):
        out = out[0]
    a, g, o = (ev.WORD.findall(x) for x in (r["input"], r["target"], out))
    if not (len(a) == len(g) == len(o)):
        continue
    for k, (x, y, z) in enumerate(zip(a, g, o)):
        if x == y and z != x:
            fps.append({"i": i, "pos": k, "typed": x, "changed_to": z,
                        "context": " ".join(g[max(0, k - 8):k + 9])})

random.Random(args.seed).shuffle(sample := list(fps))
sample = sorted(sample[: args.n], key=lambda d: (d["i"], d["pos"]))
args.out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in sample), encoding="utf-8")
print(f"{len(fps)} false positives on correct words; wrote {len(sample)} to {args.out} "
      f"(git {ev.git_commit()}, seed {args.seed})")
