#!/usr/bin/env python3
"""Build the evaluation sets from the held-out sq-corpus validation split.

Outputs (JSONL, ml/data/out/eval/):
  diacritics.jsonl   T1  ë/ç stripped from each diacritic word with probability --p
  twins.jsonl        T2  only the ambiguous twins stripped (në→ne, të→te, së→se, më→me, ça→ca);
                         plain forms that were already correct (ne = "we") stay as negatives
  predict.jsonl      T5  {context, target}: predict/complete the next word after `context`

T1/T2 follow ml/data/schema.json. Built only from the validation split, which is never
trained on. Deterministic for a given --seed.

Usage (repo root):  services/ai/.venv/bin/python ml/data/build_eval_sets.py [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ml" / "data" / "out" / "sq-corpus" / "validation.parquet"
OUT = ROOT / "ml" / "data" / "out" / "eval"

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZËÇ])")
TWINS = {"në": "ne", "të": "te", "së": "se", "më": "me", "ça": "ca"}


def strip(w: str) -> str:
    return w.replace("ë", "e").replace("ç", "c").replace("Ë", "E").replace("Ç", "C")


def match_case(src: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def sentences(min_words: int = 6, max_words: int = 40) -> list[str]:
    out = []
    for para in pq.read_table(CORPUS, columns=["text"]).column(0).to_pylist():
        for s in SENT_END.split(para):
            n = len(WORD.findall(s))
            if min_words <= n <= max_words:
                out.append(s.strip())
    return out


def noisy(sent: str, rng: random.Random, p: float, twins_only: bool):
    errors = []

    def sub(m: re.Match) -> str:
        w = m.group(0)
        if twins_only:
            plain = TWINS.get(w.lower())
            if plain is None:
                return w
            new = match_case(w, plain)
        else:
            if strip(w) == w or rng.random() >= p:
                return w
            new = strip(w)
        errors.append({"type": "missing_diacritic", "original": new, "correction": w})
        return new

    return WORD.sub(sub, sent), errors


def record(sent: str, inp: str, errors: list) -> dict:
    return {
        "input": inp,
        "target": sent,
        "errors": errors,
        "dialect": "standard",
        "source": "synthetic:wikipedia-sq-20231101",
        "license": "CC-BY-SA-4.0",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="items per set")
    ap.add_argument("--p", type=float, default=0.6, help="T1: probability of stripping a word")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sents = sentences()
    rng.shuffle(sents)
    print(f"{len(sents)} candidate sentences from {CORPUS.name}")
    OUT.mkdir(parents=True, exist_ok=True)

    # Disjoint sentence pools per set, so no sentence is reused across tasks.
    pools = iter(sents)
    t1, t2, t5 = [], [], []
    for s in pools:
        if len(t1) < args.n:
            inp, errs = noisy(s, rng, args.p, twins_only=False)
            if errs:
                t1.append(record(s, inp, errs))
            continue
        if len(t2) < args.n:
            lows = [w.lower() for w in WORD.findall(s)]
            if any(w in TWINS for w in lows):
                inp, errs = noisy(s, rng, args.p, twins_only=True)
                t2.append(record(s, inp, errs))
            continue
        if len(t5) < args.n:
            # Position ≥ 3, on a word followed by a space in the original text.
            spans = list(WORD.finditer(s))
            ok = [m for m in spans[3:] if m.start() > 0 and s[m.start() - 1] == " "]
            if ok:
                m = rng.choice(ok)
                t5.append({"context": s[: m.start()], "target": m.group(0)})
            continue
        break

    for name, rows in (("diacritics", t1), ("twins", t2), ("predict", t5)):
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path.relative_to(ROOT)}: {len(rows)} items")
    if min(len(t1), len(t2), len(t5)) < args.n:
        print("warning: fewer items than requested (validation split exhausted)")


if __name__ == "__main__":
    main()
