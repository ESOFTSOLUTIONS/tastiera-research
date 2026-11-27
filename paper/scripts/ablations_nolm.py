#!/usr/bin/env python3
"""No-LM ablations of the Tastiera lexicon pipeline (paper, Results > Ablations).

Patches the pipeline in memory (the product code is not modified) and scores each
variant with the official scorer from ml/eval/eval_lm.py on T1 (diacritics) and T2
(twins), in auto (conf >= 0.95) and all modes.

Variants
  full           the lexicon system as evaluated in the paper
  no-proper      proper-noun heuristic off (capitalised mid-sentence words get all candidates)
  gen:fold       candidates = same letters ignoring ë/ç only (no edit distance, no Hunspell restore)
  gen:fold+ed1   + edit distance 1 (no Hunspell restore)
  no-table       safe autocorrection table off (step 1 skipped)

Usage:
  python paper/scripts/ablations_nolm.py \
      --repo . --limit 1000 --out paper/results/ablations_nolm.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
repo = args.repo.expanduser().resolve()

os.environ["TASTIERA_LM_DISABLED"] = "1"
os.environ.setdefault("TASTIERA_DICT_DIR", str(repo / "dictionary" / "data"))
sys.path.insert(0, str(repo / "services" / "ai"))

spec = importlib.util.spec_from_file_location("eval_lm", repo / "ml" / "eval" / "eval_lm.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

from tastiera_ai.pipeline import lexicon as lexmod  # noqa: E402
from tastiera_ai.pipeline import pipeline  # noqa: E402
from tastiera_ai.pipeline.lexicon import fold  # noqa: E402

lex = lexmod.get()
assert lex is not None, "no lexicon; run `make dict` in the product repo"
L = type(lex)
orig_candidates = L.candidates
orig_check = pipeline._check_word


def cand_fold(self, word, limit=5):
    w = word.lower()
    found = set(self.by_fold.get(fold(w), []))
    found.discard(w)
    return sorted(found, key=lambda c: -self.freq.get(c, 0))[:limit]


def cand_fold_ed1(self, word, limit=5):
    w = word.lower()
    fw = fold(w)
    found = set(self.by_fold.get(fw, []))
    for k in {fw} | {fw[:i] + fw[i + 1 :] for i in range(len(fw))}:
        found |= self.deletes.get(k, set())
    found.discard(w)
    return sorted(found, key=lambda c: (fold(c) != fw, -self.freq.get(c, 0)))[:limit]


def check_no_proper(text, m, use_lm):
    """_check_word with the proper-noun filter removed (step 3 only)."""
    word = m.group(0)
    low = word.lower()
    if low in lex.autocorrect or low in pipeline.AMBIGUOUS:
        return orig_check(text, m, use_lm)
    if lex.is_valid(word):
        return None
    cands = lex.candidates(word if word[:1].isupper() else low)
    if not cands:
        return None
    best, conf, reason = pipeline._match_case(word, cands[0]), 0.85, "drejtshkrim"
    if fold(cands[0]) == fold(low):
        conf, reason = 0.95, "mungon ë/ç"
    kind = "diacritics" if fold(best.lower()) == fold(low) else "spelling"
    return pipeline._make(text, m, best, kind, conf, reason)


def check_no_table(text, m, use_lm):
    saved = lex.autocorrect
    lex.autocorrect = {}
    try:
        return orig_check(text, m, use_lm)
    finally:
        lex.autocorrect = saved


VARIANTS = {
    "full": {},
    "no-proper": {"check": check_no_proper},
    "gen:fold": {"cand": cand_fold},
    "gen:fold+ed1": {"cand": cand_fold_ed1},
    "no-table": {"check": check_no_table},
}

sets = {s: ev.load_jsonl(s, args.limit) for s in ("diacritics", "twins")}
report = {"git": ev.git_commit(), "limit": args.limit, "results": {}}
for vname, v in VARIANTS.items():
    L.candidates = v.get("cand", orig_candidates)
    pipeline._check_word = v.get("check", orig_check)
    systems = ev.correction_systems(["lexicon"], lex, None)
    for sname, rows in sets.items():
        for mode in ("auto", "all"):
            r = ev.score_correction(rows, systems[f"lexicon:{mode}"])
            report["results"].setdefault(sname, {})[f"{vname}:{mode}"] = r
            print(
                f"{sname:10s} {vname:14s} {mode:4s}  P={r['precision']:.3f} R={r['recall']:.3f} "
                f"F0.5={r['f0.5']:.3f} FPR={r['false_positive_rate']*100:.2f}%"
            )
L.candidates, pipeline._check_word = orig_candidates, orig_check

if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
