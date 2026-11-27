#!/usr/bin/env python3
"""Per-token LM margins for the ambiguous twins, for the τ sweep and the error analysis.

The pipeline changes a plain twin (ne, te, se, me, ca) into its diacritic form (në, të,
së, më, ça) when Δ = log p(sentence with twin) − log p(sentence as typed) ≥ τ (LM_MARGIN,
2 nats), and applies it automatically only when its confidence 0.80 + Δ/40 ≥ 0.95, i.e.
Δ ≥ 6. This script records Δ for every twin token of the T1/T2 sets in three contexts, so
any τ / threshold can be evaluated offline from one model pass:

  two     the pipeline's context: pass-1 dictionary fixes applied, other twins as typed
  single  the raw input (no pass 1) — single- vs two-pass ablation
  oracle  every other word as in the reference — is the noisy context to blame?

Output: one JSON line per twin token {set, i, pos, word, twin, gold, d_two, d_single,
d_oracle}; gold = the reference has the diacritic form. Then a summary table.

  python ml/eval/twin_margins.py --model ml/runs/lm-v1/merged --out ml/eval/results/lm-v1-twins.jsonl
  python ml/eval/twin_margins.py --summarize ml/eval/results/lm-v1-twins.jsonl [...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "ml" / "data" / "out" / "eval"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
TWINS = {"ne": "në", "te": "të", "se": "së", "me": "më", "ca": "ça"}
CONTEXTS = ("two", "single", "oracle")
TAUS = [0, 0.5, 1, 2, 3, 4, 6, 8]
AUTO_DELTA = 6.0  # confidence 0.80 + Δ/40 reaches AUTO_APPLY_THRESHOLD 0.95 at Δ = 6


def match_case(src: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def splice(text: str, m: re.Match, word: str) -> str:
    return text[: m.start()] + word + text[m.end() :]


def pass1(text: str) -> str:
    """The pipeline's pass 1: confident dictionary fixes, twins left as typed."""
    from tastiera_ai.pipeline import pipeline as pl

    words = list(WORD.finditer(text))
    fixed = {}
    for i, m in enumerate(words):
        if m.group(0).lower() not in pl.AMBIGUOUS:
            s = pl._check_word(text, m, use_lm=False)
            if s is not None and s.confidence >= 0.95:
                fixed[i] = s
    return pl._apply(text, words, fixed)


def margins(rows: list[dict], set_name: str) -> list[dict]:
    from tastiera_ai.pipeline import lm

    out = []
    for i, r in enumerate(rows):
        src, tgt = r["input"], r["target"]
        ctx = {"single": src, "two": pass1(src), "oracle": tgt}
        ws = {k: list(WORD.finditer(v)) for k, v in ctx.items()}
        gold_words = WORD.findall(tgt)
        if not (len(ws["single"]) == len(ws["two"]) == len(gold_words)):
            continue
        items, texts = [], []
        for pos, m in enumerate(ws["single"]):
            word = m.group(0)
            if word.lower() not in TWINS:
                continue
            twin = match_case(word, TWINS[word.lower()])
            items.append((pos, word, twin))
            for k in CONTEXTS:
                mk = ws[k][pos]
                texts += [splice(ctx[k], mk, word), splice(ctx[k], mk, twin)]
        if not items:
            continue
        lp = lm.score(texts)
        for j, (pos, word, twin) in enumerate(items):
            rec = {
                "set": set_name,
                "i": i,
                "pos": pos,
                "word": word,
                "twin": twin,
                "gold": gold_words[pos] == twin,
            }
            for c, k in enumerate(CONTEXTS):
                a, b = (
                    lp[(j * len(CONTEXTS) + c) * 2],
                    lp[(j * len(CONTEXTS) + c) * 2 + 1],
                )
                rec[f"d_{k}"] = round(b - a, 4)
            out.append(rec)
        if (i + 1) % 100 == 0:
            print(f"[{set_name}] {i + 1}/{len(rows)}", flush=True)
    return out


def curve(recs: list[dict], ctx: str, tau: float) -> dict:
    """Twin-level counts when a twin is changed iff Δ ≥ tau."""
    tp = sum(1 for r in recs if r[f"d_{ctx}"] >= tau and r["gold"])
    fp = sum(1 for r in recs if r[f"d_{ctx}"] >= tau and not r["gold"])
    pos = sum(1 for r in recs if r["gold"])
    neg = len(recs) - pos
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "fn": pos - tp,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / pos, 4) if pos else None,
        "fpr": round(fp / neg, 4) if neg else None,
        "accuracy": round((tp + neg - fp) / len(recs), 4) if recs else None,
    }


def summarize(recs: list[dict]) -> dict:
    out = {}
    for s in sorted({r["set"] for r in recs}):
        rs = [r for r in recs if r["set"] == s]
        out[s] = {
            "tokens": len(rs),
            "gold_twin": sum(r["gold"] for r in rs),
            # argmax accuracy: pick the more probable spelling, no margin
            "argmax_accuracy": {k: curve(rs, k, 0)["accuracy"] for k in CONTEXTS},
            "sweep": {k: [curve(rs, k, t) for t in TAUS] for k in CONTEXTS},
            "auto": {k: curve(rs, k, AUTO_DELTA) for k in CONTEXTS},
        }
    return out


def print_summary(name: str, summ: dict) -> None:
    print(f"\n## {name}")
    for s, d in summ.items():
        if not isinstance(d, dict):  # model / revision
            continue
        print(
            f"\n{s}: {d['tokens']} twin tokens, {d['gold_twin']} need the diacritic; "
            f"argmax accuracy "
            + ", ".join(f"{k} {v}" for k, v in d["argmax_accuracy"].items())
        )
        print("  ctx     tau   P      R      FPR    acc")
        for k in CONTEXTS:
            for c in d["sweep"][k]:
                print(
                    f"  {k:<7} {c['tau']:<4}  {c['precision']}  {c['recall']}  {c['fpr']}  {c['accuracy']}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--model", help="Hub id or local folder")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--sets", default="diacritics,twins")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument(
        "--out", type=Path, help="per-token JSONL (summary goes to <out>.summary.json)"
    )
    ap.add_argument(
        "--summarize", nargs="+", type=Path, help="only summarize existing JSONL files"
    )
    args = ap.parse_args()

    if args.summarize:
        for p in args.summarize:
            recs = [json.loads(line) for line in p.open()]
            print_summary(p.stem, summarize(recs))
        return

    if not (args.model and args.out):
        ap.error("--model and --out are required (or use --summarize)")
    p = Path(args.model)
    os.environ["TASTIERA_LM_MODEL"] = str(p.resolve()) if p.exists() else args.model
    os.environ["TASTIERA_LM_REVISION"] = args.revision or ""
    os.environ.setdefault("TASTIERA_DICT_DIR", str(ROOT / "dictionary" / "data"))
    sys.path.insert(0, str(ROOT / "services" / "ai"))

    recs = []
    for s in args.sets.split(","):
        rows = [json.loads(line) for line in (args.eval_dir / f"{s}.jsonl").open()][
            : args.limit
        ]
        recs += margins(rows, s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    summ = summarize(recs)
    summ["model"], summ["revision"] = args.model, args.revision
    args.out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print_summary(args.out.stem, summ)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
