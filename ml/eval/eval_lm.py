#!/usr/bin/env python3
"""Evaluate a language model inside the real Tastiera pipeline, against baselines.

Runs the systems below on the sets from ml/data/build_eval_sets.py and writes one JSON
file per run to ml/eval/results/<name>.json (plus a summary table on stdout).

Correction systems (on diacritics.jsonl = T1, twins.jsonl = T2):
  identity       no change (floor)
  autocorrect    context-free table only (dictionary/data/autocorrect.tsv)
  device         autocorrect + n-gram twin fix — what the offline keyboard does
  lexicon        server pipeline without the LM (frequency ranking)
  lm             server pipeline with the LM under test
  Pipeline systems are reported in two modes: `:auto` applies only suggestions with
  confidence ≥ 0.95 (keyboard autocorrect), `:all` applies every suggestion (user accepts all).

Prediction systems (on predict.jsonl = T5):
  ngram          on-device bigram/trigram model + frequency completion
  lm             the service's suggest() with the LM under test

Metrics: word-level precision / recall / F0.5, false-positive rate on words that were
already correct, sentence accuracy, accuracy on ambiguous twins; next-word top-1/top-3,
completion top-3 after 2 letters, keystroke savings rate (KSR, top-3 bar); LM bits per
character on held-out text; latency p50/p95.

Usage (repo root):
  services/ai/.venv/bin/python ml/eval/eval_lm.py --model Qwen/Qwen3-1.7B-Base --name base
  services/ai/.venv/bin/python ml/eval/eval_lm.py --model ml/runs/lm-v1/merged --name lm-v1
  services/ai/.venv/bin/python ml/eval/eval_lm.py --systems identity,autocorrect,device,lexicon,ngram --name no-lm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "ml" / "data" / "out" / "eval"
CORPUS_VAL = ROOT / "ml" / "data" / "out" / "sq-corpus" / "validation.parquet"
DICT = ROOT / "dictionary" / "data"
RESULTS = ROOT / "ml" / "eval" / "results"

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
TWINS = {"ne": "në", "te": "të", "se": "së", "me": "më", "ca": "ça"}
CORR_SYSTEMS = ["identity", "autocorrect", "device", "lexicon", "lm"]
PRED_SYSTEMS = ["ngram", "lm"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--model", default=None, help="Hub id or local folder (default: service default)"
    )
    ap.add_argument("--revision", default=None, help="Hub tag/branch/commit")
    ap.add_argument("--name", default=None, help="results file name (default: from --model)")
    ap.add_argument("--systems", default=",".join(dict.fromkeys(CORR_SYSTEMS + PRED_SYSTEMS)))
    ap.add_argument("--sets", default="diacritics,twins,predict")
    ap.add_argument("--limit", type=int, default=300, help="items per set")
    ap.add_argument(
        "--bpc-rows", type=int, default=300, help="validation paragraphs for BPC (0 = skip)"
    )
    ap.add_argument(
        "--twins",
        choices=["joint", "independent"],
        default=None,
        help="how the pipeline decides ne/në-type twins (default: TASTIERA_TWINS or joint)",
    )
    ap.add_argument("--no-ksr", action="store_true", help="skip keystroke savings (slowest metric)")
    ap.add_argument(
        "--eval-dir", type=Path, default=None, help=f"folder with the sets (default: {EVAL_DIR})"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="results file (default: ml/eval/results/<name>.json)"
    )
    return ap.parse_args()


# ---------------------------------------------------------------- helpers


def load_jsonl(name: str, limit: int) -> list[dict]:
    path = EVAL_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run ml/data/build_eval_sets.py first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def latency(ms: list[float]) -> dict:
    return {"p50_ms": round(pct(ms, 0.5), 1), "p95_ms": round(pct(ms, 0.95), 1), "n": len(ms)}


def git_commit() -> str:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip()


def match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def apply_suggestions(text: str, suggestions, min_conf: float) -> str:
    """Apply suggestions (UTF-8 byte offsets) with confidence ≥ min_conf."""
    b = text.encode()
    out, last = [], 0
    for s in sorted(suggestions, key=lambda s: s.start):
        if s.confidence < min_conf:
            continue
        out += [b[last : s.start], s.replacement.encode()]
        last = s.end
    out.append(b[last:])
    return b"".join(out).decode()


# ---------------------------------------------------------------- on-device n-gram model


class Ngrams:
    """Python port of apps/mobile/ios/TastieraKeyboard/NgramModel.swift (same file, same rules)."""

    START = "<s>"

    def __init__(self, path: Path):
        self.table: dict[str, list[tuple[str, int]]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            ctx, rest = line.split("\t", 1)
            parts = rest.split(" ")
            self.table[ctx] = [(parts[i], int(parts[i + 1])) for i in range(0, len(parts) - 1, 2)]

    def next(self, prev2: str | None, prev1: str | None, k: int = 3) -> list[str]:
        if prev1 is None:
            return [w for w, _ in self.table.get(self.START, [])][:k]
        out = [w for w, _ in self.table.get(f"{prev2} {prev1}", [])] if prev2 else []
        out += [w for w, _ in self.table.get(prev1, []) if w not in out]
        words = set(out)
        out = [w for w in out if TWINS.get(w) not in words]  # "te" next to "të": drop plain twin
        return out[:k]

    def count(self, word: str, prev: str | None) -> int:
        return dict(self.table.get(prev or self.START, [])).get(word, 0)

    def contextual_fix(self, word: str, prev: str | None) -> str | None:
        twin = TWINS.get(word.lower())
        if prev is None or twin is None:
            return None
        a, b = self.count(word.lower(), prev), self.count(twin, prev)
        return twin if b >= 20 and b >= 5 * max(a, 1) else None


def prev_words(text: str, end: int) -> tuple[str | None, str | None]:
    """Last two words before `end` within the current sentence, lower-cased."""
    before = re.split(r"[.!?]\s", text[:end])[-1]
    ws = [w.lower() for w in WORD.findall(before)]
    return (ws[-2] if len(ws) > 1 else None, ws[-1] if ws else None)


# ---------------------------------------------------------------- correction


def score_correction(rows: list[dict], fn) -> dict:
    tp = fp = fn_ = neg = neg_changed = sent_ok = misaligned = 0
    twin_n = twin_ok = 0
    ms: list[float] = []
    for r in rows:
        t0 = time.perf_counter()
        out = fn(r["input"])
        if isinstance(out, tuple):  # (text, pipeline ms) from a cached read-out
            out, took = out
        else:
            took = (time.perf_counter() - t0) * 1000
        ms.append(took)
        sent_ok += out == r["target"]
        a, g, o = (WORD.findall(x) for x in (r["input"], r["target"], out))
        if not (len(a) == len(g) == len(o)):
            misaligned += 1
            continue
        for x, y, z in zip(a, g, o, strict=True):
            need, changed = x != y, z != x
            if changed and z == y:
                tp += 1
            else:
                fp += changed
                fn_ += need
            if not need:
                neg += 1
                neg_changed += changed
            if x.lower() in TWINS:
                twin_n += 1
                twin_ok += z == y
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn_) if tp + fn_ else 0.0
    f05 = 1.25 * p * rc / (0.25 * p + rc) if p + rc else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(rc, 4),
        "f0.5": round(f05, 4),
        "false_positive_rate": round(neg_changed / neg, 5) if neg else None,
        "sentence_accuracy": round(sent_ok / len(rows), 4),
        "twin_accuracy": round(twin_ok / twin_n, 4) if twin_n else None,
        "counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn_,
            "correct_words": neg,
            "twins": twin_n,
            "misaligned": misaligned,
        },
        "latency": latency(ms),
        "n": len(rows),
    }


def correction_systems(names: list[str], lex, ngrams):
    from tastiera_ai.pipeline import pipeline

    def autocorrect(text: str) -> str:
        return WORD.sub(
            lambda m: match_case(m.group(0), lex.autocorrect.get(m.group(0).lower(), m.group(0))),
            text,
        )

    def device(text: str) -> str:
        def sub(m: re.Match) -> str:
            w = m.group(0)
            if w.lower() in lex.autocorrect:
                return match_case(w, lex.autocorrect[w.lower()])
            fix = ngrams.contextual_fix(w, prev_words(text, m.start())[1])
            return match_case(w, fix) if fix else w

        return WORD.sub(sub, text)

    systems = {}
    for name in names:
        if name == "identity":
            systems[name] = lambda t: t
        elif name == "autocorrect":
            systems[name] = autocorrect
        elif name == "device":
            systems[name] = device
        elif name in ("lexicon", "lm"):
            use_lm = name == "lm"
            # One pipeline call per sentence, two read-outs: cache (result, ms) by input.
            cache: dict[str, tuple] = {}

            def run(t, use_lm=use_lm, cache=cache):
                if t not in cache:
                    t0 = time.perf_counter()
                    res = pipeline.correct(t, mode="suggest", use_lm=use_lm)
                    cache[t] = (res, (time.perf_counter() - t0) * 1000)
                return cache[t]

            def readout(t, run=run, min_conf=0.0):
                res, ms = run(t)
                return apply_suggestions(t, res.suggestions, min_conf), ms

            systems[f"{name}:auto"] = lambda t, f=readout: f(t, min_conf=0.95)
            systems[f"{name}:all"] = lambda t, f=readout: f(t, min_conf=0.0)
    return systems


# ---------------------------------------------------------------- prediction


def prediction_system(name: str, lex, ngrams):
    """Returns f(context) -> top-3 words (context ends with a space or a partial word)."""
    if name == "lm":
        from tastiera_ai.pipeline import suggest

        return lambda ctx: [p.text for p in suggest.suggest(ctx, 3)]

    def ngram(ctx: str) -> list[str]:
        partial = "" if not ctx or ctx[-1].isspace() else WORD.findall(ctx)[-1]
        p2, p1 = prev_words(ctx, len(ctx) - len(partial))
        if not partial:
            return ngrams.next(p2, p1, 3)
        from tastiera_ai.pipeline.lexicon import fold

        cands = [w for w in ngrams.next(p2, p1, 50) if fold(w).startswith(fold(partial.lower()))]
        cands += [w for w in lex.complete(partial, 6) if w not in cands]
        return cands[:3]

    return ngram


def score_prediction(rows: list[dict], fn, ksr: bool, lex) -> dict:
    """Accuracy over all targets and over targets in the lexicon (names and rare words are
    unreachable for every system, since suggestions are restricted to valid words)."""
    top1 = top3 = comp3 = comp_n = 0
    iv_n = iv_top3 = iv_comp3 = iv_comp_n = 0
    typed_with = typed_without = 0
    ms: list[float] = []

    def hit(ctx: str, target: str) -> list[str]:
        t0 = time.perf_counter()
        preds = [p.lower() for p in fn(ctx)]
        ms.append((time.perf_counter() - t0) * 1000)
        return preds

    for r in rows:
        ctx, target = r["context"], r["target"]
        tl = target.lower()
        preds = hit(ctx, target)
        top1 += bool(preds) and preds[0] == tl
        top3 += tl in preds
        in_vocab = tl in lex.freq
        iv_n += in_vocab
        iv_top3 += in_vocab and tl in preds
        if len(target) > 3:
            c = tl in hit(ctx + target[:2], target)
            comp_n += 1
            comp3 += c
            iv_comp_n += in_vocab
            iv_comp3 += in_vocab and c
        if ksr:
            # Keystrokes to enter "target ": type letters until it shows in the top-3 bar,
            # then one tap (which also inserts the space). Without the bar: len + 1.
            cost = len(target) + 1
            for k in range(len(target)):
                if tl in (preds if k == 0 else hit(ctx + target[:k], target)):
                    cost = k + 1
                    break
            typed_with += cost
            typed_without += len(target) + 1
    n = len(rows)
    res = {
        "next_word_top1": round(top1 / n, 4),
        "next_word_top3": round(top3 / n, 4),
        "completion_top3_after_2": round(comp3 / comp_n, 4) if comp_n else None,
        "in_lexicon_rate": round(iv_n / n, 4),
        "in_lexicon_next_word_top3": round(iv_top3 / iv_n, 4) if iv_n else None,
        "in_lexicon_completion_top3_after_2": round(iv_comp3 / iv_comp_n, 4) if iv_comp_n else None,
        "latency": latency(ms),
        "n": n,
    }
    if ksr:
        res["ksr_top3"] = round(1 - typed_with / typed_without, 4)
    return res


# ---------------------------------------------------------------- bits per character


def bits_per_char(rows: int) -> dict:
    import pyarrow.parquet as pq
    import torch
    from tastiera_ai.pipeline import lm

    tok, model, device = lm._load()
    texts = pq.read_table(CORPUS_VAL, columns=["text"]).column(0).to_pylist()[:rows]
    nll = chars = ntok = words = 0
    for t in texts:
        # One start token whatever the tokenizer adds itself (Llama/Gemma/Mistral add BOS,
        # Qwen adds nothing), so bpc is comparable across base models.
        ids = lm.encode(tok, t)[1:1024]
        covered = tok.decode(ids)
        x = torch.tensor([[lm.start_id(tok)] + ids], device=device)
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, x[0, 1:].unsqueeze(-1))
        nll -= lp.sum().item()
        chars += len(covered)
        ntok += len(ids)
        words += len(covered.split())
    return {
        "bpc": round(nll / chars / math.log(2), 4),
        "ppl_token": round(math.exp(nll / ntok), 3),
        "tokens_per_word": round(ntok / words, 3),
        "paragraphs": len(texts),
        "chars": chars,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    global EVAL_DIR
    args = parse_args()
    if args.eval_dir:
        EVAL_DIR = args.eval_dir.resolve()
    if args.model:
        p = Path(args.model)
        os.environ["TASTIERA_LM_MODEL"] = str(p.resolve()) if p.exists() else args.model
        # A revision in .env belongs to the .env model: with --model, only --revision counts.
        os.environ["TASTIERA_LM_REVISION"] = args.revision or ""
    elif args.revision:
        os.environ["TASTIERA_LM_REVISION"] = args.revision
    os.environ.setdefault("TASTIERA_DICT_DIR", str(DICT))
    if args.twins:
        os.environ["TASTIERA_TWINS"] = args.twins
    sys.path.insert(0, str(ROOT / "services" / "ai"))

    from tastiera_ai.pipeline import lexicon as lexmod
    from tastiera_ai.pipeline import lm

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    uses_lm = "lm" in systems or args.bpc_rows > 0
    if uses_lm and not lm.available():
        raise SystemExit(
            "the LM is not available (install the `ml` extra or drop `lm` and use --bpc-rows 0)"
        )

    lex = lexmod.get()
    if lex is None:
        raise SystemExit("no lexicon; run `make dict`")
    ngrams = Ngrams(DICT / "ngrams.tsv")
    name = args.name or Path(lm.MODEL_NAME).name + (
        f"@{lm.MODEL_REVISION}" if lm.MODEL_REVISION else ""
    )

    report: dict = {
        "name": name,
        "model": lm.MODEL_NAME if uses_lm else None,
        "revision": lm.MODEL_REVISION,
        "twin_strategy": os.getenv("TASTIERA_TWINS", "joint"),
        "git": git_commit(),
        "limit": args.limit,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "results": {},
    }
    if uses_lm:
        t0 = time.time()
        lm.warmup()
        report["device"] = lm._state.get("device")
        report["load_seconds"] = round(time.time() - t0, 1)

    for set_name in sets:
        if set_name == "predict":
            rows = load_jsonl("predict", args.limit)
            for s in (x for x in systems if x in PRED_SYSTEMS):
                print(f"[predict] {s} …", flush=True)
                fn = prediction_system(s, lex, ngrams)
                report["results"].setdefault("predict", {})[s] = score_prediction(
                    rows, fn, not args.no_ksr, lex
                )
        else:
            rows = load_jsonl(set_name, args.limit)
            for s, fn in correction_systems(
                [x for x in systems if x in CORR_SYSTEMS], lex, ngrams
            ).items():
                print(f"[{set_name}] {s} …", flush=True)
                report["results"].setdefault(set_name, {})[s] = score_correction(rows, fn)

    if args.bpc_rows > 0:
        print("[bpc] …", flush=True)
        report["lm_intrinsic"] = bits_per_char(args.bpc_rows)

    path = args.out.resolve() if args.out else RESULTS / f"{name.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_summary(report)
    print(f"\n→ {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def print_summary(report: dict) -> None:
    print(f"\n## {report['name']}  (model {report['model']}, git {report['git']})")
    for set_name, res in report["results"].items():
        if set_name == "predict":
            print(
                f"\n{set_name}: system | top1 | top3 | compl@2 | top3 (lex) | compl@2 (lex) | KSR | p50 ms"
            )
            for s, m in res.items():
                print(
                    f"  {s:<14} {m['next_word_top1']:.3f}  {m['next_word_top3']:.3f}  "
                    f"{m['completion_top3_after_2'] or 0:.3f}  {m['in_lexicon_next_word_top3'] or 0:.3f}  "
                    f"{m['in_lexicon_completion_top3_after_2'] or 0:.3f}  "
                    f"{m.get('ksr_top3', float('nan')):.3f}  {m['latency']['p50_ms']}"
                )
            print(
                f"  ({next(iter(res.values()))['in_lexicon_rate']:.0%} of targets are in the lexicon)"
            )
        else:
            print(f"\n{set_name}: system | P | R | F0.5 | FPR | sent acc | twin acc | p50 ms")
            for s, m in res.items():
                twin = m["twin_accuracy"]
                print(
                    f"  {s:<14} {m['precision']:.3f}  {m['recall']:.3f}  {m['f0.5']:.3f}  "
                    f"{m['false_positive_rate'] or 0:.4f}  {m['sentence_accuracy']:.3f}  "
                    f"{'  -  ' if twin is None else f'{twin:.3f}'}  {m['latency']['p50_ms']}"
                )
    if "lm_intrinsic" in report:
        i = report["lm_intrinsic"]
        print(
            f"\nLM: {i['bpc']} bits/char, token ppl {i['ppl_token']}, {i['tokens_per_word']} tokens/word"
        )


if __name__ == "__main__":
    main()
