#!/usr/bin/env python3
"""Build the on-device next-word model (word bigrams + trigrams) for the keyboards.

Source (downloaded into dictionary/sources/, not committed):
  - Albanian Wikipedia, wikimedia/wikipedia 20231101.sq (CC BY-SA 4.0)

Paragraphs of the LM corpus's held-out split (ml/data/out/sq-corpus/validation.parquet,
built by ml/data/build_corpus.py) are skipped, so the n-gram baseline never sees the
evaluation sentences (ml/data/build_eval_sets.py). Build the corpus first; without it the
script warns and uses everything.

Only words of data/lexicon.tsv are kept (valid spelling, with ë/ç), so the model can
never suggest a misspelling. Each context keeps its top candidates only, to stay a few MB
inside the iOS keyboard-extension memory limit.

Output (committed, bundled in the keyboards):
  data/ngrams.tsv   context<TAB>word count word count …   (best first)
                    context = "<s>" (sentence start) | "w1" (bigram) | "w1 w2" (trigram)

Usage:  python build_ngrams.py     (needs: pip install pyarrow huggingface_hub)
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sources"
OUT = ROOT / "data"

WIKI_REPO = "wikimedia/wikipedia"
WIKI_FILE = "20231101.sq/train-00000-of-00001.parquet"

START = "<s>"
BIGRAM_TOP, BIGRAM_MIN = 8, 3  # candidates kept per previous word, min count
TRIGRAM_TOP, TRIGRAM_MIN = 5, 4
TRIGRAM_CONTEXT_MIN = 40  # only contexts "w1 w2" seen this often get trigram entries
SINGLE_LETTER_WORDS = {"a", "e", "i", "o", "u"}  # drop fragments like "t" (from "t'i")

SENTENCE = re.compile(r"[.!?;:\n]+|\s[-–—]\s")
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


def fetch_wikipedia() -> Path:
    dest = SRC / "wikipedia_sq_20231101.parquet"
    if not dest.exists():
        from huggingface_hub import hf_hub_download

        SRC.mkdir(exist_ok=True)
        print(f"downloading {WIKI_REPO}/{WIKI_FILE} …")
        path = hf_hub_download(WIKI_REPO, WIKI_FILE, repo_type="dataset")
        dest.symlink_to(path)
    return dest


def load_vocab() -> set[str]:
    vocab = set()
    for line in (OUT / "lexicon.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            vocab.add(line.split("\t", 1)[0])
    return {w for w in vocab if len(w) > 1 or w in SINGLE_LETTER_WORDS}


HELD_OUT = ROOT.parent / "ml" / "data" / "out" / "sq-corpus" / "validation.parquet"


def para_key(para: str) -> bytes:
    """Same normalisation and hash as ml/data/build_corpus.py."""
    return hashlib.md5(" ".join(para.split()).encode()).digest()


def held_out() -> set[bytes]:
    if not HELD_OUT.exists():
        print(f"warning: {HELD_OUT} missing — evaluation paragraphs are NOT excluded")
        return set()
    import pyarrow.parquet as pq

    keys = {para_key(t) for t in pq.read_table(HELD_OUT, columns=["text"]).column(0).to_pylist()}
    print(f"excluding {len(keys)} held-out paragraphs")
    return keys


def sentences(parquet: Path, exclude: set[bytes]):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["text"])
    skipped = 0
    for batch in table.to_batches(max_chunksize=2000):
        for article in batch.column(0).to_pylist():
            for para in article.split("\n"):
                if exclude and para_key(para) in exclude:
                    skipped += 1
                    continue
                # skip lists, tables, references and other non-prose lines
                if (
                    len(para) < 60
                    or para.count("|") > 1
                    or para.lstrip().startswith(("*", "{", "<"))
                ):
                    continue
                yield from SENTENCE.split(para)
    print(f"skipped {skipped} held-out paragraphs")


def main() -> None:
    t0 = time.time()
    vocab = load_vocab()
    print(f"lexicon: {len(vocab)} words")
    index = {w: i for i, w in enumerate(sorted(vocab))}
    words = sorted(vocab)
    s_id = len(words)
    words.append(START)
    n = len(words)

    # Pass 1: bigrams (+ sentence starts). Words outside the lexicon break the chain, so
    # no n-gram spans an unknown word, number or name we cannot suggest.
    bi: Counter[int] = Counter()
    tokens = 0
    sents = []
    for sent in sentences(fetch_wikipedia(), held_out()):
        ids = [index.get(w.lower(), -1) for w in WORD.findall(sent)]
        if len(ids) < 2:
            continue
        tokens += len(ids)
        prev = s_id
        for i in ids:
            if i >= 0 and prev >= 0:
                bi[prev * n + i] += 1
            prev = i
        sents.append(ids)
    print(
        f"{len(sents)} sentences, {tokens / 1e6:.1f}M tokens, {len(bi)} bigram types "
        f"({time.time() - t0:.0f}s)"
    )

    # Pass 2: trigrams only for frequent two-word contexts.
    hot = {k for k, c in bi.items() if c >= TRIGRAM_CONTEXT_MIN}
    tri: Counter[tuple[int, int]] = Counter()
    for ids in sents:
        a, b = -1, s_id
        for i in ids:
            if a >= 0 and b >= 0 and i >= 0 and (a * n + b) in hot:
                tri[(a * n + b, i)] += 1
            a, b = b, i
    print(
        f"{len(hot)} trigram contexts, {len(tri)} trigram types ({time.time() - t0:.0f}s)"
    )

    def top(counts, k_top, k_min):
        by_ctx: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key, c in counts:
            if c >= k_min:
                by_ctx[key[0]].append((c, key[1]))
        return {ctx: sorted(v, reverse=True)[:k_top] for ctx, v in by_ctx.items()}

    bigrams = top(((divmod(k, n), c) for k, c in bi.items()), BIGRAM_TOP, BIGRAM_MIN)
    trigrams = top(tri.items(), TRIGRAM_TOP, TRIGRAM_MIN)

    def name(ctx: int) -> str:
        a, b = divmod(ctx, n)
        return words[a] if a != s_id else START, words[b]

    lines = []
    for prev, cands in bigrams.items():
        lines.append(words[prev] + "\t" + " ".join(f"{words[w]} {c}" for c, w in cands))
    kept_tri = 0
    for ctx, cands in trigrams.items():
        a, b = name(ctx)
        if a == START:
            continue
        # Skip trigrams that only repeat the bigram ranking of the last word.
        bi_top = [words[w] for _, w in bigrams.get(index[b], [])][: len(cands)]
        if [words[w] for _, w in cands] == bi_top:
            continue
        lines.append(f"{a} {b}\t" + " ".join(f"{words[w]} {c}" for c, w in cands))
        kept_tri += 1
    # Sorted by the UTF-8 bytes of the context: the keyboard memory-maps the file and
    # binary-searches it instead of loading it into a dictionary.
    lines.sort(key=lambda l: l.split("\t", 1)[0].encode())
    header = (
        "# context\tword count … — generated by dictionary/build_ngrams.py from Albanian "
        "Wikipedia (CC BY-SA 4.0) minus the LM held-out split; sorted by context bytes"
    )
    path = OUT / "ngrams.tsv"
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {path.relative_to(ROOT)}: {len(bigrams)} bigram contexts, {kept_tri} trigram "
        f"contexts, {path.stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
