#!/usr/bin/env python3
"""Build the Tastiera Albanian dictionary & thesaurus (Fjalori) from open sources.

Sources (downloaded into dictionary/sources/, not committed):
  - Hunspell sq_AL headwords (bgodole/korrektor, GPL-3.0): ~110k lemmas → the word list
  - Wiktionary via kaikki.org (CC-BY-SA-4.0): English meanings, inflections, IPA, etymology
  - AlbaNet, the Albanian WordNet (Open Multilingual Wordnet 1.4; GPL-3 or CC-BY-3.0):
    Albanian definitions, examples and synonym sets (the thesaurus)
  - data/lexicon.tsv (build_lexicon.py): word frequencies

Output: data/fjalor.jsonl.gz — one JSON object per headword:
  {"w": word, "f": freq, "pos": [...], "ipa": str, "def": [{"t": sq definition,
   "ex": [examples], "syn": [synonyms], "pos": p}], "en": [{"pos": p, "g": [glosses]}],
   "forms": [[form, "tags"]], "ety": str, "of": headword (inflected-form stubs),
   "src": ["hunspell", "wiktionary", "albanet"]}

Usage: python build_fjalor.py   (run build_lexicon.py first)
"""

from __future__ import annotations

import gzip
import json
import re
import ssl
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sources"
OUT = ROOT / "data"

KAIKKI = "https://kaikki.org/dictionary/Albanian/kaikki.org-dictionary-Albanian.jsonl"
OMW = "https://github.com/omwn/omw-data/releases/download/v1.4/omw-1.4.tar.xz"

POS_SQ = {
    "noun": "emër", "n": "emër", "verb": "folje", "v": "folje", "adj": "mbiemër", "a": "mbiemër",
    "s": "mbiemër", "adv": "ndajfolje", "r": "ndajfolje", "name": "emër i përveçëm",
    "pron": "përemër", "num": "numëror", "intj": "pasthirrmë", "prep": "parafjalë",
    "conj": "lidhëz", "particle": "pjesëz", "phrase": "frazë",
}
SKIP_FORM_TAGS = {"table-tags", "inflection-template", "class"}
MAX_FORMS = 48


def _ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print("download", url)
        with urllib.request.urlopen(url, context=_ctx()) as r, open(dest, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    return dest


def albanet_xml() -> Path:
    out = SRC / "omw-sq.xml"
    if not out.exists():
        tar = download(OMW, SRC / "omw-1.4.tar.xz")
        with tarfile.open(tar) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("omw-sq/omw-sq.xml"))
            out.write_bytes(t.extractfile(member).read())
    return out


def load_autocorrect() -> dict[str, str]:
    fixes = {}
    path = OUT / "autocorrect.tsv"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                a, b = line.split("\t")
                fixes[a] = b
    return fixes


def normalizer(fixes: dict[str, str]):
    """Restore missing ë/ç in AlbaNet text using the safe autocorrect table."""

    def fix_word(m: re.Match) -> str:
        w = m.group(0)
        r = fixes.get(w.lower())
        if r is None:
            return w
        return r[0].upper() + r[1:] if w[0].isupper() else r

    return lambda text: re.sub(r"[^\W\d_]+", fix_word, text)


def main() -> None:
    SRC.mkdir(exist_ok=True)
    norm = normalizer(load_autocorrect())

    freq: dict[str, int] = {}
    for line in (OUT / "lexicon.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            w, f = line.split("\t")
            freq[w] = int(f)

    entries: dict[str, dict] = {}

    def entry(word: str) -> dict:
        e = entries.get(word)
        if e is None:
            e = entries[word] = {"w": word, "src": []}
        return e

    def add_src(e: dict, s: str) -> None:
        if s not in e["src"]:
            e["src"].append(s)

    # 1) Hunspell headwords: the complete word list.
    dic = (OUT / "hunspell" / "sq_AL.dic").read_text(encoding="utf-8").splitlines()[1:]
    for line in dic:
        word = line.split("/", 1)[0].strip()
        if word and word.replace("-", "").replace(" ", "").isalpha():
            add_src(entry(word), "hunspell")

    # 2) Wiktionary (kaikki): meanings in English, forms, IPA, etymology, synonyms.
    with open(download(KAIKKI, SRC / "kaikki-sq.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            k = json.loads(line)
            word, pos = k.get("word", ""), k.get("pos", "")
            if not word or pos in {"character", "suffix", "prefix", "symbol", "romanization"}:
                continue
            senses = k.get("senses", [])
            bases = [f["word"] for sn in senses for f in sn.get("form_of", []) if f.get("word")]
            if senses and len(bases) >= len(senses):
                # Pure inflection stub ("definite plural of shtëpi"): link to the headword.
                e = entry(word)
                add_src(e, "wiktionary")
                e.setdefault("of", bases[0])
                continue
            e = entry(word)
            add_src(e, "wiktionary")
            p = POS_SQ.get(pos, pos)
            e.setdefault("pos", [])
            if p not in e["pos"]:
                e["pos"].append(p)
            glosses = []
            syn = set()
            for s in k.get("senses", []):
                g = "; ".join(s.get("glosses", [])[-1:])
                if g and g not in glosses:
                    glosses.append(g)
                syn |= {x["word"] for x in s.get("synonyms", []) if "word" in x}
            syn |= {x["word"] for x in k.get("synonyms", []) if "word" in x}
            if glosses:
                e.setdefault("en", []).append({"pos": p, "g": glosses[:10]})
            if syn:
                e["_wsyn"] = sorted(set(e.get("_wsyn", [])) | syn)
            if "ipa" not in e:
                ipa = next((s["ipa"] for s in k.get("sounds", []) if s.get("ipa")), None)
                if ipa:
                    e["ipa"] = ipa
            if "ety" not in e and k.get("etymology_text"):
                e["ety"] = k["etymology_text"][:600]
            forms = e.setdefault("forms", [])
            seen = {f[0] + f[1] for f in forms}
            for f in k.get("forms", []):
                tags = [t for t in f.get("tags", []) if t not in SKIP_FORM_TAGS]
                form = f.get("form", "")
                if not form or not tags or set(f.get("tags", [])) & SKIP_FORM_TAGS:
                    continue
                key = form + " ".join(tags)
                if key not in seen and len(forms) < MAX_FORMS:
                    forms.append([form, " ".join(tags)])
                    seen.add(key)
            if not forms:
                e.pop("forms")

    # 3) AlbaNet: Albanian definitions, examples, synonym sets.
    root = ET.parse(albanet_xml()).getroot()
    lemma_of_sense: dict[str, str] = {}
    lemmas_of_synset: dict[str, list[str]] = defaultdict(list)
    for le in root.iter("LexicalEntry"):
        lemma = le.find("Lemma").get("writtenForm").strip()
        for s in le.iter("Sense"):
            lemma_of_sense[s.get("id")] = lemma
            lemmas_of_synset[s.get("synset")].append(lemma)
    for ss in root.iter("Synset"):
        sid = ss.get("id")
        members = [norm(m) for m in lemmas_of_synset.get(sid, [])]
        d = ss.find("Definition")
        text = norm(d.text.strip()) if d is not None and d.text else ""
        examples = [norm(x.text.strip()) for x in ss.iter("Example") if x.text]
        pos = POS_SQ.get(ss.get("partOfSpeech"), ss.get("partOfSpeech"))
        for lemma in members:
            if " " in lemma and lemma not in entries and not text:
                continue
            e = entry(lemma)
            add_src(e, "albanet")
            others = [m for m in members if m != lemma]
            if text or others:
                e.setdefault("def", []).append({"t": text, "ex": examples[:3], "syn": others, "pos": pos})

    # 4) Finish: frequencies, merged synonyms, drop internal fields.
    out = []
    for w, e in entries.items():
        f = freq.get(w) or freq.get(w.lower())
        if f:
            e["f"] = f
        wsyn = e.pop("_wsyn", None)
        if wsyn:
            # Wiktionary synonyms not already covered by an AlbaNet synset.
            have = {s for d in e.get("def", []) for s in d["syn"]}
            extra = [s for s in wsyn if s not in have and s != w]
            if extra:
                e.setdefault("def", []).append({"t": "", "ex": [], "syn": extra, "pos": ""})
        out.append(e)
    out.sort(key=lambda e: e["w"].lower())

    dest = OUT / "fjalor.jsonl.gz"
    with gzip.open(dest, "wt", encoding="utf-8", compresslevel=9) as fh:
        for e in out:
            fh.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")
    n_def = sum(1 for e in out if any(d["t"] for d in e.get("def", [])))
    n_en = sum(1 for e in out if e.get("en"))
    n_syn = sum(1 for e in out if any(d["syn"] for d in e.get("def", [])))
    n_of = sum(1 for e in out if "of" in e)
    print(
        f"fjalor: {len(out)} headwords; sq definitions {n_def}; en meanings {n_en}; "
        f"with synonyms {n_syn}; inflection links {n_of}; {dest.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
