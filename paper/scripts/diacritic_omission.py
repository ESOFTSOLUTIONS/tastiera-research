#!/usr/bin/env python3
"""How often is ë/ç omitted in informal Albanian? (paper, Introduction)

For every safe pair in the correction table (plain form -> diacritic form,
e.g. eshte -> është) count how often each form occurs. The omission rate is
plain / (plain + diacritic), pooled over all pairs. Because the plain forms are not
valid words, every occurrence of one is a word typed without its ë/ç.

Inputs: SHAJ comments (strombergnlp/shaj, CC BY 4.0; YouTube/Instagram comments) and,
for contrast, the reference sentences of T1 (Wikipedia).

Usage:
  python3 -I paper/scripts/diacritic_omission.py AUTOCORRECT_TSV SHAJ_CSV T1_JSONL
"""

import csv
import json
import re
import sys
from collections import Counter

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
table_path, shaj_path, t1_path = sys.argv[1:4]

pairs = {}
for line in open(table_path, encoding="utf-8"):
    if line.startswith("#") or "\t" not in line:
        continue
    plain, diac = line.rstrip("\n").split("\t")[:2]
    pairs[plain.lower()] = diac.lower()
diac_forms = set(pairs.values())


def measure(texts):
    c = Counter()
    n_texts = n_with = n_all_plain = 0
    for t in texts:
        ws = [w.lower() for w in WORD.findall(t)]
        p = sum(w in pairs for w in ws)
        d = sum(w in diac_forms for w in ws)
        c["plain"] += p
        c["diac"] += d
        n_texts += 1
        if p + d:
            n_with += 1
            n_all_plain += d == 0
    rate = c["plain"] / (c["plain"] + c["diac"])
    return {
        "texts": n_texts,
        "plain": c["plain"],
        "diacritic": c["diac"],
        "omission_rate": round(rate, 4),
        "texts_with_table_word": n_with,
        "share_texts_all_plain": round(n_all_plain / n_with, 4),
    }


with open(shaj_path, encoding="utf-8", newline="") as f:
    shaj = [row[0] for row in csv.reader(f, delimiter=";") if row and row[0].strip()]
t1 = [json.loads(line)["target"] for line in open(t1_path, encoding="utf-8") if line.strip()]

out = {"pairs": len(pairs), "shaj": measure(shaj), "t1_wikipedia": measure(t1)}
print(json.dumps(out, indent=2, ensure_ascii=False))
