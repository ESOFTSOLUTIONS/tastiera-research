# Tastiera research release

Code, data and evaluation for the paper

> **Tastiera: A Hybrid Lexicon and Language-Model Approach to Diacritic Restoration,
> Spelling Correction and Word Prediction for Albanian.**
> Eledi Dyrkaj and Freda Dyrkaj, 2026. *(arXiv link to follow.)*

Albanian is routinely typed without the letters **ë** and **ç**. Most such errors can be
fixed with a dictionary, but several of the most frequent words exist in both forms
(*ne/në*, *te/të*, *se/së*, *me/më*), so the right spelling depends on the sentence.
This repository contains the lexical resources, the test sets, the evaluation and
scoring code, and the result files behind the paper's tables. The correction pipeline
itself (§5 of the paper) is part of the Tastiera product and is not included.

## Contents

| Path | What |
|---|---|
| `dictionary/data/` | Lexical resources: frequency lexicon (32.6k words), safe correction table (2,819 rules), on-device n-gram model (2.6 MB), Fjalori dictionary & thesaurus (125k headwords), Hunspell `sq_AL` |
| `dictionary/build_*.py` | Scripts that rebuild the resources from their open sources |
| `ml/data/out/eval/` | Test sets: `diacritics.jsonl` (T1), `twins.jsonl` (T2), `predict.jsonl` (next-word prediction) |
| `ml/data/out/sq-corpus/validation.parquet` | Held-out Wikipedia paragraphs the test sets are drawn from |
| `ml/data/build_*.py` | Scripts that build the corpus and the test sets |
| `ml/eval/eval_lm.py` | Evaluation of all systems (precision, recall, F0.5, false-positive rate, twin accuracy, next-word accuracy, keystroke savings, bits per character, latency) |
| `ml/eval/twin_margins.py` | Per-token language-model margins for the ambiguous twins |
| `paper/scripts/` | Ablations, false-positive inspection, diacritic-omission rate, resource statistics |
| `paper/results/` | The result files behind the paper's tables, including the hand-labelled false positives |

## Evaluation

`ml/eval/eval_lm.py` defines every metric reported in the paper (word-level precision,
recall, F0.5, false-positive rate on correct words, sentence accuracy, twin accuracy,
next-word top-k accuracy, keystroke savings, bits per character, latency); see
`score_correction()` for the exact token alignment and counting. The scripts import the
correction pipeline (`tastiera_ai.pipeline`), which is not part of this release, so they
document the evaluation rather than run stand-alone. The outputs they produced are in
`paper/results/`.

Each test-set line is a JSON object with the system `input` and the reference `target`
(T1, T2) or a context and the next word (prediction set); any system can be scored
against them with the metric definitions above.

## Licences

The **code** in this repository (Python files) is released under the **GNU General Public License v3.0**
(`LICENSE`). Each **data** resource carries the licence of its sources:

| Resource | Derived from | Licence |
|---|---|---|
| `dictionary/data/lexicon.tsv`, `autocorrect.tsv`, `hunspell/` | Hunspell `sq_AL` from [bgodole/korrektor](https://github.com/bgodole/korrektor) (GPL-3.0); [FrequencyWords](https://github.com/hermitdave/FrequencyWords) OpenSubtitles 2018 frequencies (CC BY-SA 4.0) | GPL-3.0 |
| `dictionary/data/fjalor.jsonl.gz` | Hunspell lemmas (GPL-3.0); [AlbaNet / Open Multilingual Wordnet](https://github.com/omwn/omw-data) (GPL-3.0 or CC BY 3.0); [Wiktionary via kaikki.org](https://kaikki.org/dictionary/Albanian/) (CC BY-SA 4.0) | GPL-3.0, with CC BY-SA 4.0 attribution for Wiktionary content |
| `dictionary/data/ngrams.tsv`, `ml/data/out/`, test sets | [Albanian Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 2023-11-01 (CC BY-SA 4.0) | CC BY-SA 4.0 |

## Citation

```bibtex
@misc{dyrkaj2026tastiera,
  title  = {{Tastiera}: A Hybrid Lexicon and Language-Model Approach to Diacritic Restoration,
            Spelling Correction and Word Prediction for {Albanian}},
  author = {Dyrkaj, Eledi and Dyrkaj, Freda},
  year   = {2026},
  note   = {arXiv preprint}
}
```

Tastiera is a product of [E-Soft Solutions](https://e-soft.al). This repository is a
frozen research release accompanying the paper.
