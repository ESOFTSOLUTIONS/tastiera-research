# dictionary/ — Albanian lexicon, correction table, n-grams and Fjalori

The build scripts download their sources into `sources/` (not committed) and write to `data/`:

```sh
pip install spylls certifi pyarrow
python dictionary/build_lexicon.py   # lexicon.tsv, autocorrect.tsv, hunspell/
python dictionary/build_ngrams.py    # ngrams.tsv (needs the Wikipedia parquet and the held-out split)
python dictionary/build_fjalor.py    # fjalor.jsonl.gz (needs build_lexicon output)
```

| Output | Content |
|---|---|
| `data/lexicon.tsv` | 32.6k valid words with frequencies |
| `data/autocorrect.tsv` | 2,819 safe context-free fixes (eshte→është, nje→një, shtepi→shtëpi, …) |
| `data/hunspell/sq_AL.{dic,aff}` | Hunspell dictionary, full validity check (~1.3M word forms) |
| `data/ngrams.tsv` | On-device next-word model: bigram and trigram followers, 2.6 MB, sorted for binary search |
| `data/fjalor.jsonl.gz` | Dictionary & thesaurus: ~125k headwords, ~5.9k Albanian definitions, ~8.4k synonym sets, ~13.5k English glosses with inflection, IPA and etymology, ~9.7k inflected forms linked to their lemma |

## Correction rules

- **Invalid word**: corrected to the most frequent valid word with the same letters ignoring ë/ç; otherwise to the closest word at edit distance 1.
- **Valid but rare variant**: corrected only if the diacritic form is at least 5× more frequent (eshte→është, nje→një).
- **Diacritics are only added automatically, never removed.**
- **Ambiguous twins** (`ne/në`, `te/të`, `se/së`, `me/më`, …) are real words in both forms and are never in the table; the language model decides them from context, with a margin.

## Sources & licences

| Source | Licence | Role |
|---|---|---|
| [bgodole/korrektor](https://github.com/bgodole/korrektor): Hunspell sq_AL | GPL-3.0 | Which words are valid |
| [hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords) sq_50k (OpenSubtitles 2018) | data CC BY-SA 4.0 | Word frequencies |
| [Wiktionary via kaikki.org](https://kaikki.org/dictionary/Albanian/) | CC BY-SA 4.0 | English meanings, inflections, IPA, etymology |
| [AlbaNet, Open Multilingual Wordnet 1.4](https://github.com/omwn/omw-data) | GPL-3.0 or CC BY 3.0 | Albanian definitions, examples, synonym sets |
| [Albanian Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.sq) | CC BY-SA 4.0 | Word sequences for `ngrams.tsv` |

The derived `data/` files inherit these terms (see the top-level README).
