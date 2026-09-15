# Beam-search attack on STA-1 — working notes

## Detector verification

`research/attacks/sta_core.py` reproduces the official
`SampleWatermarkDetector` from `sample_watermark/` exactly:
error 0.00e+00 on all five baseline samples.

Confirm with:

    python research/attacks/verify_sta_core.py

Note on T: the official detector uses T = the number of
adjacent token PAIRS in the z-score, since `T = len(result)`
and `result` holds one entry per pair. The paper's Eq. (3)
writes T as the token count. The two differ by one. The code
is authoritative and this implementation follows it.

## Pipeline

| module | role |
| --- | --- |
| `sta_core.py` | detector, cached green-pair oracle |
| `edit_ops.py` | token-level edits, decode/re-encode |
| `candidates.py` | vulnerability scan, grammatical filters |
| `beam.py` | beam search, semantic and grammar gates |
| `run_attack.py` | entry point |
| `measure_ops.py` | per-operation predicted vs actual |

Edits are applied at the token level, decoded to text, then
re-tokenised before scoring. A real detector receives text,
not token ids.

## Status

**Sample 3 (prompt_id 2)** reaches z = 1.9635 from a baseline
of 3.0496 using 9 substitutions, cosine similarity 0.9990,
no new LanguageTool errors. Statistically below the
detection threshold of 2.0.

**This is not yet a clean result.** Manual reading found 2 of
the 9 edits ungrammatical:

- `details->detail` — "Caveney detail his formative years"
- `matters->matter` — "These are also matter, however"

Both are subject-verb disagreements. Cause: WordNet lemma
names are base forms, so an inflected source word returns its
own uninflected form as a "synonym". Fix pending — match
spaCy `tag_` between source and replacement.

Two further edits are semantically off: `companion->familiar`
and `reaches->hit`.

**Sample 2 (prompt_id 1)** not yet run under the current
filter set.

## Findings that hold

**1. Embedding similarity is a weak quality gate.**
At depth 2, 1150 states passed cosine >= 0.95 while only 400
passed grammar checking. Roughly two thirds of edits judged
semantically acceptable by MiniLM were grammatically broken.
Sentence embeddings measure topical overlap and are close to
blind to agreement errors and wrong word sense.

**2. Local prediction held exactly for the filtered pool.**
All 122 single operations measured on Sample 3 matched their
locally predicted green-count change. Substitutions and
word-initial deletions round-trip without re-segmentation.

An earlier hypothesis — that re-tokenisation ripple was a
significant attack surface against the locality claim in
Section 4.4 — is **not supported** for this edit class. It
remains untested for subword deletions and multi-token
substitutions.

## Open theoretical point

Theorem 3 treats the green indicators as independent
Bernoulli variables and sums their variances. Adjacent pairs
share a token, so they are dependent, and every covariance
term is dropped. The null distribution of the z-test is
therefore misspecified and the type II bound in Corollary 1
does not follow as derived. Not yet quantified.

## Reproducing

    python research/attacks/verify_sta_core.py
    python research/attacks/scan_samples.py
    python research/attacks/measure_ops.py 2
    python research/attacks/run_attack.py 2 --depth 12

Requires `nltk` with wordnet, `spacy` with `en_core_web_sm`,
`language-tool-python` (needs Java), and a local Llama-2-7B
tokenizer at `hf_models/Llama-2-7b-hf`.
