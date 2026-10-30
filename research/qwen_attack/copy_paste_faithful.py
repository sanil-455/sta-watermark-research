"""
Item 3, corrected: the paper's OWN copy-paste attack, reimplemented from its source
(attack/copy_paste_attack.py, single_insertion()), not my earlier guess at "standard
literature practice" (which was a different, block-prepend operation -- see COPY_PASTE.md
for that one, kept as a separate, honestly-labelled experiment).

The paper's actual method: for each flagged span, take the SAME-PROMPT unwatermarked
control (not generic corpus text), and replace k = min(round(ratio * len(span_tokens)),
min_attack_length) RANDOMLY CHOSEN positions within the first min_attack_length=100
tokens of the watermarked span with the control's tokens at those same positions. Pairs
where either side has fewer than min_attack_length tokens are skipped entirely, exactly
as the paper's code does (`if min(len_wm, len_no_wm) < min_attack_length: continue`).
Random positions use Python's random module with a fixed seed (not torch's RNG, so not
bit-identical to the paper's own run, but the same algorithm and the same guarantees:
a uniformly random k-subset of the first 100 positions).

CPU-only: needs the tokenizer and the unwatermarked controls you already generated
(results/raw/controls_len200.json / controls_len500.json), no GPU, no paraphraser.
Only spans whose prompt_id has a matching control of the same length are testable;
this is reported explicitly, never silently padded or guessed.

  python research/qwen_attack/copy_paste_faithful.py

Writes results/qwen_attack/COPY_PASTE_FAITHFUL.md and .csv.
"""

import csv
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RATIOS = (0.10, 0.25, 0.50)
MIN_ATTACK_LENGTH = 100
SEED = 42


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def single_insertion(rng, ratio, span_ids, control_ids):
    """Faithful to the paper's single_insertion(): same algorithm, Python RNG not torch's."""
    pool = min(MIN_ATTACK_LENGTH, len(span_ids), len(control_ids))
    if min(len(span_ids), len(control_ids)) < MIN_ATTACK_LENGTH:
        return None   # the paper's code: `continue`, i.e. this pair is not attacked at all
    attack_len = min(int(round(ratio * len(span_ids))), MIN_ATTACK_LENGTH)
    positions = rng.sample(range(pool), attack_len)
    out = list(span_ids)
    for i in positions:
        out[i] = control_ids[i]
    return out


def latest_full_runs(root):
    found = {}
    for d in sorted(Path(root).glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if not run["args"].get("only"):
            found["500" if any("len500" in f for f in run["args"].get("files", [])) else "200"] = d
    return found


def analyse(label, d, samples, controls, tok, score_fn, rng):
    flagged = {x["sample_id"]: x["z0"] for x in json.load(open(d / "screening.json")) if x["flagged"]}
    span_by_prompt = {s.prompt_id: s for s in samples if s.sample_id in flagged}
    ctrl_by_prompt = {r["prompt_id"]: r["text"] for r in controls}
    matched = sorted(set(span_by_prompt) & set(ctrl_by_prompt))
    n_flagged = len(flagged)
    L = ["## %s (%d flagged spans; %d have a same-prompt unwatermarked control to draw from)\n" % (label, n_flagged, len(matched)),
         "| insertion ratio | min_attack_length | tested | too-short pairs skipped | still detected (z>2.0) | detection rate (95%% interval) |", "|---|---|---|---|---|---|"]
    rows = []
    for r in RATIOS:
        k_detected, tested, skipped = 0, 0, 0
        for pid in matched:
            s = span_by_prompt[pid]
            span_ids = tok(s.span, add_special_tokens=False)["input_ids"]
            ctrl_ids = tok(ctrl_by_prompt[pid], add_special_tokens=False)["input_ids"]
            out = single_insertion(rng, r, span_ids, ctrl_ids)
            if out is None:
                skipped += 1
                continue
            tested += 1
            text = tok.decode(out, skip_special_tokens=True)
            if score_fn(text) > 2.0:
                k_detected += 1
        p, lo, hi = wilson(k_detected, tested)
        L.append("| %.0f%% | %d | %d | %d | %d of %d | %s |" % (100 * r, MIN_ATTACK_LENGTH, tested, skipped, k_detected, tested,
                  ("%.0f%% [%.0f%%, %.0f%%]" % (p, lo, hi)) if tested else "n/a (no matched pairs)"))
        rows.append([label, r, tested, skipped, k_detected, "%.1f" % p, "%.1f" % lo, "%.1f" % hi])
    return L + [""], rows, len(matched), n_flagged


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core")]
    import samples
    import sta_core
    tok = sta_core.load_tokenizer()
    score_fn = lambda t: sta_core.score_text(tok, t)["z"]
    groups = latest_full_runs(ROOT / "results" / "qwen_attack")
    doc = ["# Copy-paste attack, faithful to the paper's own code (attack/copy_paste_attack.py, single_insertion)\n",
           "Reimplemented from the paper's source, not a guess at 'standard practice': random positions within the first %d tokens of the "
           "watermarked span are replaced with the SAME-PROMPT unwatermarked control's tokens at those positions (not generic corpus text, "
           "and not a prepended block -- see COPY_PASTE.md for that separate, differently-defined experiment). Pairs where either text has "
           "fewer than %d tokens are skipped entirely, exactly as the paper's code does. Random positions use Python's RNG (seed %d), "
           "not torch's, so counts will differ run to run from the paper's own numbers even under the same algorithm; only spans whose "
           "prompt also has a usable unwatermarked control (from generate_controls.py) can be tested here. Lengths are never pooled.\n"
           % (MIN_ATTACK_LENGTH, MIN_ATTACK_LENGTH, SEED)]
    rows = []
    for key in ("200", "500"):
        if key not in groups:
            continue
        cf = Path("results/raw/controls_len%s.json" % key)
        if not cf.exists():
            doc.append("## %s-token spans\n\nno controls_len%s.json found; run generate_controls.py first.\n" % (key, key))
            continue
        controls = json.load(open(cf))
        files = [f for f in json.load(open(groups[key] / "run.json"))["args"]["files"] if os.path.exists(f)]
        rng = random.Random(SEED)
        L, r, matched_n, flagged_n = analyse("%s-token spans" % key, groups[key], samples.load_samples(files), controls, tok, score_fn, rng)
        doc += L
        rows += r
        if matched_n < flagged_n:
            doc.append("*Only %d of %d flagged spans had a same-prompt control; the rest are not covered by this experiment.*\n" % (matched_n, flagged_n))
    out = ROOT / "results" / "qwen_attack"
    (out / "COPY_PASTE_FAITHFUL.md").write_text("\n".join(doc) + "\n")
    with open(out / "COPY_PASTE_FAITHFUL.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "ratio", "tested", "skipped_too_short", "detected", "rate_pct", "ci_low", "ci_high"])
        w.writerows(rows)
    print("\n".join(doc))
    print("written:", out / "COPY_PASTE_FAITHFUL.md")


if __name__ == "__main__":
    main()
