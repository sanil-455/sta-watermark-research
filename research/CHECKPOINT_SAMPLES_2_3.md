# STA Watermark Research — Samples 2 & 3 Checkpoint

## Checkpoint purpose

This checkpoint preserves the complete targeted meaningful-word substitution
attack work performed on baseline Samples 2 and 3.

The repository should be sufficient to resume the research without rerunning
the completed attacks.

---

# Baseline

Detection threshold:

    z > 2.0

## Sample 2

Baseline watermarked z:

    2.8160458729081004

Baseline detection:

    TRUE

## Sample 3

Baseline watermarked z:

    3.049618749953805

Baseline detection:

    TRUE

---

# Attack 1 — Individual meaningful-word substitution

## Sample 2

13 individual replacement experiments completed.

Errors:

    0

Watermark breaks:

    0

Strongest observed individual reduction:

    releases -> disengages

    Δz = -0.145975

Final z:

    2.670071

Detection:

    TRUE -> TRUE

## Sample 3

12 individual replacement experiments completed.

Errors:

    0

Watermark breaks:

    0

Strongest observed individual reduction:

    interest -> appreciation

    Original z = 3.049618749953805
    Attacked z = 2.8382651618409196
    Δz = -0.2113535881128854

Detection:

    TRUE -> TRUE

---

# Attack 2 — Exhaustive coordinated substitutions

## Sample 2

Completed combinations:

    96

Errors:

    0

Watermark breaks:

    0

Best observed combination:

    separation -> severance
    growth -> demand
    capacity -> output
    releases -> disengages
    making -> achieving
    some -> significant

Original z:

    2.8160458729081004

Attacked z:

    2.382245750691506

Δz:

    -0.43380012221659436

Detection:

    TRUE -> TRUE

## Sample 3

Completed combinations:

    36

Errors:

    0

Watermark breaks:

    0

Best observed combination:

    interest -> appreciation
    away -> far
    times -> moments
    stretch -> strain
    reaches -> attains
    However -> Nonetheless

Original z:

    3.049618749953805

Attacked z:

    2.246103893180798

Δz:

    -0.803514856773007

Detection:

    TRUE -> TRUE

---

# Overall conclusion at this checkpoint

No tested individual meaningful-word substitution changed detection:

    TRUE -> FALSE

No tested coordinated substitution changed detection:

    TRUE -> FALSE

Therefore this attack family has NOT broken STA on Samples 2 or 3.

The strongest result so far is Sample 3's coordinated attack:

    z = 3.049619 -> 2.246104
    Δz = -0.803515

This is a substantial reduction but remains above the detection threshold.

This conclusion is scoped to the tested Samples 2 and 3 and the tested
replacement vocabulary. It is NOT a claim that STA is generally robust.

---

# Important observations to preserve

Several different lexical substitutions produced exactly the same final z.

This is consistent with the fact that the detector ultimately uses an aggregate
green-token count rather than preserving the complete positional structure.

This observation should be investigated mathematically later rather than
treated as proof of a vulnerability.

The coordinated attack also demonstrates that individually non-breaking edits
can collectively produce a substantially larger reduction in z.

---

# Completed files

Important directories:

    results/raw/manual_attacks/sample2_replacements/
    results/raw/manual_attacks/sample3_replacements/

    results/raw/manual_attacks/sample2_combined/
    results/raw/manual_attacks/sample3_combined/

Break extraction:

    results/raw/manual_attacks/BREAKS/

Important summaries:

    results/raw/manual_attacks/sample2_combined_best_and_summary.json
    results/raw/manual_attacks/sample3_combined_best_and_summary.json
    results/raw/manual_attacks/sample2_best_single_attack.json
    results/raw/manual_attacks/sample3_best_single_attack.json
    results/raw/manual_attacks/samples2_3_adaptive_substitution_checkpoint_data.json

---

# Attack scripts

Individual attacks:

    research/attacks/attack_sample2_replacements.py
    research/attacks/attack_sample3_replacements.py

Combined attack:

    research/attacks/analyze_and_combined_sample2_3.py

Break extraction:

    research/attacks/extract_watermark_breaks.py

---

# Next research directions

The meaningful lexical substitution attack should NOT be treated as a
successful STA break.

Potential next attack families:

1. Adaptive minimum-edit attack
2. Token-level targeted attack
3. Capitalization / orthographic changes
4. Insertion attacks
5. Deletion attacks
6. Clustered versus dispersed edits
7. Copy-paste attack
8. Paraphrase attacks
9. Attacks explicitly exploiting overlapping STA token pairs
10. Combined adaptive attacks

Before moving on, preserve this checkpoint in Git.

