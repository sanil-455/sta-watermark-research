# Final results after the repair step

Every number is read from the run folders. The attack sees only the generated span; a span counts as broken when its z is 2.0 or below under the same detector. The two lengths are reported separately and never pooled. "Checked clean" means every rewrite in the final span was either judged faithful or minor drift in the reading of READING.md (an AI reading) or accepted by you during the repair. Percentages of tokens, words and characters are of the ORIGINAL span (not kept) or of the FINAL span (new), measured as in each run's REPORT.md.

## 200-token spans (run folder run_20261010_043023_lenient)

- spans loaded 50 | flagged by the detector (z0 > 2) 46 | attacked 46
- lenient attack: **35 broken (76% of attacked)**; by the reading, 24 of those had every rewrite acceptable and 11 had at least one rewrite marked meaning changed
- repair (strict gates, every proposal answered by you): 11 spans needed it, 11 finished (8 broken, 3 not broken), 0 not finished; your answers: yes 10, no 2, new batch 0, skip 2
- **after repair: 32 of 46 attacked spans are broken and checked clean (70%)**; the other 14 are still detected at the paper's default threshold or have an unrepaired flagged rewrite
- attacked spans not detected at the paper's other thresholds: z <= 1.9 31, z <= 1.5 8 (of 46)
- the 32 clean broken spans, mean / median / max: sentences rewritten % of eligible 65.1 / 63.3 / 100.0 | tokens not kept % 19.8 / 18.2 / 43.3 | words not kept % 25.6 / 24.2 / 51.8 | characters not kept % 16.3 / 16.1 / 34.8

Share of spans still detected (z > threshold), paper thresholds, all 50 loaded spans, before and after:

| threshold | before | after |
|---|---|---|
| z > 1.5 | 48 (96%) | 40 (80%) |
| z > 1.9 | 46 (92%) | 15 (30%) |
| z > 2.0 | 46 (92%) | 14 (28%) |
| z > 2.5 | 44 (88%) | 11 (22%) |
| z > 3.0 | 41 (82%) | 8 (16%) |
| z > 3.5 | 36 (72%) | 5 (10%) |
| z > 4.0 | 28 (56%) | 4 (8%) |

Per span (the same rows, with every edit, are in PER_SPAN_FINAL.csv and PER_EDIT_FINAL.csv in the run folder):

| span | z0 | z final | result | checked clean | z<=1.9 | z<=1.5 | rewritten | tokens not kept % | words not kept % | chars not kept % |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 3.394 | 1.895 | broken | yes (reading) | YES | no | 2 of 2 | 11.4 | 14.6 | 9.7 |
| baseline_extra:22 | 4.667 | 1.781 | broken | yes (reading + your approvals) | YES | no | 3 of 6 | 23.9 | 29.4 | 18.4 |
| baseline_extra:23 | 5.233 | 2.783 | not broken | - | no | no | 4 of 4 | 19.4 | 28.2 | 25.8 |
| baseline_extra:24 | 5.091 | 0.457 | broken | yes (reading) | YES | YES | 4 of 4 | 43.3 | 51.8 | 34.8 |
| baseline_extra:25 | 3.818 | 2.597 | not broken | - | no | no | 2 of 2 | 9.0 | 8.7 | 4.9 |
| baseline_extra:26 | 2.828 | 1.126 | broken | yes (reading) | YES | YES | 1 of 4 | 7.0 | 11.3 | 5.8 |
| baseline_extra:27 | 6.223 | 1.741 | broken | yes (reading) | YES | no | 6 of 6 | 32.8 | 43.0 | 28.3 |
| baseline_safe:0 | 3.818 | 2.558 | not broken | - | no | no | 2 of 2 | 13.4 | 26.0 | 16.5 |
| baseline_safe:1 | 6.505 | 3.411 | not broken | - | no | no | 3 of 4 | 16.9 | 19.1 | 12.1 |
| baseline_safe:2 | 5.374 | 3.031 | not broken | - | no | no | 2 of 3 | 18.9 | 25.0 | 15.1 |
| baseline_safe:3 | 3.681 | 1.852 | broken | yes (reading + your approvals) | YES | no | 3 of 3 | 11.7 | 16.4 | 9.8 |
| baseline_safe:4 | 3.536 | 1.924 | broken | yes (reading) | no | no | 2 of 6 | 15.9 | 17.4 | 10.7 |
| samples_len200:28 | 5.374 | 2.231 | not broken | - | no | no | 5 of 5 | 20.4 | 25.9 | 16.5 |
| samples_len200:29 | 4.808 | 1.790 | broken | yes (reading) | YES | no | 2 of 5 | 28.9 | 33.1 | 23.2 |
| samples_len200:30 | 4.101 | 1.706 | broken | yes (reading) | YES | no | 3 of 3 | 15.4 | 20.4 | 12.0 |
| samples_len200:31 | 5.091 | 1.706 | broken | yes (reading) | YES | no | 3 of 3 | 18.4 | 23.7 | 19.4 |
| samples_len200:34 | 3.253 | 1.622 | broken | yes (reading) | YES | no | 3 of 5 | 14.9 | 20.5 | 9.9 |
| samples_len200:35 | 5.940 | 1.347 | broken | yes (reading) | YES | YES | 4 of 8 | 26.4 | 38.1 | 18.9 |
| samples_len200:36 | 5.091 | 2.274 | not broken | - | no | no | 3 of 3 | 18.9 | 20.7 | 12.7 |
| samples_len200:40 | 4.101 | 4.101 | not broken | - | no | no | 0 of 1 | 0.0 | 0.0 | 0.0 |
| samples_len200:42 | 3.677 | 1.451 | broken | yes (reading + your approvals) | YES | YES | 1 of 1 | 19.4 | 23.9 | 15.3 |
| samples_len200:43 | 6.788 | 4.062 | not broken | - | no | no | 3 of 5 | 13.4 | 17.0 | 8.0 |
| samples_len200:44 | 5.091 | 1.085 | broken | yes (reading) | YES | YES | 3 of 6 | 28.9 | 38.4 | 21.0 |
| samples_len200:46 | 2.121 | 1.790 | broken | yes (reading) | YES | no | 1 of 3 | 8.5 | 10.9 | 7.7 |
| samples_len200:47 | 5.515 | 1.161 | broken | yes (reading) | YES | YES | 3 of 4 | 30.3 | 39.6 | 26.3 |
| samples_len200:51 | 3.394 | 2.416 | not broken | - | no | no | 1 of 1 | 7.5 | 8.2 | 5.2 |
| samples_len200:52 | 5.233 | 1.556 | broken | yes (reading) | YES | no | 5 of 10 | 25.9 | 31.2 | 18.7 |
| samples_len200:56 | 3.536 | 1.605 | broken | yes (reading) | YES | no | 2 of 6 | 16.9 | 24.4 | 16.8 |
| samples_len200:60 | 5.515 | 1.578 | broken | yes (reading) | YES | no | 2 of 3 | 25.4 | 29.7 | 23.5 |
| samples_len200:61 | 5.091 | 0.895 | broken | yes (reading + your approvals) | YES | YES | 4 of 5 | 27.4 | 32.0 | 23.4 |
| samples_len200:63 | 6.647 | 3.311 | not broken | - | no | no | 4 of 4 | 15.9 | 26.2 | 15.3 |
| samples_len200:64 | 4.101 | 1.895 | broken | yes (reading + your approvals) | YES | no | 4 of 6 | 17.9 | 19.7 | 11.3 |
| samples_len200:65 | 2.263 | 1.772 | broken | yes (reading) | YES | no | 2 of 2 | 8.5 | 10.4 | 6.2 |
| samples_len200:66 | 5.798 | 1.149 | broken | yes (reading + your approvals) | YES | YES | 4 of 4 | 33.8 | 40.9 | 26.5 |
| samples_len200:69 | 3.111 | 1.781 | broken | yes (reading) | YES | no | 1 of 4 | 10.9 | 12.4 | 8.0 |
| samples_len200:70 | 4.667 | 1.800 | broken | yes (reading) | YES | no | 2 of 2 | 24.9 | 35.6 | 26.8 |
| samples_len200:71 | 4.950 | 4.950 | not broken | - | no | no | 0 of 0 | 0.0 | 0.0 | 0.0 |
| samples_len200:72 | 5.233 | 3.762 | not broken | - | no | no | 3 of 3 | 10.4 | 15.2 | 8.5 |
| samples_len200:73 | 3.960 | 1.781 | broken | yes (reading + your approvals) | YES | no | 3 of 4 | 15.4 | 23.1 | 14.0 |
| samples_len200:74 | 2.970 | 1.571 | broken | yes (reading + your approvals) | YES | no | 2 of 4 | 10.0 | 11.1 | 5.6 |
| samples_len200:78 | 4.950 | 1.714 | broken | yes (reading) | YES | no | 3 of 6 | 19.9 | 31.7 | 19.6 |
| samples_len200:81 | 5.233 | 1.656 | broken | yes (reading) | YES | no | 4 of 6 | 21.9 | 31.7 | 18.0 |
| samples_len200:84 | 2.546 | 1.732 | broken | yes (reading) | YES | no | 1 of 8 | 10.0 | 13.8 | 9.7 |
| samples_len200:85 | 3.960 | 1.614 | broken | yes (reading) | YES | no | 3 of 6 | 17.9 | 24.8 | 15.1 |
| samples_len200:86 | 3.394 | 1.772 | broken | yes (reading) | YES | no | 2 of 5 | 11.4 | 15.0 | 7.2 |
| samples_len200:87 | 4.808 | 4.225 | not broken | - | no | no | 2 of 3 | 9.5 | 12.3 | 7.6 |

## 500-token spans (run folder run_20261010_061701_lenient)

- spans loaded 21 | flagged by the detector (z0 > 2) 21 | attacked 21
- lenient attack: **18 broken (86% of attacked)**; by the reading, 8 of those had every rewrite acceptable and 10 had at least one rewrite marked meaning changed
- repair (strict gates, every proposal answered by you): 10 spans needed it, 10 finished (6 broken, 4 not broken), 0 not finished; your answers: yes 12, no 18, new batch 0, skip 4
- **after repair: 14 of 21 attacked spans are broken and checked clean (67%)**; the other 7 are still detected at the paper's default threshold or have an unrepaired flagged rewrite
- attacked spans not detected at the paper's other thresholds: z <= 1.9 12, z <= 1.5 1 (of 21)
- the 14 clean broken spans, mean / median / max: sentences rewritten % of eligible 63.8 / 66.2 / 100.0 | tokens not kept % 23.8 / 24.0 / 39.3 | words not kept % 31.3 / 30.8 / 49.0 | characters not kept % 18.5 / 17.9 / 31.1

Share of spans still detected (z > threshold), paper thresholds, all 21 loaded spans, before and after:

| threshold | before | after |
|---|---|---|
| z > 1.5 | 21 (100%) | 20 (95%) |
| z > 1.9 | 21 (100%) | 9 (43%) |
| z > 2.0 | 21 (100%) | 7 (33%) |
| z > 2.5 | 21 (100%) | 6 (29%) |
| z > 3.0 | 21 (100%) | 4 (19%) |
| z > 3.5 | 21 (100%) | 3 (14%) |
| z > 4.0 | 21 (100%) | 2 (10%) |

Per span (the same rows, with every edit, are in PER_SPAN_FINAL.csv and PER_EDIT_FINAL.csv in the run folder):

| span | z0 | z final | result | checked clean | z<=1.9 | z<=1.5 | rewritten | tokens not kept % | words not kept % | chars not kept % |
|---|---|---|---|---|---|---|---|---|---|---|
| samples_len500:28 | 7.871 | 3.226 | not broken | - | no | no | 9 of 9 | 17.2 | 23.5 | 14.2 |
| samples_len500:30 | 7.334 | 1.712 | broken | yes (reading + your approvals) | YES | no | 11 of 12 | 20.8 | 26.0 | 16.6 |
| samples_len500:34 | 6.887 | 1.623 | broken | yes (reading) | YES | no | 10 of 16 | 24.6 | 33.7 | 19.5 |
| samples_len500:43 | 7.603 | 3.850 | not broken | - | no | no | 6 of 11 | 28.3 | 29.7 | 19.9 |
| samples_len500:44 | 6.887 | 1.746 | broken | yes (reading) | YES | no | 7 of 16 | 23.4 | 30.6 | 17.4 |
| samples_len500:46 | 6.172 | 2.923 | not broken | - | no | no | 7 of 8 | 16.4 | 20.7 | 13.5 |
| samples_len500:47 | 8.765 | 1.473 | broken | yes (reading + your approvals) | YES | YES | 7 of 10 | 39.3 | 49.0 | 31.1 |
| samples_len500:51 | 6.261 | 1.972 | broken | yes (reading) | no | no | 6 of 6 | 21.0 | 28.5 | 17.8 |
| samples_len500:52 | 7.603 | 1.699 | broken | yes (reading) | YES | no | 14 of 19 | 24.8 | 37.7 | 18.0 |
| samples_len500:56 | 7.692 | 1.720 | broken | yes (reading) | YES | no | 14 of 16 | 31.7 | 41.2 | 24.9 |
| samples_len500:62 | 4.204 | 1.847 | broken | yes (reading + your approvals) | YES | no | 2 of 9 | 9.4 | 13.5 | 8.3 |
| samples_len500:64 | 6.708 | 1.650 | broken | yes (reading) | YES | no | 8 of 16 | 24.8 | 31.0 | 18.9 |
| samples_len500:65 | 6.440 | 5.356 | not broken | - | no | no | 2 of 2 | 4.4 | 5.2 | 3.1 |
| samples_len500:66 | 7.513 | 2.010 | not broken | - | no | no | 8 of 9 | 22.8 | 27.1 | 17.3 |
| samples_len500:69 | 6.529 | 1.875 | broken | yes (reading + your approvals) | YES | no | 6 of 13 | 22.2 | 27.8 | 14.0 |
| samples_len500:71 | 7.513 | 7.155 | not broken | - | no | no | 1 of 1 | 2.4 | 2.9 | 2.4 |
| samples_len500:72 | 7.924 | 2.567 | not broken | - | no | no | 10 of 11 | 24.2 | 35.4 | 21.6 |
| samples_len500:74 | 6.082 | 1.894 | broken | yes (reading + your approvals) | YES | no | 6 of 11 | 17.2 | 20.7 | 12.0 |
| samples_len500:78 | 8.676 | 1.660 | broken | yes (reading) | YES | no | 7 of 15 | 28.7 | 35.8 | 27.1 |
| samples_len500:81 | 7.782 | 1.656 | broken | yes (reading + your approvals) | YES | no | 11 of 15 | 25.3 | 36.9 | 20.3 |
| samples_len500:86 | 6.798 | 1.968 | broken | yes (reading) | no | no | 10 of 14 | 19.6 | 25.4 | 12.9 |

