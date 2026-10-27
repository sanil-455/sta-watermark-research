# Automatic audit of the rewrites

**This is an automatic judgement, not a human reading.** Each rewrite was judged in both directions by `FacebookAI/roberta-large-mnli` (355,362,819 parameters) and by rule checks. Thresholds: faithful if both entailment probabilities >= 0.80; minor drift if both >= 0.40; meaning changed below that, if either contradiction probability >= 0.50, or if a number, quotation, negation count or capitalised name changed; garbled if a new unknown word or glued punctuation appears.

**Not yet calibrated against human verdicts** (run `audit.py sample` and `audit.py score`).

broken spans audited: 35 | clean: 25 | not clean: 10 | rewrites: 99 (faithful 80, minor drift 9, meaning changed 10, garbled 0)

reasons behind meaning changed / garbled: {'name changed': 6, 'entailment': 5, 'contradiction': 1}

| span | z0 -> z | rewrites | verdicts | result |
|---|---|---|---|---|
| baseline_extra:20 | 3.39 -> 1.90 | 2 | faithful, faithful | **clean** |
| baseline_extra:22 | 4.67 -> 1.35 | 4 | minor drift, faithful, faithful, faithful | **clean** |
| baseline_extra:24 | 5.09 -> 0.46 | 4 | faithful, meaning changed, faithful, faithful | **not clean** |
| baseline_extra:26 | 2.83 -> 1.13 | 1 | faithful | **clean** |
| baseline_extra:27 | 6.22 -> 1.74 | 6 | faithful, faithful, meaning changed, faithful, faithful, faithful | **not clean** |
| baseline_safe:2 | 5.37 -> 1.56 | 3 | faithful, faithful, faithful | **clean** |
| baseline_safe:3 | 3.68 -> 1.72 | 3 | minor drift, faithful, faithful | **clean** |
| baseline_safe:4 | 3.54 -> 1.92 | 2 | faithful, faithful | **clean** |
| samples_len200:29 | 4.81 -> 1.79 | 2 | faithful, meaning changed | **not clean** |
| samples_len200:30 | 4.10 -> 1.71 | 3 | faithful, faithful, faithful | **clean** |
| samples_len200:31 | 5.09 -> 1.71 | 3 | meaning changed, faithful, faithful | **not clean** |
| samples_len200:34 | 3.25 -> 1.62 | 3 | faithful, meaning changed, faithful | **not clean** |
| samples_len200:35 | 5.94 -> 1.35 | 4 | faithful, faithful, faithful, faithful | **clean** |
| samples_len200:36 | 5.09 -> 1.78 | 3 | faithful, meaning changed, faithful | **not clean** |
| samples_len200:42 | 3.68 -> 1.95 | 1 | faithful | **clean** |
| samples_len200:43 | 6.79 -> 2.00 | 5 | faithful, minor drift, minor drift, faithful, faithful | **clean** |
| samples_len200:44 | 5.09 -> 1.09 | 3 | faithful, faithful, faithful | **clean** |
| samples_len200:46 | 2.12 -> 1.79 | 1 | minor drift | **clean** |
| samples_len200:47 | 5.52 -> 1.16 | 3 | faithful, minor drift, faithful | **clean** |
| samples_len200:52 | 5.23 -> 1.56 | 5 | faithful, meaning changed, faithful, faithful, faithful | **not clean** |
| samples_len200:56 | 3.54 -> 1.60 | 2 | meaning changed, faithful | **not clean** |
| samples_len200:60 | 5.52 -> 1.58 | 2 | faithful, meaning changed | **not clean** |
| samples_len200:61 | 5.09 -> 0.98 | 4 | meaning changed, faithful, faithful, faithful | **not clean** |
| samples_len200:64 | 4.10 -> 1.80 | 4 | faithful, faithful, faithful, minor drift | **clean** |
| samples_len200:65 | 2.26 -> 1.77 | 2 | faithful, faithful | **clean** |
| samples_len200:66 | 5.80 -> 0.99 | 4 | faithful, faithful, faithful, faithful | **clean** |
| samples_len200:69 | 3.11 -> 1.78 | 1 | faithful | **clean** |
| samples_len200:70 | 4.67 -> 1.80 | 2 | faithful, minor drift | **clean** |
| samples_len200:73 | 3.96 -> 1.98 | 2 | faithful, faithful | **clean** |
| samples_len200:74 | 2.97 -> 1.64 | 2 | faithful, faithful | **clean** |
| samples_len200:78 | 4.95 -> 1.71 | 3 | faithful, faithful, faithful | **clean** |
| samples_len200:81 | 5.23 -> 1.66 | 4 | faithful, faithful, faithful, faithful | **clean** |
| samples_len200:84 | 2.55 -> 1.73 | 1 | faithful | **clean** |
| samples_len200:85 | 3.96 -> 1.61 | 3 | faithful, minor drift, faithful | **clean** |
| samples_len200:86 | 3.39 -> 1.77 | 2 | faithful, faithful | **clean** |

## Every rewrite that was not judged faithful

**baseline_extra:22 [1] MINOR DRIFT** (no rule fired; entailment 0.68 / 0.52)

- original: It had to do with the way she felt about the new boyfriend and the sense of freedom her responsibility-free job gave her.
- rewrite: It pertained to how she felt about her new boyfriend and the sense of freedom from responsibilities her job lacked.

**baseline_extra:24 [3] MEANING CHANGED** (entailment 0.13; entailment 0.99 / 0.13)

- original: It is a rather lengthy 52 pages, but the document basically outlines the first steps manufacturers need to take in order to have their autonomous vehicles considered for licensing.
- rewrite: The document, spanning 52 pages, primarily details the initial steps manufacturers must follow to gain licensing for their autonomous vehicles.

**baseline_extra:27 [2] MEANING CHANGED** (name changed; entailment 0.99 / 0.99)

- original: Whether you're looking for Texan Tex-Mex or American barbecue, you can find it along the Riverwalk.
- rewrite: Whatever your preference for Tex-Mex from Texas or barbecue from America, you'll find it along the Riverwalk.

**baseline_safe:3 [1] MINOR DRIFT** (no rule fired; entailment 0.99 / 0.66)

- original: Sustainability policies should also give room for communities who opt for different energy and climate policies over and above state mandates.
- rewrite: Community choices for various energy and climate policies deserve consideration alongside existing sustainability mandates.

**samples_len200:29 [5] MEANING CHANGED** (contradiction 0.63; entailment 0.30; entailment 0.99 / 0.30)

- original: In an interview, Followay paid tribute to the hard work being done in helping youth fail fast and learn how to be successful after their disabilities, during their journey through both school and career.
- rewrite: In an interview, Followay honored the significant efforts dedicated to assisting youth in quickly identifying challenges and acquiring success skills throughout their education and professional life despite disabilities.

**samples_len200:31 [1] MEANING CHANGED** (entailment 0.13; entailment 0.99 / 0.13)

- original: Johnson, who identifies as pro-choice to the audience, made the controversial transformation to pro-life activist after she witnessed the procedure of an abortion.
- rewrite: Johnson, who portrays herself as pro-choice publicly, shifted to becoming a pro-life activist after witnessing an abortion procedure.

**samples_len200:34 [4] MEANING CHANGED** (name changed; entailment 0.98 / 0.94)

- original: The Next Step in Mental Health: Supports, Case Management, Community Services, Treatment.
- rewrite: Advancing Mental Health Support: Services, Case Management, Community Programs, Treatments.

**samples_len200:36 [2] MEANING CHANGED** (name changed; entailment 0.99 / 0.94)

- original: It also supports most common operating systems, including Windows, Linux, and Mac OS X. The driver enables a single-user-to-several-user installation according to the needs of the workgroup in large networks.
- rewrite: It supports a range of operating systems like Windows, Linux, and Mac OS X, and the driver facilitates a single license for multiple user installations based on the group size in extensive network environments.

**samples_len200:43 [3] MINOR DRIFT** (no rule fired; entailment 0.95 / 0.57)

- original: And because it involved a madressah in Lahore which is location A, tribes are already forming on location B and according to you’re the mind whfleingly, tribal tensions in Yemen.
- rewrite: And since it involved a madressah in Lahore (location A), tribal tensions are already rising in location B, and as you know, tribal rivalries are intensifying in Yemen.

**samples_len200:43 [4] MINOR DRIFT** (no rule fired; entailment 0.99 / 0.78)

- original: Bad news always gets buried, good news is spoken about by all, no matter how trivial, no point picking on it.
- rewrite: Good news is often highlighted, while bad news remains unnoticed, no matter how significant, not worth focusing on the latter.

**samples_len200:46 [3] MINOR DRIFT** (no rule fired; entailment 0.99 / 0.66)

- original: It came on the heels of an offseason during which the head coach and offensive coordinator reportedly made comments that were “belittling,” Jimmyond Mathews, a former Browns second-round draft pick, said publicly on social media.
- rewrite: It followed an offseason where the head coach and offensive coordinator reportedly made comments perceived as “belittling,” according to publicly posted remarks by Jimmyond Mathews, a former Browns second-round draft pick.

**samples_len200:47 [3] MINOR DRIFT** (no rule fired; entailment 0.98 / 0.49)

- original: Developed jointly by Skoda’s design team and the Czech engineers at Volkswagen’s RCD division, its soft-touch interior follows the form and looks similar to its sister products the Audi A1, A4 and A6.
- rewrite: Created collaboratively by Skoda’s design team and Volkswagen’s RCD division in Czech, its soft-touch interior mirrors the design of sister Audi models, including the A1, A4, and A6.

**samples_len200:52 [7] MEANING CHANGED** (name changed; entailment 0.99 / 0.99)

- original: Any American, irrespective of which location or region he or she is from, benefits.
- rewrite: Anyone in America, regardless of location or region, derives benefit.

**samples_len200:56 [3] MEANING CHANGED** (entailment 0.31; entailment 0.99 / 0.31)

- original: Water is not only natural, safe and healthy but drinking a glass of water can curb your appetite because water fills you up sooner than any other food source in your diet.
- rewrite: Water is not only natural, safe, and healthy, but also helps curb your appetite by filling you up more quickly than most foods in your diet.

**samples_len200:60 [4] MEANING CHANGED** (name changed; entailment 0.99 / 0.94)

- original: Photo-sharing app Instagram to go pro with new tool To prevent copyright infringement, Instagram is bolstering its service with a new analytics tool for companies to track their brands on the hot photo-sharing vehicle.
- rewrite: Instagram is enhancing its platform for businesses to combat copyright infringement by introducing new analytics tools that permit companies to monitor their brands across the popular photo-sharing site.

**samples_len200:61 [1] MEANING CHANGED** (name changed; entailment 0.02; entailment 0.97 / 0.02)

- original: It sounded as ominous as the Impact's Sigi Schmid anyone-but-Toronto soccer chant.
- rewrite: It seemed just as ominous as any chants against Sigi Schmid by the Impact fans.

**samples_len200:64 [8] MINOR DRIFT** (no rule fired; entailment 0.72 / 0.98)

- original: Working on Wall Street, I learned about humor from the very top.
- rewrite: At Wall Street, I discovered that even the highest levels imbue humor into everyday operations.

**samples_len200:70 [2] MINOR DRIFT** (no rule fired; entailment 0.92 / 0.43)

- original: Clichy was in the wars again in the 76th minute as his challenge left the visiting keeper looking at the sky and Yevhen Konoplyanka was sent clean through on goal.
- rewrite: Clichy caused chaos for the visiting keeper with a challenge in the 76th minute, leaving the keeper looking skyward, and creating a clear goal-scoring opportunity for Yevhen Konoplyanka.

**samples_len200:85 [2] MINOR DRIFT** (no rule fired; entailment 0.99 / 0.66)

- original: The ETF is holding up well this year as consumer staples stocks have rallied stronger than the broad market and emerging markets, where the China-US tariff dispute began.
- rewrite: This year, the ETF is performing well as consumer staples stocks have gained more than the broader market and emerging markets, where the trade disagreement between China and the US started.

