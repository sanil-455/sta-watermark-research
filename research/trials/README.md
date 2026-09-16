# Attack trials

Seven attempts on STA-1. Every one of them drove the detection
score below the threshold. The first six also produced text a
reader would notice something wrong with.

Each folder holds what was tried, the sentence that exposed
the flaw, why it happened, and what the next attempt changed
in response.

| Trial | Problem | Example |
|---|---|---|
| [01](01_subword_fragments/) | Deleted pieces of words | "he grew up in Carr" |
| [02](02_wrong_sense/) | Right word, wrong meaning | "vehicle mart opportunity" |
| [03](03_morphology/) | Singular and plural mixed up | "These are also matter" |
| [04](04_over_filtering/) | The fix cut too deep | pool fell from 159 to 29 |
| [05](05_contextual_fit/) | Synonym that does not fit | "denoted its purpose" |
| [06](06_collocation/) | Broken fixed phrase | "doing some good progress" |
| [07](07_sample3_blacklist/) | Filters were not enough | "he's went from pornography" |

Read them in order. Each only makes sense given the one
before it.

Trials 1 to 6 worked on Sample 2, a financial news article of
788 words. Trial 7 applied the finished pipeline to Sample 3,
a book review of 574 words, and found that the automated
checks still missed three errors a person caught immediately.

Both documents were broken in the end, with no grammatical
errors in either. The final results are in
[`../attacks/README.md`](../attacks/README.md).
