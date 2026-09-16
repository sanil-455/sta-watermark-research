# Attack trials

Six attempts on STA-1. Each one drove the detection score
below the threshold. Each one also produced text a reader
would notice something wrong with.

Every folder here holds what was tried, the sentence that
exposed the flaw, why it happened, and what the next attempt
changed in response.

| Trial | Problem | Example |
|---|---|---|
| [01](01_subword_fragments/) | Deleted pieces of words | "he grew up in Carr" |
| [02](02_wrong_sense/) | Right word, wrong meaning | "vehicle mart opportunity" |
| [03](03_morphology/) | Singular and plural mixed up | "These are also matter" |
| [04](04_over_filtering/) | Fix was too aggressive | pool fell 159 to 29 |
| [05](05_contextual_fit/) | Synonym that does not fit | "denoted its purpose" |
| [06](06_collocation/) | Broken fixed phrase | "doing some good progress" |

The seventh attempt worked. It is in
[`../attacks/README.md`](../attacks/README.md).

Read them in order. Each one only makes sense given the one
before it.
