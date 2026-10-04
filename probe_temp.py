"""
Find the temperature where Qwen produces enough distinct
candidates without losing fidelity.

At temperature 1.2 Qwen returned 3-5 distinct rewrites from 60
requested, where Llama returned 30-35. The sweep then evaded 0
of 9, against Llama's 6 of 9, purely because best-of-3 is not
selection.

Qwen converges on the correct phrasing, which is why it keeps
facts and why it lacks variety. Higher temperature flattens the
distribution so it explores more wordings.

Prints the distinct count and marks whether "overestimated"
survives, since reversing that was the failure that made us
leave Llama.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research" / "blackbox"))

from paraphrase import Paraphraser

TEMPS = [1.5, 1.8, 2.1, 2.4]

SENT = ("Ultimately, her number overestimated how long she "
        "could go without a change in her antibiotics.")


def main():
    para = Paraphraser()

    for t in TEMPS:
        enc = para.tok(para._prompt(SENT),
                       return_tensors="pt").to(para.model.device)
        plen = enc["input_ids"].shape[1]

        seen = []
        for _ in range(10):
            with torch.no_grad():
                gen = para.model.generate(
                    **enc, do_sample=True, temperature=t,
                    top_p=0.98, max_new_tokens=100,
                    num_return_sequences=4,
                    pad_token_id=para.tok.eos_token_id,
                )
            for row in gen:
                line = para.tok.decode(
                    row[plen:], skip_special_tokens=True
                ).strip().split("\n")[0].strip()
                if len(line) > 10 and line not in seen:
                    seen.append(line)
            del gen
            torch.cuda.empty_cache()

        kept = sum(1 for s in seen
                   if "overestimat" in s.lower()
                   or "overstat" in s.lower())

        print("=" * 66)
        print("temperature %.1f -> %d distinct of 40, %d keep direction"
              % (t, len(seen), kept))
        print("=" * 66)
        for s in seen[:12]:
            ok = ("overestimat" in s.lower()
                  or "overstat" in s.lower())
            print("  %s %s" % ("KEEP " if ok else "DRIFT", s[:110]))
        print()


if __name__ == "__main__":
    main()
