"""
Check Module 2 on the real spans.

Run:  python research/qwen_attack/check_sentences.py

Loads every sample file that exists (Module 1), cuts each span into
segments (Module 2) and prints one line per span, then the totals, the
skip reasons, a reassembly check, and examples of every kind of skipped
segment so the rules can be judged by eye. Uses no model and no GPU.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["results/raw/baseline_safe.json", "results/raw/baseline_extra.json",
         "results/raw/samples_len200.json", "results/raw/samples_len500.json"]


def report(samples_list):
    import sentences
    rows, by_file, reasons, n_seg, examples, identical = [], {}, {}, 0, {}, 0
    for s in samples_list:
        seg = sentences.split_span(s.span)
        identical += seg.reassemble() == s.span
        sm = seg.summary()
        n_seg += sm["segments"]
        tag = Path(s.source).stem
        by_file.setdefault(tag, []).append(sm)
        for k, v in sm["skip_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
        for g in seg.segments:
            if g.skip:
                examples.setdefault(g.skip, []).append("%s: %s" % (s.sample_id, g.text[:80].replace("\n", " ")))
        rows.append("%-24s %6d %5d %9d %8d %9.0f%%  %-7s %-8s"
                    % (s.sample_id, sm["chars"], sm["segments"], sm["eligible"], sm["skipped"],
                       100 * sm["eligible_char_share"],
                       "leading" if sm["leading_fragment"] else "-",
                       "trailing" if sm["trailing_fragment"] else "-"))
    out = ["%-24s %6s %5s %9s %8s %10s  %-7s %-8s"
           % ("sample", "chars", "segs", "eligible", "skipped", "eligible%", "start", "end")] + rows + [""]
    out.append("%-18s %5s %14s %18s %10s %10s"
               % ("file", "spans", "mean eligible", "eligible chars %", "leading", "trailing"))
    for tag, sms in by_file.items():
        out.append("%-18s %5d %14.1f %11.0f%% (min %2.0f%%) %7d %10d"
                   % (tag, len(sms), sum(x["eligible"] for x in sms) / len(sms),
                      100 * sum(x["eligible_char_share"] for x in sms) / len(sms),
                      100 * min(x["eligible_char_share"] for x in sms),
                      sum(1 for x in sms if x["leading_fragment"]),
                      sum(1 for x in sms if x["trailing_fragment"])))
    out.append("")
    out.append("skipped segments by reason (of %d segments in %d spans):" % (n_seg, len(samples_list)))
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        out.append("   %-34s %5d  (%.1f%%)" % (k, v, 100 * v / n_seg))
    out.append("reassembly with no replacements identical to the span: %d of %d spans"
               % (identical, len(samples_list)))
    out.append("settings: %s" % sentences.settings())
    for k, lst in examples.items():
        out.append("")
        out.append("examples, %s (%d in total):" % (k, len(lst)))
        for e in lst[:6]:
            out.append("   " + e)
    return "\n".join(out)


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE)]
    import samples
    files = [p for p in FILES if os.path.exists(p)]
    print(report(samples.load_samples(files)))


if __name__ == "__main__":
    main()
