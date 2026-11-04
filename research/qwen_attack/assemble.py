"""Reads every verified result folder and writes results/FINAL_RESULTS.md -- one authoritative
source of all numbers, with Wilson 95% intervals, error counts, and methodology notes.
Read-only; CPU-only. Paths confirmed to exist on disk."""
import glob, json, math, os

RUNS = [
    ("STA-1 200-tok, adaptive (batch1)",      "results/qwen_attack/run_20261010_043023_lenient", "attack", ""),
    ("STA-1 500-tok, adaptive (batch1)",      "results/qwen_attack/run_20261010_061701_lenient",  "attack", "n=21, small"),
    ("STA-1 500-tok, adaptive (scaled)",      "results/qwen_attack_fullscale/adaptive500",        "attack", "1 span excluded by prompt-leakage guard"),
    ("STA-1 200-tok, no-box (batch2)",        "results/qwen_attack_nobox/run_batch2",             "nobox",  ""),
    ("STA-1 500-tok, no-box",                 "results/qwen_attack_nobox/run500",                 "nobox",  "n=21, small"),
    ("STA-16 200-tok, adaptive",              "results/qwen_attack_fullscale/sta16_attack",       "attack", "robust variant"),
    ("STA-16 200-tok, no-box",                "results/qwen_attack_fullscale/sta16_nobox",        "nobox",  "robust variant"),
    ("STA-16 500-tok, adaptive",              "results/qwen_attack_sta16_500/attack",             "attack", "robust variant, n=53"),
    ("STA-16 500-tok, no-box",                "results/qwen_attack_sta16_500/nobox",              "nobox",  "robust variant, n=53"),
]

def wilson(k, n):
    if n == 0: return (0.0, 0.0, 0.0)
    p=k/n; z=1.96; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*p, 100*(c-h), 100*(c+h)

def rate(folder, key):
    b=t=e=0
    for f in glob.glob(folder+"/samples/*.json"):
        s=json.load(open(f))
        if key not in s or "status" not in s.get(key,{}):
            e+=1; continue
        t+=1; b += s[key]["status"]=="broken"
    return b, t, e

def main():
    L = ["# FINAL RESULTS -- STA watermark adaptive attack\n",
         "All numbers read directly from result folders. 'broken' = span z <= 2.0 under the same detector. "
         "Intervals are Wilson 95%. Each attack sees only the generated span, never the prompt (enforced by an assertion guard).\n",
         "## Attack success\n",
         "| setting | broken | n | 95% interval | errors | note |",
         "|---|---|---|---|---|---|"]
    for label, folder, key, note in RUNS:
        b, t, e = rate(folder, key)
        p, lo, hi = wilson(b, t)
        L.append("| %s | %d | %d | %.0f%% [%.0f%%, %.0f%%] | %d | %s |" % (label, b, t, p, lo, hi, e, note))
    L += ["", "## Headline: detector guidance vs blind paraphrase (same spans, same gates)\n",
          "| watermark / length | adaptive | no-box | n(adaptive/nobox) |", "|---|---|---|---|"]
    def pct(folder,key):
        b,t,e=rate(folder,key); p,lo,hi=wilson(b,t); return "%.0f%% [%.0f-%.0f]"%(p,lo,hi), t
    a200,na=pct("results/qwen_attack/run_20261010_043023_lenient","attack")
    nb200,nn=pct("results/qwen_attack_nobox/run_batch2","nobox")
    a500,a5=pct("results/qwen_attack_fullscale/adaptive500","attack")
    nb500,n5=pct("results/qwen_attack_nobox/run500","nobox")
    s16a,sa=pct("results/qwen_attack_fullscale/sta16_attack","attack")
    s16n,sn=pct("results/qwen_attack_fullscale/sta16_nobox","nobox")
    L.append("| STA-1 200-tok | %s | %s | %d/%d |" % (a200, nb200, na, nn))
    L.append("| STA-1 500-tok | %s | %s | %d/%d |" % (a500, nb500, a5, n5))
    L.append("| STA-16 (robust) 200-tok | %s | %s | %d/%d |" % (s16a, s16n, sa, sn))
    s16a5,sa5=pct("results/qwen_attack_sta16_500/attack","attack")
    s16n5,sn5=pct("results/qwen_attack_sta16_500/nobox","nobox")
    L.append("| STA-16 (robust) 500-tok | %s | %s | %d/%d |" % (s16a5, s16n5, sa5, sn5))
    L += ["", "## Supporting analyses (full detail in the named files)\n",
          "- Bound (predicts breakability): results/qwen_attack/BOUND_CHECK.md",
          "- Query-budget defence: results/qwen_attack/QUERY_BUDGET.md",
          "- Copy-paste baseline (paper-faithful): results/qwen_attack/COPY_PASTE_FAITHFUL.md",
          "- Ablations (seed/n_cands/preset/target): results/qwen_attack_ablation/SUMMARY.md",
          "- ROC/AUC/TPR vs controls: results/qwen_attack/ROC_EVAL.md",
          "",
          "## Methodology notes\n",
          "- 1 span (samples_len500:351) excluded from scaled 500-tok adaptive: the generated text echoed its prompt at char 1020, tripping the prompt-leakage guard. 134 clean spans reported.",
          "- Quality of rewrites judged by AI reading + author repair; independent human rating NOT yet done (stated limitation).",
          "- 500-tok no-box and STA-1 500-tok batch1 are n=21 (small); report with their wide intervals.",
          ""]
    os.makedirs("results", exist_ok=True)
    open("results/FINAL_RESULTS.md","w").write("\n".join(L)+"\n")
    print("\n".join(L))
    print("written: results/FINAL_RESULTS.md")

if __name__ == "__main__":
    main()
