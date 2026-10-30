# ROC evaluation: AUC and TPR at fixed false-positive rates

Detected means z > threshold. Thresholds come from negatives that were not attacked (the attack only runs on flagged spans). Intervals are 95% bootstrap percentiles (1000 resamples of both the spans and the negatives). The positives are a small selection (spans where the model kept generating), so the intervals are wide; a 1% false-positive level needs at least 100 negatives to be resolved at all. Model controls are unwatermarked generations from the same model, sampling and prompts rule; human text is the start of C4 documents.

## 200-token spans (50 spans loaded, never pooled with the other length)

### negatives: model controls (unwatermarked) (n = 200, mean 0.37, std 1.19, max 3.82)

false-positive rate of the paper's fixed thresholds on these negatives: z > 1.5: 20.0%, z > 1.9: 9.0%, z > 2.0: 7.5%

| measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| AUC (95% interval) | 0.972 [0.943, 0.993] | 0.854 [0.807, 0.899] | 0.861 [0.814, 0.905] |
| TPR at FPR 1% (threshold z > 3.39, actual FPR 0.5%) | 72% [56%, 92%] | 10% [2%, 24%] | 12% [4%, 30%] |
| TPR at FPR 5% (threshold z > 2.26, actual FPR 3.5%) | 88% [78%, 98%] | 20% [10%, 34%] | 26% [14%, 40%] |
| TPR at FPR 10% (threshold z > 1.84, actual FPR 9.0%) | 92% [84%, 98%] | 32% [16%, 68%] | 36% [20%, 68%] |
The paper's own definitions on the same negatives (evaluation_metrics/detection.py): its "auc" is computed on thresholded labels, i.e. (TPR + TNR) / 2; its F1 is shown class-balanced (equal weight for spans and negatives); its TPR at a target FPR is read at the ROC point whose FPR is closest to the target.

| paper-style measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| z > 1.5 (FPR 20.0% on these negatives) | TPR 96%, auc 0.880, F1 0.889 | TPR 80%, auc 0.800, F1 0.800 | TPR 80%, auc 0.800, F1 0.800 |
| z > 1.9 (FPR 9.0% on these negatives) | TPR 92%, auc 0.915, F1 0.915 | TPR 30%, auc 0.605, F1 0.432 | TPR 30%, auc 0.605, F1 0.432 |
| z > 2.0 (FPR 7.5% on these negatives) | TPR 92%, auc 0.922, F1 0.922 | TPR 22%, auc 0.573, F1 0.340 | TPR 28%, auc 0.603, F1 0.413 |
| TPR at target FPR 1% (closest ROC point) | 78% (FPR 1.5%) | 10% (FPR 1.5%) | 12% (FPR 1.5%) |
| TPR at target FPR 5% (closest ROC point) | 88% (FPR 3.5%) | 20% (FPR 5.5%) | 26% (FPR 5.5%) |
| TPR at target FPR 10% (closest ROC point) | 92% (FPR 9.0%) | 26% (FPR 9.0%) | 28% (FPR 9.0%) |

### negatives: human text (C4) (n = 798, mean 0.23, std 1.10, max 3.54)

false-positive rate of the paper's fixed thresholds on these negatives: z > 1.5: 13.3%, z > 1.9: 6.9%, z > 2.0: 5.0%

| measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| AUC (95% interval) | 0.978 [0.956, 0.996] | 0.895 [0.865, 0.923] | 0.901 [0.870, 0.927] |
| TPR at FPR 1% (threshold z > 2.97, actual FPR 0.9%) | 82% [68%, 92%] | 12% [4%, 22%] | 16% [6%, 26%] |
| TPR at FPR 5% (threshold z > 2.12, actual FPR 4.0%) | 90% [82%, 98%] | 22% [12%, 36%] | 28% [16%, 40%] |
| TPR at FPR 10% (threshold z > 1.70, actual FPR 8.8%) | 94% [86%, 100%] | 62% [26%, 86%] | 64% [32%, 86%] |
The paper's own definitions on the same negatives (evaluation_metrics/detection.py): its "auc" is computed on thresholded labels, i.e. (TPR + TNR) / 2; its F1 is shown class-balanced (equal weight for spans and negatives); its TPR at a target FPR is read at the ROC point whose FPR is closest to the target.

| paper-style measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| z > 1.5 (FPR 13.3% on these negatives) | TPR 96%, auc 0.914, F1 0.917 | TPR 80%, auc 0.834, F1 0.828 | TPR 80%, auc 0.834, F1 0.828 |
| z > 1.9 (FPR 6.9% on these negatives) | TPR 92%, auc 0.926, F1 0.925 | TPR 30%, auc 0.616, F1 0.438 | TPR 30%, auc 0.616, F1 0.438 |
| z > 2.0 (FPR 5.0% on these negatives) | TPR 92%, auc 0.935, F1 0.934 | TPR 22%, auc 0.585, F1 0.346 | TPR 28%, auc 0.615, F1 0.421 |
| TPR at target FPR 1% (closest ROC point) | 84% (FPR 1.0%) | 12% (FPR 1.0%) | 16% (FPR 1.0%) |
| TPR at target FPR 5% (closest ROC point) | 92% (FPR 5.0%) | 22% (FPR 5.0%) | 28% (FPR 5.0%) |
| TPR at target FPR 10% (closest ROC point) | 94% (FPR 10.2%) | 62% (FPR 10.2%) | 64% (FPR 10.2%) |

## 500-token spans (21 spans loaded, never pooled with the other length)

### negatives: model controls (unwatermarked) (n = 120, mean 0.52, std 1.27, max 4.47)

false-positive rate of the paper's fixed thresholds on these negatives: z > 1.5: 23.3%, z > 1.9: 12.5%, z > 2.0: 9.2%

| measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| AUC (95% interval) | 1.000 [0.998, 1.000] | 0.870 [0.807, 0.921] | 0.890 [0.831, 0.939] |
| TPR at FPR 1% (threshold z > 4.02, actual FPR 0.8%) | 100% [90%, 100%] | 10% [0%, 24%] | 10% [0%, 33%] |
| TPR at FPR 5% (threshold z > 2.19, actual FPR 5.0%) | 100% [100%, 100%] | 14% [0%, 33%] | 29% [10%, 52%] |
| TPR at FPR 10% (threshold z > 1.97, actual FPR 9.2%) | 100% [100%, 100%] | 24% [5%, 81%] | 43% [14%, 81%] |
The paper's own definitions on the same negatives (evaluation_metrics/detection.py): its "auc" is computed on thresholded labels, i.e. (TPR + TNR) / 2; its F1 is shown class-balanced (equal weight for spans and negatives); its TPR at a target FPR is read at the ROC point whose FPR is closest to the target.

| paper-style measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| z > 1.5 (FPR 23.3% on these negatives) | TPR 100%, auc 0.883, F1 0.896 | TPR 86%, auc 0.812, F1 0.820 | TPR 95%, auc 0.860, F1 0.871 |
| z > 1.9 (FPR 12.5% on these negatives) | TPR 100%, auc 0.938, F1 0.941 | TPR 43%, auc 0.652, F1 0.552 | TPR 43%, auc 0.652, F1 0.552 |
| z > 2.0 (FPR 9.2% on these negatives) | TPR 100%, auc 0.954, F1 0.956 | TPR 14%, auc 0.526, F1 0.231 | TPR 33%, auc 0.621, F1 0.468 |
| TPR at target FPR 1% (closest ROC point) | 95% (FPR 0.8%) | 0% (FPR 0.0%) | 10% (FPR 1.7%) |
| TPR at target FPR 5% (closest ROC point) | 100% (FPR 5.0%) | 14% (FPR 5.0%) | 24% (FPR 5.0%) |
| TPR at target FPR 10% (closest ROC point) | 100% (FPR 9.2%) | 14% (FPR 9.2%) | 29% (FPR 9.2%) |

### negatives: human text (C4) (n = 455, mean 0.40, std 1.20, max 5.10)

false-positive rate of the paper's fixed thresholds on these negatives: z > 1.5: 17.8%, z > 1.9: 10.3%, z > 2.0: 9.7%

| measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| AUC (95% interval) | 1.000 [0.999, 1.000] | 0.889 [0.854, 0.921] | 0.905 [0.872, 0.935] |
| TPR at FPR 1% (threshold z > 3.40, actual FPR 0.7%) | 100% [95%, 100%] | 10% [0%, 24%] | 14% [0%, 33%] |
| TPR at FPR 5% (threshold z > 2.50, actual FPR 4.2%) | 100% [100%, 100%] | 14% [0%, 29%] | 29% [10%, 48%] |
| TPR at FPR 10% (threshold z > 1.97, actual FPR 9.7%) | 100% [100%, 100%] | 24% [5%, 67%] | 43% [14%, 71%] |
The paper's own definitions on the same negatives (evaluation_metrics/detection.py): its "auc" is computed on thresholded labels, i.e. (TPR + TNR) / 2; its F1 is shown class-balanced (equal weight for spans and negatives); its TPR at a target FPR is read at the ROC point whose FPR is closest to the target.

| paper-style measure | before attack | after lenient attack | after repair |
|---|---|---|---|
| z > 1.5 (FPR 17.8% on these negatives) | TPR 100%, auc 0.911, F1 0.918 | TPR 86%, auc 0.840, F1 0.842 | TPR 95%, auc 0.887, F1 0.894 |
| z > 1.9 (FPR 10.3% on these negatives) | TPR 100%, auc 0.948, F1 0.951 | TPR 43%, auc 0.663, F1 0.560 | TPR 43%, auc 0.663, F1 0.560 |
| z > 2.0 (FPR 9.7% on these negatives) | TPR 100%, auc 0.952, F1 0.954 | TPR 14%, auc 0.523, F1 0.230 | TPR 33%, auc 0.618, F1 0.466 |
| TPR at target FPR 1% (closest ROC point) | 100% (FPR 1.1%) | 10% (FPR 1.1%) | 14% (FPR 1.1%) |
| TPR at target FPR 5% (closest ROC point) | 100% (FPR 5.1%) | 14% (FPR 5.1%) | 29% (FPR 5.1%) |
| TPR at target FPR 10% (closest ROC point) | 100% (FPR 10.3%) | 29% (FPR 10.3%) | 43% (FPR 10.3%) |

