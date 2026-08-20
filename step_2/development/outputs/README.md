# Experiment output index

This directory contains the saved development results for Phases 3–5. All
reported predictions are for the 1,693-image development set. None of these
folders contains predictions for the 330-image sealed test set.

## Result summary

| Experiment | Accuracy | Macro F1 | Main record | Predictions | Confusion matrix |
|---|---:|---:|---|---|---|
| Phase 3 classical baseline | 63.3196% | 59.2031% | [`metrics.json`](phase_3_classical_baseline/metrics.json) | [`oof_predictions.csv`](phase_3_classical_baseline/oof_predictions.csv) | [`CSV`](phase_3_classical_baseline/confusion_matrix.csv) · [`PNG`](phase_3_classical_baseline/confusion_matrix.png) |
| Phase 4 pretrained encoder | 93.0892% | 92.8447% | [`metrics.json`](phase_4_pretrained/metrics.json) | [`oof_predictions.csv`](phase_4_pretrained/oof_predictions.csv) | [`CSV`](phase_4_pretrained/confusion_matrix.csv) · [`PNG`](phase_4_pretrained/confusion_matrix.png) |
| Phase 5 v1 source-balanced selection | 93.9752% | 93.6135% | [`metrics.json`](phase_5_source_robustness/metrics.json) | [`nested OOF`](phase_5_source_robustness/nested_oof_predictions.csv) · [`selected OOF`](phase_5_source_robustness/selected_oof_predictions.csv) | [`CSV`](phase_5_source_robustness/confusion_matrix.csv) · [`PNG`](phase_5_source_robustness/confusion_matrix.png) |
| Phase 5 v2 pairwise correction | 93.7980% | 93.4211% | [`metrics.json`](phase_5_source_robustness_v2/metrics.json) | [`nested OOF`](phase_5_source_robustness_v2/nested_oof_predictions.csv) · [`selected OOF`](phase_5_source_robustness_v2/selected_oof_predictions.csv) | [`CSV`](phase_5_source_robustness_v2/confusion_matrix.csv) · [`PNG`](phase_5_source_robustness_v2/confusion_matrix.png) |
| Phase 5 v3 encoder consensus | 93.9161% | 93.5815% | [`metrics.json`](phase_5_source_robustness_v3/metrics.json) | [`nested OOF`](phase_5_source_robustness_v3/nested_oof_predictions.csv) · [`selected OOF`](phase_5_source_robustness_v3/selected_oof_predictions.csv) | [`CSV`](phase_5_source_robustness_v3/confusion_matrix.csv) · [`PNG`](phase_5_source_robustness_v3/confusion_matrix.png) |
| Phase 5 v4 motif localization | **94.6840%** | **94.3622%** | [`metrics.json`](phase_5_source_robustness_v4/metrics.json) | [`nested OOF`](phase_5_source_robustness_v4/nested_oof_predictions.csv) · [`selected OOF`](phase_5_source_robustness_v4/selected_oof_predictions.csv) | [`CSV`](phase_5_source_robustness_v4/confusion_matrix.csv) · [`PNG`](phase_5_source_robustness_v4/confusion_matrix.png) |
| Phase 5 v5 fixed motif consensus | 94.5068% | 94.1957% | [`metrics.json`](phase_5_source_robustness_v5/metrics.json) | [`nested OOF`](phase_5_source_robustness_v5/nested_oof_predictions.csv) · [`selected OOF`](phase_5_source_robustness_v5/selected_oof_predictions.csv) | [`CSV`](phase_5_source_robustness_v5/confusion_matrix.csv) · [`PNG`](phase_5_source_robustness_v5/confusion_matrix.png) |

The Phase 5 accuracy and macro-F1 values above are the selection-aware nested
results. `nested_oof_predictions.csv` contains the predictions made after
candidate selection was repeated inside each outer training side. The
`selected_oof_predictions.csv` files describe the recipe selected using the
full development set and should not replace the nested estimate.

The Phase 5 progression and promotion decisions are summarized in
[`../PHASE_5_EVOLUTION.md`](../PHASE_5_EVOLUTION.md). Exact gate thresholds and
outcomes are in [`../PHASE_5_GATES.md`](../PHASE_5_GATES.md).

## Reading the confusion matrices

In every `confusion_matrix.csv`, rows are true classes and columns are
predicted classes. The first column is named `true_class`; the remaining five
column names are the predicted labels. Diagonal cells are correct predictions.
An off-diagonal cell in the Opishnyan row and Bubnivka column therefore counts
an Opishnyan → Bubnivka error. The PNG versions use the same orientation: true
label on the vertical axis and predicted label on the horizontal axis.

Each experiment directory also contains `confusion_pairs.csv`, which lists
directed off-diagonal errors by frequency, and `per_class_metrics.csv`, which
records precision, recall, F1, and support for each class.

## Directory contents

- [`phase_3_classical_baseline/`](phase_3_classical_baseline/) compares the
  handcrafted HSV, HOG, and LBP baseline candidates.
- [`phase_4_pretrained/`](phase_4_pretrained/) is the canonical comparison of
  13 frozen pretrained representation and view combinations.
- [`phase_4_pretrained_quick/`](phase_4_pretrained_quick/) is a reduced
  diagnostic run. It is not the canonical Phase 4 result.
- [`phase_5_source_robustness/`](phase_5_source_robustness/) is Phase 5 v1.
- [`phase_5_source_robustness_v2/`](phase_5_source_robustness_v2/) is the
  pairwise ceramic-correction experiment.
- [`phase_5_source_robustness_v3/`](phase_5_source_robustness_v3/) is the
  cross-encoder consensus experiment.
- [`phase_5_source_robustness_v4/`](phase_5_source_robustness_v4/) is the
  label-free motif-localization experiment.
- [`phase_5_source_robustness_v5/`](phase_5_source_robustness_v5/) is the fixed
  motif-consensus experiment.

Aggregate metrics should always be read with the fold, source-cohort, and
class-level diagnostics. The experiments used the same development records
for method development and selection, so they are not an independent estimate
of performance on new sources.
