# Phase 3 findings - source-blocked classical baseline

Phase 3 reran the frozen Step 1 classical feature pipeline on the source-atomic v3 development folds. It evaluated the same HSV, HOG, and LBP features and five predeclared classifier candidates. It did not read or evaluate the sealed test set.

## Result

`knn_cosine_k5` remains the winner under the predeclared rule of mean five-fold accuracy.

- Aggregate OOF accuracy: **63.3196%**
- Aggregate macro F1: **59.2031%**
- Aggregate balanced accuracy: **59.1724%**
- Aggregate top-3 accuracy: **87.9504%**
- Unweighted fold accuracy: **61.4701% ± 10.8879 percentage points**
- Fixed validation accuracy: **59.5679%**

The model makes 1,072 correct and 621 incorrect predictions among 1,693 development images. Reaching at least 90% on the same records would require 452 additional correct predictions, removing about 73% of the current errors.

The weighted aggregate accuracy and unweighted fold mean differ because the source-blocked folds contain 500/244/270/342/337 images. Their individual accuracies are 78.2000%, 61.4754%, 55.5556%, 63.1579%, and 48.9614%. This large variation is part of the honest source-shift result, not random row-level CV noise.

## Candidate comparison

The candidate ranking by unweighted mean fold accuracy is:

- `knn_cosine_k5`: 61.4701% ± 10.8879 points; mean macro F1 54.3660%.
- `knn_cosine_k7`: 61.1912% ± 11.5122 points.
- Logistic regression: 61.0988% ± 9.2703 points; mean macro F1 55.7665% and top-3 91.4587%.
- `knn_cosine_k3`: 60.4566% ± 11.3885 points.
- Color-only linear SVM: 59.0343% ± 6.6063 points.

The candidates differ by only 2.44 points from first to last. Neighbour-count or simple classifier tuning cannot plausibly close the performance gap.

## Per-class findings

- Opishnyan ceramics: 79.6407% recall.
- Petrykivka painting: 72.6923% recall.
- Ornek: 66.2870% recall.
- Kosiv ceramics: 42.5000% recall.
- Bubnivka ceramics: 34.7418% recall.

The largest directed confusions are Kosiv → Ornek (76), Bubnivka → Ornek (68), Ornek → Petrykivka (57), Ornek → Kosiv (52), and Opishnyan → Ornek (51). Handcrafted color and local texture descriptors do not consistently separate motifs when object form, painting style, and source conditions change together.

## Diagnostic slices

- Images originating from the old development folder score 63.8816%; those from the old test folder score 62.1324%. The old folder label is provenance only.
- The 348 manually reviewed development records score 79.5977%; 1,345 singleton records score 59.1078%. The cohorts differ in class, object type, and source structure, so this is not evidence that manual review or multi-view status causes higher accuracy.
- Mixed displays/collections score 24.1379% and plates/bowls 63.1579% within the small reviewed object-type slice. Cups, vessels, and containers are much easier. These class-correlated slices are diagnostic leads, not population estimates.

Source-cohort performance is highly variable. For example, the large Opishnyan roundmark cohort is almost perfectly recognized while several Bubnivka and Kosiv acquisition cohorts are difficult. This is consistent with the fold spread and reinforces why source-atomic evaluation is required.

## Historical correction and conclusion

The earlier 70.35% v1 result is retained only as a historical diagnostic. V1 was invalidated after missed semantic source/object links were found, and v2 was subsequently invalidated by the global acquisition-source audit. The v3 result—63.3196% aggregate OOF accuracy—is the valid classical reference for subsequent development comparisons.

The 87.9504% top-3 result shows that the classical representation often retains class signal, but ranks overlapping motifs poorly. The Phase 4 pretrained representation screen is the appropriate next comparison; it uses the same v3 records and source-blocked folds. No conclusion about unseen performance should be drawn until a frozen pipeline is evaluated once on the still-sealed v3 test set.
