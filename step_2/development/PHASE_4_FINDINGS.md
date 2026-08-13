# Phase 4 findings - frozen pretrained representations

Phase 4 screened frozen pretrained image encoders and view policies on the source-atomic v3 development folds. Each representation feeds a class-balanced logistic probe with fixed `C=10`. The runner validates the development manifest, source/group isolation, every image hash, encoder weights, preprocessing, and cache identity before fitting.

The sealed test manifest was not read for embeddings or predictions, and the 330-image v3 test set remains untouched.

## Selected development result

The selection rule chooses `dinov3_vits16__cls__global_fivecrop`: the DINOv3 ViT-S/16 CLS embedding averaged across a letterboxed global view plus five 224-pixel crops.

- Exploratory aggregate OOF accuracy: **93.0892%**
- Exploratory aggregate macro F1: **92.8447%**
- Exploratory aggregate balanced accuracy: **94.0378%**
- Exploratory aggregate top-3 accuracy: **99.5275%**
- Unweighted fold accuracy: **93.0985% ± 5.9735 percentage points**
- Worst aggregate class recall: **83.4331%**

The selected representation makes 1,576 correct and 117 incorrect predictions among 1,693 development images. Fold accuracies are 97.4000%, 97.1311%, 97.7778%, 87.4269%, and 85.7567% for folds of 500/244/270/342/337 images.

## Essential interpretation

These are **selection-conditional exploratory metrics, not an unbiased performance estimate**. The same OOF results were used to compare and select representations, and the candidate set and `C=10` setting were informed by earlier prototype exploration. The 93.0892% figure therefore shows that the project has a strong development candidate; it does not establish 93% performance on unseen sources.

The aggregate `predict_proba` values are uncalibrated ranking scores. They must not be presented as trustworthy confidence, and no acceptance/rejection threshold may be chosen from this run. No calibration study was performed.

## Representation comparison

The most relevant aggregate and fold results are:

| Representation | Aggregate accuracy | Macro F1 | Balanced accuracy | Top-3 | Fold accuracy mean ± SD |
|---|---:|---:|---:|---:|---:|
| DINOv3 CLS, global + five crops (selected) | 93.0892% | 92.8447% | 94.0378% | 99.5275% | 93.0985% ± 5.9735 |
| DINOv3 CLS, center crop | 92.4395% | 92.0740% | 93.2379% | 99.5275% | 92.4495% ± 6.4247 |
| DINOv2 small reg4, center crop | 90.5493% | 90.3602% | 91.6557% | 99.2321% | 90.4781% ± 7.6850 |
| ConvNeXt Tiny, center crop | 90.0768% | 90.0136% | 91.3233% | 99.5275% | 89.8397% ± 6.9627 |

DINOv2 and ConvNeXt are credible numerical fallbacks if DINOv3 cannot be used, but their own upstream model terms still require review before deployment. The DINOv3 multi-view gain over its center crop is only 0.65 aggregate accuracy points, so its six-view inference cost should be revisited after unbiased evaluation rather than assumed worthwhile.

Representation rankings are materially source-sensitive. On the secondary fixed train-to-validation diagnostic, the selected representation scores 89.81% and ranks sixth, while `dinov3_vits16__cls_patch_concat__letterbox` scores 95.06% and ranks first despite ranking fifth by OOF accuracy. The fixed validation set contains the entire 104-image hard gray-catalog cohort, which explains part of the reversal but also shows why no representation should be chosen from one source-heavy partition. All 13 diagnostic `C` curves peak at the grid ceiling of 10, so probe tuning is not settled either.

Against the frozen v3 classical reference, the selected Phase 4 candidate improves aggregate OOF accuracy by 29.77 points and macro F1 by 33.64 points. This is strong evidence that pretrained semantic features are the right modelling direction, while still being an exploratory comparison.

## Per-class findings

- Ornek recall: 98.6333%.
- Kosiv recall: 97.8571%.
- Petrykivka recall: 97.3077%.
- Bubnivka recall: 92.9577%.
- Opishnyan recall: 83.4331%.

The classical Kosiv/Ornek bottleneck is largely resolved. The dominant remaining error is now Opishnyan → Bubnivka: 58 errors, or 11.58% of development Opishnyan examples. This single direction accounts for almost half of the 117 total errors. The remaining directed confusions are small; the next largest are Opishnyan → Kosiv (17) and Bubnivka → Opishnyan (11).

The hardest source-blocked folds are folds 3 and 4. In the reviewed object-type slice, mixed displays/collections remain difficult at 51.7241%, while cups/mugs and vessels exceed 94%. These slices are non-random and class/source-correlated, so they identify review targets rather than general subgroup performance.

The three hardest Opishnyan source cohorts—shop cubbies, shelf figurines, and gray catalog B—produce 65 of all 117 errors. Named source cohorts as a whole score 88.38%, compared with 96.25% for directly unassigned images. This concentration is actionable evidence that source and scene robustness, not broad five-class separation, is now the main modelling bottleneck.

Top-2 accuracy is 98.11% and top-3 accuracy is 99.53%, leaving 32 and 8 development misses respectively. That makes a human-review workflow promising, but it cannot safely use the current score magnitudes: 23 wrong predictions have an uncalibrated maximum score of at least 0.8 and 5 exceed 0.9.

## Reproducibility and model provenance

The canonical run records:

- split version `step02_source_atomic_v3` and assignment fingerprint `59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`;
- exact development CSV and image-byte fingerprints;
- encoder revision, weight and configuration hashes;
- view policy, embedding dimensions, cache identity, runtime versions, and deterministic CPU settings;
- OOF membership and all group-isolation checks.

The known-working Phase 4 packages are pinned separately in `requirements-phase4.txt`. The full run evaluates 13 representations and writes the artifacts described in `outputs/phase_4_pretrained/README.md`.

## Deployment blockers and phase boundary

The selected `facebook/dinov3-vits16-pretrain-lvd1689m` checkpoint is gated and uses the custom DINOv3 licence. Legal review is required before it can be the deployment dependency. The dataset images also retain unresolved provenance and licence from Phase 1.

Phase 4 produces development diagnostics, not a deployable artifact. It does not serialize a final fitted probe, calibrate probabilities, define rejection thresholds, expose an inference API, or build the workshop-style prediction visualization into a user interface.

The valid next evaluation is one exactly specified, frozen pipeline run on the sealed v3 test. That evaluation has not started. Until it occurs, the project has exceeded the 90% target only on a selection-conditional development diagnostic, not on an unbiased benchmark.
