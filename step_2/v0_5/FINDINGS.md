# Assisted application v0.5 — implementation findings

## Outcome

AISTER Ornament Lens v0.5 is implemented as a self-contained, human-assisted
classifier suitable for a local run or a controlled hosted showcase. It provides a polished responsive upload and result
experience backed by the real fixed v4 model recipe, not a mock. It returns an
immediate best visual match, all five uncalibrated ranking scores, six visual
motif regions, a human confirmation/correction flow, and an optional consented
contribution path.

This outcome does **not** promote Phase 5 v4, start formal Phase 6, or evaluate
the sealed test. The application and every API result explicitly preserve:

- `status: provisional_human_assisted`;
- `calibrated: false`;
- `sealed_test_evaluated: false`;
- no unknown-class rejection;
- no production-accuracy claim.

## What was built

### Product and model contract

[`product_contract.json`](product_contract.json) freezes the application scope,
wording, ceramic-boundary policy, privacy behavior, contribution rules, and
deployment boundaries. The fixed recipe is the v4 full-development selection:

- global DINOv3 ViT-S/16 CLS averaged over a letterbox view and five crops;
- the texture-weighted mean of six label-free motif-tile embeddings;
- 768 concatenated features;
- L2 logistic regression with `C=10`;
- source-group exponent `0.5`;
- all 1,693 development records only.

The packaged linear artifact is small (about 29 KB) and lets the server avoid
refitting on startup. The DINOv3 encoder loads only the exact hash-pinned
weight. Locally it uses an existing cache or configured file; a deployment may
download the gated file at startup with a secret Hugging Face token.

### Immediate inference

[`inference.py`](src/aister_runtime/inference.py)
reproduces the exact global and motif representations, applies the packaged
linear model, and returns all five ranking scores. For a known development
image, verification against the canonical caches measured:

| Block | Maximum absolute difference |
|---|---:|
| Global-fivecrop CLS | `1.3411045e-07` |
| Texture-weighted motif mean | `1.9371510e-07` |

Both are below the `3e-7` integration-test tolerance. The five returned scores
sum to `1.0`, but the UI consistently describes them as uncalibrated ranking
scores.

### User experience

The dependency-light local web application includes:

- drag, browse, and clipboard-paste input;
- image type, byte-size, dimension, and decoded-pixel validation;
- a responsive editorial landing page and compact five-tradition field guide;
- a loading state with a concrete local-CPU expectation;
- side-by-side image/result presentation on desktop and stacked mobile layout;
- an inspectable six-region motif overlay with an accessible toggle;
- a first-class Opishnyan/Bubnivka comparison notice;
- all five alternatives with semantic progress bars;
- human confirmation or correction without automatic persistence;
- copyable reviewed-result summary;
- an optional, explicit-consent contribution drawer;
- clear model, evaluation, licence, and cultural-attribution limitations.

The Örnek, Petrykivka, and Kosiv comparison copy links to official UNESCO
Intangible Cultural Heritage descriptions:

- [Örnek](https://ich.unesco.org/en/RL/ornek-a-crimean-tatar-ornament-and-knowledge-about-it-01601?RL=01601)
- [Petrykivka decorative painting](https://ich.unesco.org/en/RL/petrykivka-decorative-painting-as-a-phenomenon-of-the-ukrainian-ornamental-folk-art-00893)
- [Kosiv painted ceramics](https://ich.unesco.org/en/RL/tradition-of-kosiv-painted-ceramics-01456?RL=01456)

### Privacy and future data

Ordinary predictions make no filesystem writes. A bounded 32-item in-memory
cache exists only so a user can choose to contribute the result during the same
server session.

After explicit rights/research consent, the application writes the original
image plus a JSON record to `contributions/pending/`. Each record is marked
`pending_expert_review`, `training_eligible: false`, and
`automatic_training_permitted: false`. The returned contribution identifier
names the local files so deletion is possible.

On the Render showcase, contribution collection is disabled: no prediction
image is cached by the application or written to its ephemeral filesystem.
Future import from a persistent contribution system still requires:

1. rights and consent review;
2. exact and perceptual deduplication;
3. acquisition-source and physical-object grouping;
4. qualified label validation;
5. a newly versioned dataset and source-atomic split.

## Verification results

### Automated checks

- All 231 development tests and all 13 standalone application tests pass after
  relocation (244 checks total).
- The application tests cover model, service, privacy, disabled-contribution,
  static-interface, and real-extractor behavior.
- A normal prediction is verified to write no contribution directory.
- A consented test contribution is verified to remain quarantined and
  ineligible for training.
- Two clean model-package builds produced byte-identical outputs:
  - model artifact SHA-256:
    `0286eede067de4028e744c3ef0d11efc07d488a6670f65d58e07316f0c0fc2c1`;
  - manifest SHA-256:
    `459ee94f9735ab8d29f1639f40aa070cb016969969283f1cdb15c8fd24617190`.

### Live application checks

The actual local server and model were exercised through the browser:

- health endpoint returned ready, provisional, uncalibrated, and
  sealed-test-unevaluated status;
- a real repository image completed upload → prediction in about `0.3s` after
  model warm-up;
- it returned the expected Opishnyan best match and the ceramic comparison
  notice;
- exactly six accessible motif regions and five ranking rows rendered;
- human correction selection updated locally and left contribution disabled;
- motif toggle state and actual SVG visibility were verified after a browser
  compatibility fix;
- desktop `1280×720` and mobile `390×844` layouts had no horizontal overflow;
- the tested result page had one visible H1, no images missing alt text, no
  visible unlabeled controls, and no browser console errors or warnings.

## Remaining limitations

The UI is close to a final product experience, but the model and operational
evidence are not final:

1. V4 failed its frozen promotion contract, mainly at the
   Opishnyan/Bubnivka boundary.
2. The displayed scores are not calibrated and cannot support a safe automatic
   acceptance threshold.
3. The classifier must choose one of five classes; it cannot recognize an
   unsupported or non-ornament image.
4. The 94.1524% figure is a selection-conditional development OOF estimate,
   not sealed or production accuracy.
5. Dataset provenance and image rights are unresolved.
6. DINOv3 has a custom licence that requires deployment review.
7. The dependency-light built-in server is appropriate for this controlled
   showcase, not a hardened high-traffic public service.
8. Expert-review administration, authentication, deletion UI, rate limiting,
   monitoring, and model-version rollout are future production work.
9. Render free-tier CPU, memory, spin-down behavior, and ephemeral storage may
   make it unsuitable for reliable low-latency inference; the prepared deploy
   is a feasibility experiment.

## Recommended next work

Use v0.5 internally and collect only explicitly consented, source-diverse,
expert-reviewable examples. Do not tune a v6 on the current repeatedly inspected
development rows. When materially new data is available, create a new dataset
version, rebuild source-atomic folds and a new sealed partition, freeze gates,
and train a new model version. If that model passes Phase 5, the existing UI,
motif visualization, API shape, privacy default, and contribution quarantine
can be promoted into formal calibration and production engineering rather than
rebuilt.
