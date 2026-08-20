# AISTER Ornament Lens v0.5

This folder contains the standalone human-assisted application. It can be
copied, run, and deployed without `step_2/development/`.

The application accepts one JPEG, PNG, or WebP image and returns:

- an immediate best visual match among five traditions;
- all five uncalibrated ranking scores;
- six model-selected motif regions;
- a non-blocking Opishnyan/Bubnivka comparison warning;
- optional human confirmation or correction;
- optional contribution collection when the deployment explicitly enables it.

V0.5 is a preview of the intended user experience, not a validated production
classifier. It has no calibrated confidence, unknown-class rejection, or
sealed-test result. See [`FINDINGS.md`](FINDINGS.md) for the supporting evidence
and known limitations.

## Folder map

```text
v0_5/
├── Dockerfile                    # Render/container build
├── requirements.txt             # container runtime dependencies
├── requirements-local.txt       # local runtime, including PyTorch
├── server.py                     # HTTP service, upload handling, and API
├── product_contract.json         # frozen product claims and restrictions
├── artifacts/                    # small packaged linear classifier
├── src/aister_runtime/           # image preprocessing and inference
├── static/                       # editable HTML, CSS, JavaScript, and logo
├── tests/                        # app-only service/model tests
└── contributions/pending/        # ignored quarantine; local opt-in only
```

## Run locally

From the repository root:

```bash
python3 -m pip install -r step_2/v0_5/requirements-local.txt
python3 step_2/v0_5/server.py --open
```

The app opens at [http://127.0.0.1:8787/](http://127.0.0.1:8787/). Stop it
with `Ctrl+C`.

The runtime first looks for the exact gated DINOv3 weight in the normal local
Hugging Face cache. You may instead point to an already downloaded file:

```bash
AISTER_DINOV3_WEIGHT=/absolute/path/model.safetensors \
python3 step_2/v0_5/server.py
```

The expected SHA-256 is pinned in the runtime. A missing or mismatched weight
fails startup rather than silently changing the model.

## Test the standalone app

```bash
PYTHONPYCACHEPREFIX=.cache/pycache \
python3 -m unittest discover -s step_2/v0_5/tests -v
```

The integration checks run when the pinned DINOv3 weight is locally available.
The development tree also has a separate reproduction test proving that this
standalone runtime matches the frozen feature pipeline.

## Deploy on Render free tier

The repository-root [`render.yaml`](../../render.yaml) is ready for a Render
Blueprint. It declares a Docker web service with:

- repository root directory `step_2/v0_5`;
- health endpoint `/api/health`;
- Render-provided `PORT` binding through `HOST=0.0.0.0`;
- one CPU inference thread to reduce memory pressure;
- one-view inference micro-batches and automatic large-photo downscaling to
  stay within the free instance memory envelope;
- one isolated prediction worker at a time, with overlapping requests rejected
  immediately instead of queued in memory;
- contribution collection disabled;
- the gated weight downloaded to ephemeral `/tmp` storage after hash checking.

Deployment steps:

1. Accept the terms and request access to
   `facebook/dinov3-vits16-pretrain-lvd1689m` on Hugging Face.
2. Create a read-only Hugging Face token that can download that gated model.
3. Push this repository and connect it to Render.
4. In Render, choose **New → Blueprint** and select the repository. Render
   reads `render.yaml` from the repository root.
5. When prompted for `HF_TOKEN`, enter it as a secret. Do not commit it.
6. Create the free service, wait for the Docker build and initial weight/model
   warm-up, then confirm `/api/health` reports `"status": "ready"`.

Free Render instances have tight CPU and memory limits, spin down after an idle
period, and use an ephemeral filesystem. The first request after a spin-down is
therefore not immediate, and the 86 MB encoder must be reacquired after a fresh
instance. The runtime streams checkpoint tensors into the encoder, processes
the 12 frozen v4 views one at a time, serializes predictions, and caps decoded
uploads at two megapixels. On Render, each accepted prediction runs in a
short-lived worker process, so PyTorch/native allocations are returned to the
operating system when the result is ready. The service never queues uploaded
images: a concurrent request receives HTTP `429` with `Retry-After: 3` before
its body is read. These controls reduce RAM without changing the model, crops,
or scoring recipe. They add several seconds of per-request model-loading
latency. This setup is suitable for a low-traffic demonstration. If it still
exceeds the free memory or request-time limits, it will need a larger Render
instance or a dedicated model-hosting service.

Render health requests every few seconds are expected platform probes. The
server now writes `Prediction accepted`, `Prediction completed`, and
`Prediction rejected` messages immediately, because the built-in HTTP access
log records a request only after a response begins.

The public Render configuration disables contributions because free-instance
files are not durable. Prediction images are neither written nor retained by
the app in this mode. Add persistent object storage and the expert-review
workflow before enabling collection on any hosted deployment.

## Edit the frontend

No frontend build step is required:

- edit [`static/index.html`](static/index.html) for content and structure;
- edit [`static/app.css`](static/app.css) for layout, colors, typography, and
  responsive behavior;
- edit [`static/app.js`](static/app.js) for upload, prediction, visualization,
  correction, and contribution interactions;
- replace [`static/aister-logo.png`](static/aister-logo.png) for branding.

Save the file and refresh the browser. Keep API routes under `/api/` and retain
the disclosure that scores are uncalibrated. The Content Security Policy allows
only local scripts and styles, so add new JavaScript and CSS as local files and
register new served assets in `STATIC_FILES` inside [`server.py`](server.py).

## Contribution behavior

Local contributions are enabled by default. Ordinary predictions stay in a
bounded memory cache only. If a user gives explicit rights/research consent and
submits a selected label, the app writes an image and matching JSON record to
`contributions/pending/`. Every record is marked:

- `review_status: pending_expert_review`;
- `training_eligible: false`;
- `automatic_training_permitted: false`.

Set `AISTER_ENABLE_CONTRIBUTIONS=false` to remove that UI and prevent image
caching or writes. Before importing any contribution into a future dataset,
perform rights review, exact/perceptual deduplication, acquisition-source and
physical-object grouping, and expert label validation.

## Rebuild the packaged classifier

The model-building script remains in the research workspace because it depends
on development-only splits and embedding caches:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache \
python3 step_2/development/scripts/build_assisted_v0_5_model.py
```

It writes the standalone artifact and manifest into this folder. The recipe is
the fixed Phase 5 v4 full-development selection: DINOv3 global-fivecrop CLS,
the texture-weighted mean of six label-free motif tiles, a 768-dimensional
input, five-class L2 logistic regression with `C=10`, source-group exponent
`0.5`, and all 1,693 development records. It never accesses the sealed test.

## Release status

The current build is ready for a controlled demonstration. A general public v1
release still requires independent model evaluation, calibrated confidence or
a clear alternative decision policy, unknown-image handling, durable storage
and expert review for contributions, and review of dataset-image rights and the
custom DINOv3 licence.
