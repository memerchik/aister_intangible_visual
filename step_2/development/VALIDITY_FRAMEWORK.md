# Validity framework

Model quality in this project is assessed at three related but distinct
layers. Evidence at one layer does not automatically establish the next.

| Layer | Main question | Evidence available in this repository | What that evidence does not establish |
|---|---|---|---|
| Computational validity | Does the implemented procedure perform as recorded under the frozen evaluation protocol? | Content and assignment fingerprints, source-atomic folds, nested out-of-fold predictions, class metrics, diagnostic slices, promotion gates, reproducibility checks, and a sealed test policy | Representative coverage of the wider visual population, cultural adequacy of the labels, or user interpretation of the output |
| Dataset validity | Do the images, labels, sources, and partitions support the intended scope of the claim? | A complete file manifest, duplicate and related-image review, exclusion reasons, source and object grouping, class counts, and explicit provenance and licence fields | Complete provenance, redistribution rights, representative sampling of all relevant objects and sources, or absence of unobserved source bias |
| Human-interpretive validity | Are the categories and model outputs meaningful and defensible in their cultural and historical context? | Operational class definitions, inspectable score rankings, motif-region visualizations, error cohorts, and a human-assisted application design | Expert agreement, historical lineage, cultural exclusivity, user understanding, or a validated account of ambiguous and mixed traditions |

The layers are cumulative. Computational validity is necessary for trusting a
reported experiment, but a reproducible score can still rest on a limited
dataset or an incomplete task definition. Dataset evidence can narrow the
claim to the material actually studied, but interpreting the classes and their
relationships still requires domain expertise and appropriate user research.

## Epistemic status of current claims

The following table separates direct records from interpretations and open
questions.

| Status | Current project statements |
|---|---|
| Observed | The audit found 2,055 supported readable images and retained 2,023. The development set contains 1,693 images and the sealed test contains 330. Five Phase 5 candidates were evaluated and all received a `reject` decision under their frozen gates. V4 recorded the highest nested development accuracy at 94.6840%; v5 recorded the highest source-robustness score at 76.8884% and passed 18 of 21 gates. Opishnyan-to-Bubnivka errors were 45, 45, 50, 43, and 46 in v1–v5. The sealed test has not been scored. |
| Inferred from repository evidence | Source-atomic grouping substantially reduces known leakage through identical files, reviewed objects, semantic sources, and identified acquisition cohorts within this collection. The remaining failures are concentrated around the Opishnyan/Bubnivka boundary rather than being evenly distributed across all five classes. |
| Working hypotheses | Visual similarity, carrier-object cues, source concentration, or limitations of a single flat label may contribute to the persistent ceramic boundary. These are directions for further study, not conclusions established by the current experiments. |
| Not established | Independent test performance; performance on the broader population of Ukrainian ornament images; historical or causal relationships among traditions; cultural completeness or exclusivity of the labels; expert agreement on ambiguous cases; user comprehension of scores and motif regions; image provenance and redistribution rights. |

## Possible future task formulations

The current task assigns one of five flat labels to every input. Future dataset
versions could support richer formulations without changing the evidence from
the completed experiments.

### Hierarchical classification

A hierarchy could first classify a broad family, material, region, or visual
tradition and then make a finer distinction within that branch. It could make
errors at different levels easier to interpret and reduce pressure on a single
five-way boundary. Before implementation, experts would need to define the
hierarchy, decide whether every image has a valid path through it, and specify
metrics for both coarse and fine levels.

### Multilabel classification

Images could carry more than one tradition, motif attribute, material, or
influence label. This would represent overlap explicitly instead of forcing
each ambiguous case into one exclusive category. It requires new image-level
annotations, guidance for uncertainty and disagreement, and multilabel metrics
such as per-label precision and recall rather than the existing confusion
matrix alone.

### Metadata-augmented classification

Where reliable records exist, the model could combine visual evidence with
date, place, maker, object type, material, collection, or provenance metadata.
This may improve interpretation as well as prediction, but the metadata must
be available at the intended point of use. Evaluation would need to keep
shared sources and metadata templates together so that the model cannot appear
to generalize by memorizing collection-specific fields.

These alternatives require a new annotation and evaluation protocol. They are
not post-processing changes to the present five-class results.

