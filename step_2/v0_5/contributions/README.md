# Contribution quarantine

The local application writes here only after a user explicitly opts in to
contribute an analyzed image. Each contribution consists of the original image
and a JSON record under `pending/`.

Pending items are **not training data**. They require rights review,
deduplication, source/object grouping, and expert label validation before a
future dataset version may import them. Deleting both files named by a returned
`contribution_id` removes that local contribution.

The prepared Render free-tier profile sets
`AISTER_ENABLE_CONTRIBUTIONS=false`, so the hosted showcase neither exposes the
contribution form nor stores prediction images. Persistent external storage and
the expert-review workflow are required before hosted collection is enabled.
