# Hands-on Workshop Materials

This folder contains a simplified notebook for recreating the classical image-classification prototype on a participant-chosen image topic.

Participants should **download and organize images before running the notebook**. The notebook is intentionally data-first: it reads whatever labeled folders participants prepare.

## What Participants Will Build

Participants will:

- choose at least two visual categories,
- download example images,
- organize images into matching class folders,
- extract classical visual features,
- train a lightweight classifier,
- evaluate predictions for every query/test image,
- inspect the model confidence for every class.

The workshop uses classical computer vision rather than AI image embeddings:

- HSV color histograms for palette information
- HOG descriptors for shape and contour layout
- LBP histograms for local texture patterns
- KNN classification with cosine distance

## Correct Data Storage

Put training/development images into class folders. The repository starts with two placeholder classes:

```text
data/dataset_dev/
  01_class/
    image_001.jpg
    image_002.jpg
  02_class/
    image_001.jpg
```

Put query/test images into matching class folders:

```text
data/dataset_test/
  01_class/
    test_001.jpg
  02_class/
    test_001.jpg
```

The folder name is the class label. You may rename `01_class` and `02_class` to real labels, for example `01_ceramics` and `02_textiles`, but the names must match exactly between `dataset_dev/` and `dataset_test/`.

Good:

```text
dataset_dev/01_ceramics/
dataset_test/01_ceramics/
```

Problem:

```text
dataset_dev/01_ceramics/
dataset_test/ceramics/
```

If `dataset_test/` is empty, the notebook automatically creates a small validation split from `dataset_dev/`.

## Suggested Image Sources

Use open-access image sources so participants can download examples themselves:

- The Fruits collection: https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images

Suggested easy category pairs:

- ceramics vs textiles
- paintings vs sculptures
- coins vs manuscripts
- musical instruments vs furniture
- birds vs insects

For a short workshop, aim for 8-15 training images per class in `dataset_dev/` and 2-5 query/test images per class in `dataset_test/`.

## Main Notebook

Run:

```text
notebooks/classical_vision_hands_on.ipynb
```

Run it only after images have been added. It does not require the private dataset in `step_1/data/`.

The prediction table shows one row per query/test image and includes:

- the true folder label,
- the predicted label,
- top-1/top-2/top-3 predictions where available,
- one confidence column for every class.

## Outputs

Generated files are written to `outputs/`:

- `workshop_dataset_summary.csv`
- `workshop_predictions.csv` with one row per query/test image and confidence columns for every class
- `workshop_metrics.json`
- `workshop_confusion_matrix.csv`
- `workshop_confusion_matrix.png`
