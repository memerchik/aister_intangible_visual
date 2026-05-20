# Step 1 - Visual heritage type classification

This folder contains the full classification workflow for identifying visual intangible cultural heritage categories from images.

## What is here

- `notebooks/step_1-01_visual_heritage_classifier.ipynb`: full prototype notebook
- `outputs/`: validation benchmark, predictions, metrics, and confusion matrix
- `data/`: labeled development and test image splits

## What the workflow does

1. Reads labeled images from `data/dataset_dev/` and `data/dataset_test/`
2. Extracts a combined descriptor from each image:
   - HSV color histogram
   - HOG shape descriptor
   - uniform LBP texture histogram
3. Benchmarks several candidate classifiers on a development validation split
4. Retrains the best model on the full dev split
5. Evaluates the result on the provided test split
6. Saves reusable outputs into `step_1/outputs/`

## Current selected model

- Model: `knn_cosine_k5`
- Feature set: `all`
- Input resolution: `96x96`
- Validation accuracy: `0.7292`
- Test accuracy: `0.6542`
- Test top-3 accuracy: `0.8957`

## Main outputs

- `outputs/model_selection.csv`
- `outputs/test_predictions.csv`
- `outputs/metrics.json`
- `outputs/confusion_matrix.csv`
- `outputs/confusion_matrix.png`
- `outputs/dataset_summary.csv`

## Run

Open and run the notebook in `notebooks/`.

The notebook contains its own install cell:

```python
%pip install numpy pandas Pillow matplotlib scikit-image scikit-learn
```

Run that cell once if your environment is missing the dependencies. The final export cell overwrites the output artifacts in `outputs/`.
