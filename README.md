# PRODIGY_ML_03

Implement a support vector machine (SVM) to classify images of cats and dogs from the [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset.

The pipeline resizes each image to 96×96, extracts grayscale HOG features plus an HSV color histogram, scales the vectors, and trains a balanced linear SVM (`LinearSVC`). Training prints metrics, writes `reports/`, and saves the model.

If the official competition download returns 403, the public [Microsoft Cats vs Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) set is used (25,000 labeled images).

## Results

Full dataset (no `--limit`), 96×96, `C=1`:

| Metric | Value |
| --- | --- |
| Validation accuracy | **69.54%** |
| Train / test images | 19,997 / 5,000 |
| Cat precision / recall | 0.70 / 0.69 |
| Dog precision / recall | 0.69 / 0.70 |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

If the dataset is already on disk (project `data/` or the kagglehub cache), train without downloading:

```powershell
python cats_dogs_svm.py --model-out .\models\cats_dogs_svm.joblib
```

A quicker run (2,000 images per class):

```powershell
python cats_dogs_svm.py --limit 2000 --model-out .\models\cats_dogs_svm.joblib
```

The script prints `Using local dataset: ...`. It does not contact Kaggle unless you pass `--download`.

### First-time download (optional)

```powershell
python cats_dogs_svm.py --download --limit 2000 --model-out .\models\cats_dogs_svm.joblib
```

Accept the competition rules at [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/rules) if you want the official files. A 403 falls back to the public Microsoft dataset.

## Predict

```powershell
python cats_dogs_svm.py --model-out .\models\cats_dogs_svm.joblib --predict .\path\to\image.jpg
```

## Report

Training writes `reports/`:

- `summary.md` — accuracy and setup
- `metrics.json` — numeric results
- `classification_report.txt` — precision, recall, F1
- `confusion_matrix.png`
- `sample_predictions.png` — correct and incorrect examples

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | auto-detect | Folder of labeled images |
| `--download` | off | Fetch from Kaggle |
| `--limit` | all images | Max images per class |
| `--image-size` | 96 | Square resize (multiple of 8, ≥ 16) |
| `--c` | 1.0 | SVM regularization |
| `--model-out` | `cats_dogs_svm.joblib` | Saved model |
| `--report-dir` | `reports` | Metrics and plots |
| `--predict` | none | Classify one image (skips training) |

Images can be named `cat.0.jpg` / `dog.0.jpg`, or placed in `cat`/`cats` and `dog`/`dogs` folders.

## Improving accuracy

This is a linear SVM on HOG, so scores typically stay in the high 60s to low 70s. To improve:

1. Train on all images (omit `--limit`).
2. Use `--image-size 128`.
3. Try `--c 0.1` or `--c 10`.
