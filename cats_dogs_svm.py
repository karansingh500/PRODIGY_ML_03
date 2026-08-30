"""Train a HOG + color-histogram linear SVM for Kaggle Dogs vs. Cats images."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import cv2
import joblib
import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kagglehub.exceptions import KaggleApiHTTPError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


IMAGE_PATTERN = re.compile(r"^(cat|dog)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FOLDER_LABELS = {
    "cat": 0,
    "cats": 0,
    "dog": 1,
    "dogs": 1,
}
CLASS_NAMES = ("cat", "dog")
HIST_BINS = (8, 8, 8)
COMPETITION_HANDLE = "dogs-vs-cats"
PUBLIC_DATASET_HANDLE = "shaunthesheep/microsoft-catsvsdogs-dataset"
COMPETITION_RULES_URL = "https://www.kaggle.com/competitions/dogs-vs-cats/rules"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset directory. If omitted, uses ./data or a previously cached Kaggle copy.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dogs-vs-cats from Kaggle with kagglehub.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("cats_dogs_svm.joblib"),
        help="Path for the trained model (default: cats_dogs_svm.joblib).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for metrics and plots (default: reports).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum images per class after shuffling. Omit to use every labeled image.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="LinearSVC regularization (C).")
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Classify one image using an existing model instead of training.",
    )
    return parser.parse_args()


def _usable_samples(data_dir: Path) -> list[tuple[Path, int]]:
    samples = find_labeled_images(data_dir)
    return [
        sample
        for sample in samples
        if not any(part.startswith("_smoke") for part in sample[0].parts)
    ]


def find_local_dataset() -> Path | None:
    """Reuse ./data or a kagglehub cache so training does not need a download."""
    from kagglehub.config import get_cache_folder

    cache = Path(get_cache_folder())
    candidates = [Path("data")]
    dataset_root = cache / "datasets" / "shaunthesheep" / "microsoft-catsvsdogs-dataset"
    versions = dataset_root / "versions"
    if versions.is_dir():
        candidates.extend(sorted(versions.iterdir(), reverse=True))
    candidates.append(dataset_root)
    candidates.append(cache / "competitions" / "dogs-vs-cats")

    best_path: Path | None = None
    best_count = 0
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            samples = _usable_samples(candidate)
        except FileNotFoundError:
            continue
        labels = {label for _, label in samples}
        if len(labels) < 2 or len(samples) <= best_count:
            continue
        best_path = candidate
        best_count = len(samples)
    return best_path


def download_dataset() -> Path:
    try:
        path = Path(kagglehub.competition_download(COMPETITION_HANDLE))
        print("Path to competition files:", path)
    except KaggleApiHTTPError as error:
        status = getattr(error.response, "status_code", None)
        if status != 403:
            raise
        print(
            "Competition download returned 403. Accept the rules in a browser at:\n"
            f"  {COMPETITION_RULES_URL}\n"
            "The api.kaggle.com/.../rules link in the error is not a usable page.\n"
            f"Downloading public dataset {PUBLIC_DATASET_HANDLE} instead..."
        )
        path = Path(kagglehub.dataset_download(PUBLIC_DATASET_HANDLE))
        print("Path to dataset files:", path)
    _extract_train_archives(path)
    return path


def _extract_train_archives(root: Path) -> None:
    for archive_path in sorted(root.rglob("train.zip")):
        target = archive_path.parent
        marker = target / "train"
        if marker.is_dir() and any(marker.rglob("*.jpg")):
            continue
        print(f"Extracting {archive_path}...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)


def _label_from_path(path: Path) -> int | None:
    match = IMAGE_PATTERN.match(path.name)
    if match:
        return int(match.group(1).lower() == "dog")
    for part in path.parts:
        label = FOLDER_LABELS.get(part.lower())
        if label is not None:
            return label
    return None


def find_labeled_images(data_dir: Path) -> list[tuple[Path, int]]:
    """Find cat/dog files under data_dir, including nested Kaggle train/ folders."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    samples: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        label = _label_from_path(path)
        if label is None:
            continue
        seen.add(resolved)
        samples.append((path, label))
    return samples


def extract_features(image_path: Path, image_size: int) -> np.ndarray:
    if image_size < 16 or image_size % 8 != 0:
        raise ValueError("--image-size must be a multiple of 8 and at least 16")
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Could not read image: {image_path}")
    color = cv2.resize(color, (image_size, image_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.HOGDescriptor(
        (image_size, image_size),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )
    hog = descriptor.compute(gray).ravel().astype(np.float32)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    histograms: list[np.ndarray] = []
    ranges = ((0, 180), (0, 256), (0, 256))
    for channel, bins, channel_range in zip(range(3), HIST_BINS, ranges, strict=True):
        hist = cv2.calcHist([hsv], [channel], None, [bins], channel_range)
        hist = cv2.normalize(hist, hist).ravel()
        histograms.append(hist.astype(np.float32))
    return np.concatenate([hog, *histograms])


def select_samples(
    samples: list[tuple[Path, int]],
    limit: int | None,
    random_state: int,
) -> list[tuple[Path, int]]:
    rng = np.random.default_rng(random_state)
    selected: list[tuple[Path, int]] = []
    for label in (0, 1):
        class_samples = [sample for sample in samples if sample[1] == label]
        order = rng.permutation(len(class_samples))
        shuffled = [class_samples[i] for i in order]
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be at least 1")
            shuffled = shuffled[:limit]
        selected.extend(shuffled)
    return selected


def build_dataset(
    samples: list[tuple[Path, int]],
    image_size: int,
    limit: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    selected = select_samples(samples, limit, random_state)
    if not selected:
        raise FileNotFoundError(
            "No labeled images found. Expected names such as cat.0.jpg and dog.0.jpg "
            "inside --data-dir or --data-dir/train, or folders named cat/cats and dog/dogs."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    for index, (path, label) in enumerate(selected, start=1):
        try:
            features.append(extract_features(path, image_size))
        except ValueError as error:
            print(f"Skipping {error}")
            continue
        labels.append(label)
        paths.append(path)
        if index % 500 == 0 or index == len(selected):
            print(f"Extracted features for {index}/{len(selected)} images")

    if len(set(labels)) < 2:
        raise ValueError(
            "Training requires at least one readable cat image and one readable dog image."
        )
    return np.vstack(features), np.asarray(labels), paths


def _read_rgb(path: Path, size: int = 128) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_report(
    report_dir: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: list[Path],
    accuracy: float,
    image_size: int,
    limit: int | None,
    n_train: int,
    c_value: float,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    report_text = classification_report(
        y_true, y_pred, target_names=list(CLASS_NAMES), zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=list(CLASS_NAMES), zero_division=0, output_dict=True
    )
    payload = {
        "accuracy": accuracy,
        "n_train": n_train,
        "n_test": int(len(y_true)),
        "image_size": image_size,
        "limit_per_class": limit,
        "c": c_value,
        "features": "HOG + HSV color histogram",
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
    }
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=list(CLASS_NAMES))
    axis.set_yticks([0, 1], labels=list(CLASS_NAMES))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Cats vs dogs SVM confusion matrix")
    for row in range(2):
        for col in range(2):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=140)
    plt.close(figure)

    correct = [i for i, (true, pred) in enumerate(zip(y_true, y_pred, strict=True)) if true == pred]
    wrong = [i for i, (true, pred) in enumerate(zip(y_true, y_pred, strict=True)) if true != pred]
    slots: list[tuple[str, int]] = [("correct", i) for i in correct[:4]]
    slots.extend(("wrong", i) for i in wrong[:4])
    if not slots:
        return

    cols = min(4, len(slots))
    rows = int(np.ceil(len(slots) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for axis in axes_flat:
        axis.axis("off")
    for axis, (kind, index) in zip(axes_flat, slots, strict=False):
        rgb = _read_rgb(paths[index])
        if rgb is None:
            continue
        true_name = CLASS_NAMES[int(y_true[index])]
        pred_name = CLASS_NAMES[int(y_pred[index])]
        axis.imshow(rgb)
        axis.set_title(f"{kind}: true {true_name}\npred {pred_name}", fontsize=9)
        axis.axis("off")
    figure.suptitle("Validation predictions", fontsize=12)
    figure.tight_layout()
    figure.savefig(report_dir / "sample_predictions.png", dpi=140)
    plt.close(figure)

    summary = (
        f"# Cats vs dogs SVM report\n\n"
        f"- Validation accuracy: **{accuracy:.4f}**\n"
        f"- Train images: {n_train}\n"
        f"- Test images: {len(y_true)}\n"
        f"- Image size: {image_size}x{image_size}\n"
        f"- Limit per class: {limit if limit is not None else 'all'}\n"
        f"- Features: HOG + HSV color histogram\n"
        f"- LinearSVC C: {c_value}\n\n"
        f"See `confusion_matrix.png`, `sample_predictions.png`, and `metrics.json`.\n"
    )
    (report_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved report to {report_dir}")


def train(args: argparse.Namespace) -> None:
    samples = find_labeled_images(args.data_dir)
    n_cats = sum(label == 0 for _, label in samples)
    n_dogs = sum(label == 1 for _, label in samples)
    print(f"Found {len(samples)} labeled images ({n_cats} cats, {n_dogs} dogs)")
    features, labels, paths = build_dataset(
        samples, args.image_size, args.limit, args.random_state
    )
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    test_paths = [paths[i] for i in test_idx]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=args.c,
                    class_weight="balanced",
                    random_state=args.random_state,
                    max_iter=25000,
                    dual=False,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(np.mean(predictions == y_test))

    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=list(CLASS_NAMES), zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "image_size": args.image_size}, args.model_out)
    print(f"Saved model to {args.model_out}")
    save_report(
        args.report_dir,
        y_true=y_test,
        y_pred=predictions,
        paths=test_paths,
        accuracy=accuracy,
        image_size=args.image_size,
        limit=args.limit,
        n_train=len(y_train),
        c_value=args.c,
    )


def predict(model_path: Path, image_path: Path) -> None:
    if not model_path.is_file():
        raise SystemExit(f"Trained model not found: {model_path}")
    bundle = joblib.load(model_path)
    features = extract_features(image_path, bundle["image_size"]).reshape(1, -1)
    label = int(bundle["model"].predict(features)[0])
    score = float(bundle["model"].decision_function(features)[0])
    print(f"{image_path}: {'dog' if label else 'cat'} (decision score: {score:.4f})")


def main() -> None:
    args = parse_args()
    if args.predict:
        predict(args.model_out, args.predict)
        return
    if args.download:
        args.data_dir = download_dataset()
    elif args.data_dir is None:
        args.data_dir = find_local_dataset()
        if args.data_dir is None:
            raise SystemExit(
                "No local dataset found. Put images in .\\data, or pass --data-dir. "
                "Use --download only if you want to fetch from Kaggle."
            )
        print(f"Using local dataset: {args.data_dir}")
    train(args)


if __name__ == "__main__":
    main()
