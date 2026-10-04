from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

from inf8239_u02_cv.data import CLASS_NAMES, normalize_images, validate_images
from inf8239_u02_cv.models import build_cnn, build_dense

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MODEL_LABELS = {"dense": "Densa (baseline)", "cnn": "CNN"}
TRAIN_COLOR, VALID_COLOR = "#2a78d6", "#eb6834"
INFERENCE_REPEATS = 5
ERROR_PAIRS, ERRORS_PER_PAIR = 4, 4
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def runtime_info() -> dict:
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "os": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
    }


def save_inference_model(model, path: Path) -> None:
    """Guarda solo arquitectura y pesos: sin el estado del optimizador, que no se usa al predecir."""
    inference = model.__class__.from_config(model.get_config())
    inference.set_weights(model.get_weights())
    inference.save(path)


def measure_inference(model, images: np.ndarray, repeats: int) -> tuple[np.ndarray, list[float]]:
    """Predice una vez para calentar (descartada) y mide `repeats` pasadas sobre el mismo lote."""
    probabilities = model.predict(images, verbose=0)
    runs = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict(images, verbose=0)
        runs.append(1000 * (perf_counter() - start) / len(images))
    return probabilities, runs


def describe_split(name: str, images: np.ndarray, labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    print(f"{name}: forma {images.shape}, rango [{images.min():.1f}, {images.max():.1f}], "
          f"{len(np.unique(labels))} clases, por clase {counts.min()}–{counts.max()}")
    return {"shape": list(images.shape), "per_class": dict(zip(CLASS_NAMES, counts.tolist()))}


def plot_learning_curves(histories: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, len(histories), figsize=(10, 3.8), sharey=True)
    for axis, (name, history) in zip(axes, histories.items()):
        epochs = np.arange(1, len(history["loss"]) + 1)
        axis.plot(epochs, history["loss"], color=TRAIN_COLOR, linewidth=2, marker="o",
                  markersize=4, label="entrenamiento")
        axis.plot(epochs, history["val_loss"], color=VALID_COLOR, linewidth=2, marker="o",
                  markersize=4, label="validación")
        best = int(np.argmin(history["val_loss"])) + 1
        axis.axvline(best, color="#9a9890", linestyle=":", linewidth=1)
        axis.set_title(f"{MODEL_LABELS[name]} · mejor época {best} de {len(epochs)}")
        axis.set_xlabel("época")
        axis.set_xticks(epochs)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pérdida (entropía cruzada)")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                         for row in rows)


def select_error_examples(labels: np.ndarray, predictions: np.ndarray,
                          probabilities: np.ndarray) -> list[dict]:
    """Para los pares más confundidos, los errores con mayor y menor confianza."""
    errors = confusion_matrix(labels, predictions)
    np.fill_diagonal(errors, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-errors, axis=None), errors.shape))[0]
    rows = []
    for real, predicted in pairs[:ERROR_PAIRS]:
        indices = np.where((labels == real) & (predictions == predicted))[0]
        ranked = indices[np.argsort(-probabilities[indices, predicted])]
        half = ERRORS_PER_PAIR // 2
        for kind, chosen in (("mayor confianza", ranked[:half]), ("menor confianza", ranked[-half:])):
            rows += [{"test_index": int(i), "real": CLASS_NAMES[real],
                      "predicted": CLASS_NAMES[predicted], "pair_errors": int(errors[real, predicted]),
                      "confidence": float(probabilities[i, predicted]),
                      "real_class_probability": float(probabilities[i, real]), "group": kind}
                     for i in chosen]
    return rows


def plot_confusion(labels: np.ndarray, predictions: np.ndarray, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=CLASS_NAMES,
                                            cmap="Blues", xticks_rotation=45, ax=axis)
    axis.set_title(f"Matriz de confusión · {title} (prueba, 10.000 imágenes)")
    axis.set_xlabel("Clase predicha")
    axis.set_ylabel("Clase real")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_error_examples(images: np.ndarray, examples: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(ERROR_PAIRS, ERRORS_PER_PAIR, figsize=(9, 10.5))
    for axis, example in zip(axes.ravel(), examples):
        axis.imshow(images[example["test_index"]].squeeze(), cmap="gray")
        axis.set_title(f"Real: {example['real']}\nPred: {example['predicted']} "
                       f"({example['confidence']:.0%})", fontsize=9)
        axis.axis("off")
    figure.suptitle("Errores de la CNN en los 4 pares más confundidos\n"
                    "Columnas 1–2: mayor confianza · columnas 3–4: menor confianza", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_all = normalize_images(x_train_all)
    x_test = normalize_images(x_test)
    x_train, y_train = x_train_all[:-6000], y_train_all[:-6000]
    x_valid, y_valid = x_train_all[-6000:], y_train_all[-6000:]
    validate_images(x_train, y_train)
    validate_images(x_valid, y_valid)
    validate_images(x_test, y_test)
    data_summary = {name: describe_split(name, images, labels) for name, images, labels in [
        ("entrenamiento", x_train, y_train),
        ("validación", x_valid, y_valid),
        ("prueba", x_test, y_test),
    ]}
    (reports / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {}
    trained = {}
    histories = {}
    per_class = []
    for name, factory in {"dense": build_dense, "cnn": build_cnn}.items():
        model = compile_model(factory(tf))
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        if name == "cnn":
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(models_dir / "best_cnn.keras", save_best_only=True))
        start = perf_counter()
        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.epochs,
                  batch_size=128, callbacks=callbacks, verbose=2)
        train_seconds = perf_counter() - start
        probabilities, inference_runs = measure_inference(model, x_test, INFERENCE_REPEATS)
        predictions = probabilities.argmax(axis=1)
        histories[name] = {key: [float(v) for v in values] for key, values in history.history.items()}
        report = classification_report(y_test, predictions, target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        per_class += [{"model": name, "class": label, "precision": report[label]["precision"],
                       "recall": report[label]["recall"], "f1": report[label]["f1-score"],
                       "support": int(report[label]["support"])} for label in CLASS_NAMES]
        metrics[name] = {
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "accuracy": float((predictions == y_test).mean()),
            "parameters": int(model.count_params()),
            "train_seconds": train_seconds,
            "inference_ms_per_image": float(np.median(inference_runs)),
            "inference_ms_runs": [round(run, 5) for run in inference_runs],
            "epochs_run": len(history.history["val_loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        }
        trained[name] = (model, predictions, probabilities)
        print(name, json.dumps(metrics[name], indent=2))
    # Se guarda al final: reconstruir un modelo consume el generador aleatorio y alteraría
    # la inicialización del siguiente modelo entrenado.
    for name, (model, _, _) in trained.items():
        model_path = models_dir / f"{name}.keras"
        save_inference_model(model, model_path)
        metrics[name]["model_file"] = model_path.name
        metrics[name]["model_file_kb"] = round(model_path.stat().st_size / 1024, 1)
    (reports / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    write_rows(per_class, reports / "per_class_metrics.csv")
    plot_learning_curves(histories, reports / "learning_curves.png")
    for name, (_, model_predictions, _) in trained.items():
        plot_confusion(y_test, model_predictions, MODEL_LABELS[name], reports / f"confusion_{name}.png")
    _, predictions, probabilities = trained["cnn"]
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=3))
    examples = select_error_examples(y_test, predictions, probabilities)
    plot_error_examples(x_test, examples, reports / "cnn_errors.png")
    write_rows(examples, reports / "cnn_error_examples.csv")
    (reports / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports / "runtime.json").write_text(json.dumps(runtime_info(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
