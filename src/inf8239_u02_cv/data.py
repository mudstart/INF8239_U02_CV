from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo",
               "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
NUM_CLASSES = len(CLASS_NAMES)


def normalize_images(images: np.ndarray) -> np.ndarray:
    values = images.astype("float32") / 255.0
    return values[..., None]


def validate_images(images: np.ndarray, labels: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forma inesperada: {images.shape}")
    if len(images) != len(labels):
        raise ValueError("Imágenes y etiquetas no coinciden")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Las imágenes deben estar normalizadas entre 0 y 1")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Las etiquetas deben ser enteras, no {labels.dtype}")
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"Etiquetas fuera de 0–{NUM_CLASSES - 1}: [{labels.min()}, {labels.max()}]")
