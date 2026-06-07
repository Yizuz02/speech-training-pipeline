"""
Módulo 3: Carga y División del Dataset
─────────────────────────────────────────
Estructura esperada del directorio de datos:

    data/
    ├── si/
    │   ├── si_001.wav
    │   ├── si_002.wav
    │   └── ...
    ├── no/
    │   ├── no_001.wav
    │   └── ...
    ├── arriba/
    └── ...

Cada subcarpeta = una clase / palabra.
Compatible con corpus libres como Google Speech Commands.
"""

import os
import random
import numpy as np
from typing import Optional
from preprocessing import preprocess_audio
from feature_extraction import extract_full_features


def load_dataset(
    data_dir: str,
    sr: int = 16_000,
    n_mfcc: int = 13,
    use_deltas: bool = True,
    as_vector: bool = True,
    max_per_class: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list, list[int], list[str]]:
    """
    Carga el dataset completo desde un directorio con subcarpetas por clase.

    Args:
        data_dir: Directorio raíz del dataset
        sr: Tasa de muestreo objetivo
        n_mfcc: Coeficientes MFCC
        use_deltas: Incluir deltas y delta-deltas
        as_vector: True → vector de estadísticas, False → matriz temporal
        max_per_class: Límite de muestras por clase (None = todas)
        verbose: Mostrar progreso

    Returns:
        X: Lista de arrays de características (uno por audio)
        y: Lista de etiquetas numéricas
        labels: Lista de nombres de clases (índice = etiqueta numérica)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No se encontró el directorio '{data_dir}'.\n"
            "Crea la carpeta 'data/' con subcarpetas por palabra.\n"
            "Ejemplo: data/si/, data/no/, data/arriba/"
        )

    # Obtener clases (subcarpetas)
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not labels:
        raise ValueError(f"No se encontraron subcarpetas en '{data_dir}'.")

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    X, y = [], []
    errors = 0

    for label in labels:
        class_dir = os.path.join(data_dir, label)
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
        ]

        if max_per_class:
            random.shuffle(files)
            files = files[:max_per_class]

        if verbose:
            print(f"  Cargando clase '{label}': {len(files)} archivos")

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                signal = preprocess_audio(fpath, sr=sr)
                features = extract_full_features(
                    signal,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    use_deltas=use_deltas,
                    as_vector=as_vector,
                )
                X.append(features)
                y.append(label_to_idx[label])
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"    [WARN] Error en {fname}: {e}")

    if verbose and errors > 0:
        print(f"  [!] {errors} archivos no pudieron cargarse.")

    return X, y, labels


def train_test_split_dataset(
    X: list,
    y: list[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    Aplica estratificación por clase para mantener proporciones.

    Args:
        X: Lista de vectores de características
        y: Lista de etiquetas
        test_size: Fracción para prueba (0.0 – 1.0)
        seed: Semilla aleatoria para reproducibilidad

    Returns:
        X_train, X_test, y_train, y_test como arrays numpy
    """
    random.seed(seed)
    np.random.seed(seed)

    X = np.array(X) if not isinstance(X[0], np.ndarray) else np.stack(X)
    y = np.array(y)

    # Estratificación manual
    indices_train, indices_test = [], []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0].tolist()
        random.shuffle(cls_indices)
        n_test = max(1, int(len(cls_indices) * test_size))
        indices_test.extend(cls_indices[:n_test])
        indices_train.extend(cls_indices[n_test:])

    return (
        X[indices_train], X[indices_test],
        y[indices_train], y[indices_test],
    )


def generate_dummy_dataset(
    data_dir: str = "data",
    words: list[str] = None,
    n_per_class: int = 20,
    sr: int = 16_000,
    duration: float = 1.0,
) -> None:
    """
    Genera un dataset sintético de tonos puros para probar el pipeline
    sin necesidad de grabar audios reales.

    Cada clase es una señal sinusoidal de frecuencia distinta
    con ruido gaussiano añadido para mayor realismo.

    Args:
        data_dir: Directorio donde guardar los archivos
        words: Lista de etiquetas/clases
        n_per_class: Número de muestras por clase
        sr: Tasa de muestreo
        duration: Duración de cada audio en segundos
    """
    import soundfile as sf

    if words is None:
        words = ["si", "no", "arriba", "abajo", "izquierda",
                 "derecha", "abre", "cierra", "para", "sigue"]

    os.makedirs(data_dir, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Frecuencias base distintas por clase (simula formantes diferentes)
    base_freqs = np.linspace(200, 800, len(words))

    print(f"Generando dataset sintético en '{data_dir}'...")
    for i, word in enumerate(words):
        class_dir = os.path.join(data_dir, word)
        os.makedirs(class_dir, exist_ok=True)

        freq = base_freqs[i]
        for j in range(n_per_class):
            # Tono puro + armónicos + ruido
            signal = (
                0.6 * np.sin(2 * np.pi * freq * t)
                + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
                + 0.1 * np.random.randn(len(t)) * 0.05
            ).astype(np.float32)

            path = os.path.join(class_dir, f"{word}_{j:03d}.wav")
            sf.write(path, signal, sr)

    print(f"✓ {len(words)} clases × {n_per_class} muestras generadas.")


# ─── Demo rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generar datos sintéticos si no existe el directorio
    if not os.path.isdir("data"):
        generate_dummy_dataset()

    X, y, labels = load_dataset("data")
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    print(f"\nClases: {labels}")
    print(f"Dimensión de características: {X_train[0].shape}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
