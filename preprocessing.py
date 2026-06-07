"""
Módulo 1: Preprocesamiento de Audio
────────────────────────────────────
- Carga y conversión a mono
- Remuestreo a 16 kHz
- Eliminación de ruido (filtro de Wiener espectral)
- Normalización de amplitud
- Pre-énfasis (resalta altas frecuencias, mejora MFCCs)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener


def load_audio(file_path: str, sr: int = 16_000) -> tuple[np.ndarray, int]:
    """
    Carga un archivo de audio y lo convierte a mono a la tasa de muestreo indicada.

    Args:
        file_path: Ruta al archivo (.wav, .flac, .mp3, etc.)
        sr: Tasa de muestreo objetivo en Hz (default 16000)

    Returns:
        (señal, sr) — numpy array float32 en [-1, 1] y tasa de muestreo
    """
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr


def remove_noise_wiener(signal: np.ndarray) -> np.ndarray:
    """
    Aplica filtro de Wiener para reducción de ruido.
    Suaviza la señal estimando la potencia de ruido localmente.

    El filtro de Wiener minimiza el error cuadrático medio entre
    la señal filtrada y la señal original limpia estimada.

    Args:
        signal: Señal de audio (numpy array)

    Returns:
        Señal filtrada (numpy array)
    """
    return wiener(signal, mysize=29).astype(np.float32)


def normalize_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Normaliza la amplitud de la señal al rango [-1, 1].
    Evita división por cero si la señal es silencio.

    Args:
        signal: Señal de audio

    Returns:
        Señal normalizada
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def preemphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Aplica filtro de pre-énfasis: y[n] = x[n] - coef * x[n-1]
    Amplifica frecuencias altas para compensar la caída natural
    del espectro vocal y mejorar la extracción de MFCCs.

    Args:
        signal: Señal de audio
        coef: Coeficiente de pre-énfasis (0.95 – 0.97 típico)

    Returns:
        Señal con pre-énfasis aplicado
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def trim_silence(signal: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
    """
    Recorta silencios al inicio y al final de la señal.

    Args:
        signal: Señal de audio
        sr: Tasa de muestreo
        top_db: Umbral en dB bajo el cual se considera silencio

    Returns:
        Señal recortada
    """
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def preprocess_audio(
    file_path: str,
    sr: int = 16_000,
    apply_noise_filter: bool = True,
    apply_preemphasis: bool = True,
    trim: bool = True,
) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento para un archivo de audio.

    Pasos:
        1. Carga y conversión a mono / 16 kHz
        2. Recorte de silencios
        3. Filtro de Wiener (reducción de ruido)
        4. Normalización de amplitud
        5. Pre-énfasis

    Args:
        file_path: Ruta al archivo de audio
        sr: Tasa de muestreo objetivo
        apply_noise_filter: Si se aplica filtro de Wiener
        apply_preemphasis: Si se aplica pre-énfasis
        trim: Si se recortan silencios

    Returns:
        Señal preprocesada como numpy array float32
    """
    # 1. Carga
    signal, sr = load_audio(file_path, sr=sr)

    # 2. Recorte de silencios
    if trim:
        signal = trim_silence(signal, sr)

    # 3. Filtro de Wiener
    if apply_noise_filter:
        signal = remove_noise_wiener(signal)

    # 4. Normalización
    signal = normalize_amplitude(signal)

    # 5. Pre-énfasis
    if apply_preemphasis:
        signal = preemphasis(signal)

    return signal


def save_audio(signal: np.ndarray, sr: int, output_path: str) -> None:
    """Guarda la señal procesada como archivo .wav."""
    sf.write(output_path, signal, sr)


# ─── Demo rápida (ejecutar este módulo directamente) ────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python preprocessing.py <ruta_audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    signal = preprocess_audio(path)
    print(f"Señal preprocesada: {len(signal)} muestras @ 16 kHz")
    print(f"Rango: [{signal.min():.4f}, {signal.max():.4f}]")
    save_audio(signal, 16_000, "output_preprocessed.wav")
    print("Guardado: output_preprocessed.wav")
