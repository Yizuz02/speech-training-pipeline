"""
Módulo 2: Extracción de Características Acústicas
──────────────────────────────────────────────────
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Deltas y Delta-Deltas (información dinámica)
- Espectrograma Log-Mel (alternativa visual)
- Estadísticas resumen por vector de características
"""

import numpy as np
import librosa


# ────────────────────────────────────────────────────────
# MFCC — Extracción principal
# ────────────────────────────────────────────────────────

def extract_mfcc(
    signal: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,     # 10 ms @ 16kHz
    win_length: int = 400,     # 25 ms @ 16kHz
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float = 8_000.0,
) -> np.ndarray:
    """
    Extrae los coeficientes MFCC de una señal de audio.

    Proceso interno:
        1. STFT (Short-Time Fourier Transform)
        2. Banco de filtros Mel (percepción humana)
        3. Logaritmo de energías Mel
        4. DCT (Discrete Cosine Transform) → MFCCs

    Args:
        signal: Señal preprocesada (numpy array float32)
        sr: Tasa de muestreo (Hz)
        n_mfcc: Número de coeficientes (13–40 típico)
        n_fft: Tamaño de ventana FFT
        hop_length: Paso entre ventanas en muestras
        win_length: Tamaño de ventana de análisis
        n_mels: Número de filtros Mel
        fmin: Frecuencia mínima del banco Mel (Hz)
        fmax: Frecuencia máxima del banco Mel (Hz)

    Returns:
        MFCC matrix de forma (n_mfcc, T) donde T = número de frames
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window="hamming",
    )
    return mfcc


def extract_delta(mfcc: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Calcula deltas (velocidades) y delta-deltas (aceleraciones) de los MFCCs.
    Capturan la dinámica temporal de la voz (cómo cambian los coeficientes).

    Args:
        mfcc: Matriz MFCC (n_mfcc, T)
        order: 1 = solo delta, 2 = delta + delta-delta

    Returns:
        Concatenación: [mfcc | delta | delta²] de forma (n_mfcc*3, T)
    """
    features = [mfcc]

    delta1 = librosa.feature.delta(mfcc, order=1)
    features.append(delta1)

    if order >= 2:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(delta2)

    return np.vstack(features)


def features_to_vector(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Convierte la matriz de características (n_features, T) en un
    vector de estadísticas resumen de longitud fija.

    Estadísticas calculadas por coeficiente:
        - Media (captura valor central)
        - Desviación estándar (captura variabilidad)

    Esto permite que modelos sin soporte para secuencias variables
    (SVM, MLP simple) trabajen con vectores de tamaño fijo.

    Args:
        feature_matrix: (n_features, T)

    Returns:
        Vector 1D de longitud (n_features * 2)
    """
    mean = np.mean(feature_matrix, axis=1)
    std  = np.std(feature_matrix, axis=1)
    return np.concatenate([mean, std])


def extract_log_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
) -> np.ndarray:
    """
    Calcula el espectrograma Log-Mel.
    Alternativa a MFCCs: útil para redes neuronales convolucionales (CNN).

    Args:
        signal: Señal preprocesada
        sr: Tasa de muestreo
        n_fft: Tamaño de ventana FFT
        hop_length: Paso entre ventanas
        n_mels: Número de bandas Mel

    Returns:
        Log-Mel spectrogram de forma (n_mels, T)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window="hamming",
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def extract_full_features(
    signal: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 13,
    use_deltas: bool = True,
    as_vector: bool = True,
) -> np.ndarray:
    """
    Pipeline completo de extracción de características.

    Args:
        signal: Señal preprocesada
        sr: Tasa de muestreo
        n_mfcc: Número de coeficientes MFCC
        use_deltas: Si se incluyen deltas y delta-deltas
        as_vector: Si se retorna como vector de estadísticas (True)
                   o como matriz temporal (False)

    Returns:
        Vector 1D (para HMM/SVM/MLP) o matriz 2D (para RNN/LSTM)
    """
    mfcc = extract_mfcc(signal, sr=sr, n_mfcc=n_mfcc)

    if use_deltas:
        features = extract_delta(mfcc, order=2)
    else:
        features = mfcc

    if as_vector:
        return features_to_vector(features)
    else:
        return features  # (n_features, T) para modelos secuenciales


# ─── Demo rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from preprocessing import preprocess_audio

    if len(sys.argv) < 2:
        print("Uso: python feature_extraction.py <ruta_audio.wav>")
        sys.exit(1)

    signal = preprocess_audio(sys.argv[1])
    vec = extract_full_features(signal, as_vector=True)
    mat = extract_full_features(signal, as_vector=False)

    print(f"Vector de características: shape={vec.shape}")
    print(f"Matriz temporal:           shape={mat.shape}")
    print(f"Primeros 5 valores: {vec[:5]}")
