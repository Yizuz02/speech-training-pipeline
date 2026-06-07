"""
Módulo 6: Inferencia en tiempo real o por archivo
──────────────────────────────────────────────────
- Reconocimiento desde archivo .wav
- Reconocimiento desde micrófono en tiempo real
- Función de alto nivel `recognize()` compatible con cualquier modelo
"""

import numpy as np
from preprocessing import preprocess_audio
from feature_extraction import extract_full_features


def recognize(
    audio_input,
    model,
    sr: int = 16_000,
    n_mfcc: int = 13,
    use_deltas: bool = True,
    use_vector: bool = True,
) -> str:
    """
    Reconoce una palabra desde un archivo de audio o señal numpy.

    Flujo:
        audio → preprocesamiento → MFCC → modelo.predict_single()

    Args:
        audio_input: Ruta a archivo .wav (str) o señal numpy (np.ndarray)
        model: Modelo entrenado (HMMRecognizer o NeuralRecognizer)
        sr: Tasa de muestreo
        n_mfcc: Coeficientes MFCC (debe coincidir con el entrenamiento)
        use_deltas: Incluir deltas (debe coincidir con el entrenamiento)

    Returns:
        Nombre de la clase/palabra reconocida
    """
    # 1. Cargar o usar señal directamente
    if isinstance(audio_input, str):
        signal = preprocess_audio(audio_input, sr=sr)
    elif isinstance(audio_input, np.ndarray):
        signal = audio_input
    else:
        raise TypeError("audio_input debe ser str (ruta) o np.ndarray")

    # 2. Extraer características
    features_vector, features_matrix = extract_full_features(
        signal, sr=sr, n_mfcc=n_mfcc, use_deltas=use_deltas
    )

    # 3. Predecir
    if use_vector:
        features = features_vector
    else:
        features = features_matrix
    word, confidence = model.predict_single(features)
    return word, confidence 


def recognize_with_confidence(
    audio_input,
    model,
    sr: int = 16_000,
    n_mfcc: int = 13,
    use_deltas: bool = True,
) -> tuple[str, float]:
    """
    Como recognize(), pero también retorna la confianza del modelo.

    Returns:
        (palabra, confianza)  — confianza en [0, 1] para NN, log-prob para HMM
    """
    if isinstance(audio_input, str):
        signal = preprocess_audio(audio_input, sr=sr)
    else:
        signal = audio_input

    features = extract_full_features(
        signal, sr=sr, n_mfcc=n_mfcc, use_deltas=use_deltas, as_vector=True
    )
    return model.predict_single(features)


def recognize_from_microphone(
    model,
    sr: int = 16_000,
    duration: float = 2.0,
    n_mfcc: int = 13,
) -> str:
    """
    Captura audio del micrófono y lo reconoce.

    Requiere: pip install sounddevice

    Args:
        model: Modelo entrenado
        sr: Tasa de muestreo
        duration: Segundos de grabación
        n_mfcc: Coeficientes MFCC

    Returns:
        Palabra reconocida
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Instala sounddevice: pip install sounddevice")

    print(f"  Grabando {duration}s... (habla ahora)")
    recording = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    signal = recording.flatten()

    return recognize(signal, model, sr=sr, n_mfcc=n_mfcc)


def batch_recognize(
    file_paths: list[str],
    model,
    sr: int = 16_000,
    n_mfcc: int = 13,
) -> list[dict]:
    """
    Reconoce una lista de archivos de audio.

    Args:
        file_paths: Lista de rutas a archivos .wav
        model: Modelo entrenado
        sr: Tasa de muestreo
        n_mfcc: Coeficientes MFCC

    Returns:
        Lista de dicts con {"file", "prediction", "confidence"}
    """
    results = []
    for path in file_paths:
        try:
            word, conf = recognize_with_confidence(path, model, sr=sr, n_mfcc=n_mfcc)
            results.append({"file": path, "prediction": word, "confidence": conf})
        except Exception as e:
            results.append({"file": path, "prediction": "ERROR", "confidence": 0.0, "error": str(e)})
    return results


# ─── Demo interactiva desde micrófono ────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import pickle

    if len(sys.argv) < 3:
        print("Uso: python inference.py <modelo.pkl|modelo.pt> <audio.wav|mic>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_arg  = sys.argv[2]

    # Cargar modelo
    if model_path.endswith(".pt"):
        from model_nn import NeuralRecognizer
        model = NeuralRecognizer.load(model_path)
    else:
        from model_hmm import HMMRecognizer
        model = HMMRecognizer.load(model_path)

    # Inferencia
    if audio_arg == "mic":
        word = recognize_from_microphone(model)
    else:
        word = recognize(audio_arg, model)

    print(f"  Predicción: {word}")
