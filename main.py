"""
Sistema de Reconocimiento de Voz (ASR) - ESCOM IPN
Proyecto Final - Procesamiento Digital de Señales
Ejecutar: python main.py
"""

import os
import numpy as np
from preprocessing import preprocess_audio
from feature_extraction import extract_mfcc
from dataset import load_dataset, train_test_split_dataset
from model_hmm import HMMRecognizer
from model_nn import NeuralRecognizer
from evaluation import evaluate_model, compare_models
from inference import recognize

# ─────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────
SAMPLE_RATE   = 16_000          # Hz
N_MFCC        = 13              # Coeficientes MFCC
DATA_DIR      = "data"          # Carpeta con audios organizados por carpeta/etiqueta
RESULTS_DIR   = "results"
TEST_SIZE     = 0.2             # 20% para prueba
RANDOM_SEED   = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO DE VOZ - ESCOM IPN")
    print("=" * 60)

    # 1. CARGAR DATASET
    print("\n[1/5] Cargando dataset...")
    X, y, labels = load_dataset(DATA_DIR, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    print(f"      Muestras cargadas: {len(X)} | Clases: {labels}")

    # 2. DIVIDIR DATASET
    print("\n[2/5] Dividiendo train / test...")
    X_train, X_test, y_train, y_test = train_test_split_dataset(
        X, y, test_size=TEST_SIZE, seed=RANDOM_SEED
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. ENTRENAR MODELOS
    print("\n[3/5] Entrenando modelos...")

    hmm_model = HMMRecognizer(n_components=5, n_iter=100)
    hmm_model.fit(X_train, y_train, labels)
    print("      HMM/GMM entrenado")

    nn_model = NeuralRecognizer(
        input_dim=N_MFCC * 3,  # mfcc + delta + delta2
        hidden_dim=128,
        output_dim=len(labels),
        epochs=50
    )
    nn_model.fit(X_train, y_train)
    print("      Red Neuronal entrenada")

    # 4. EVALUAR
    print("\n[4/5] Evaluando modelos...")
    hmm_metrics = evaluate_model(hmm_model, X_test, y_test, labels, name="HMM")
    nn_metrics  = evaluate_model(nn_model,  X_test, y_test, labels, name="Red Neuronal")
    compare_models(hmm_metrics, nn_metrics, save_path=RESULTS_DIR)

    # 5. INFERENCIA DEMO
    print("\n[5/5] Inferencia de ejemplo...")
    demo_files = [
        os.path.join(DATA_DIR, label, os.listdir(os.path.join(DATA_DIR, label))[0])
        for label in labels
        if os.path.isdir(os.path.join(DATA_DIR, label))
    ]
    for path in demo_files[:3]:
        pred_hmm = recognize(path, hmm_model, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        pred_nn  = recognize(path, nn_model,  sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        real     = os.path.basename(os.path.dirname(path))
        print(f"      Archivo: {real:10s} | HMM: {pred_hmm:10s} | NN: {pred_nn}")

    print("\nPipeline completado. Resultados en /results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
