"""
Sistema de Reconocimiento de Voz (ASR) - ESCOM IPN
Proyecto Final - Procesamiento Digital de Señales
Ejecutar: python main.py
"""

import os
import numpy as np
from preprocessing import preprocess_audio
from feature_extraction import extract_mfcc
from dataset import load_dataset, train_test_split_dataset, train_test_split_dataset_hmm
from model_nn import NeuralRecognizer
from model_hmm import HMMRecognizer
from evaluation import evaluate_model, plot_model, compare_models
from inference import recognize

# ─────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────
SAMPLE_RATE   = 16_000          # Hz
N_MFCC        = 13              # Coeficientes MFCC
DATA_DIR      = "data"          # Carpeta con audios organizados por carpeta/etiqueta
RESULTS_DIR   = "results"
EXAMPLE_DIR   = "examples"       # Audios de ejemplo para inferencia demo
MODELS_DIR    = "models"         # Carpeta para guardar modelos entrenados
TEST_SIZE     = 0.2             # 20% para prueba
RANDOM_SEED   = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO DE VOZ - ESCOM IPN")
    print("=" * 60)

    # 1. CARGAR DATASET
    print("\n[1/5] Cargando dataset...")
    X_nn, y_nn, X_hmm, y_hmm, labels = load_dataset(DATA_DIR, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    print(f"      Muestras cargadas: {len(X_hmm)} | Clases: {labels}")

    # 2. DIVIDIR DATASET
    print("\n[2/5] Dividiendo train / test...")
    X_train_hmm, X_test_hmm, y_train_hmm, y_test_hmm = train_test_split_dataset_hmm(
        X_hmm, y_hmm, test_size=TEST_SIZE, seed=RANDOM_SEED
    )
    print(f"      HMM Train: {len(X_train_hmm)} | HMM Test: {len(X_test_hmm)}")
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split_dataset(
        X_nn, y_nn, test_size=TEST_SIZE, seed=RANDOM_SEED
    )
    print(f"      Train: {len(X_train_nn)} | Test: {len(X_test_nn)}")

    # 3. ENTRENAR MODELOS
    print("\n[3/5] Entrenando modelos...")

    hmm_model = HMMRecognizer(n_components=5, n_iter=100)
    hmm_model.fit(X_train_hmm, y_train_hmm, labels)
    hmm_model.save(os.path.join(MODELS_DIR, "hmm_model.pkl"))
    print("      HMM entrenado")

    nn_model = NeuralRecognizer(
        input_dim=N_MFCC * 3 * 2,  
        hidden_dim=128,
        output_dim=len(labels),
        epochs=50
    )
    nn_model.fit(X_train_nn, y_train_nn, labels=labels)
    nn_model.save(os.path.join(MODELS_DIR, "nn_model.pt"))
    print("      Red Neuronal entrenada")

    # 4. EVALUAR
    print("\n[4/5] Evaluando modelos...")
    hmm_metrics = evaluate_model(hmm_model, X_test_hmm, y_test_hmm, labels, name="HMM")
    plot_model(hmm_metrics, save_path=RESULTS_DIR)
    nn_metrics  = evaluate_model(nn_model,  X_test_nn, y_test_nn, labels, name="Red Neuronal")
    plot_model(nn_metrics, save_path=RESULTS_DIR)

    compare_models(hmm_metrics, nn_metrics, save_path=RESULTS_DIR)

    # 5. INFERENCIA DEMO
    print("\n[5/5] Inferencia de ejemplo...")
    example_files = []
    for root, _, files in os.walk(EXAMPLE_DIR):
        for f in files:
            if f.endswith(".mp3"):
                example_files.append(os.path.join(root, f))

    print("\n" + "═" * 90)
    print(
        f"{'Real':<15} │ "
        f"{'Pred NN':<15} │ "
        f"{'Conf NN':>10} │ "
        f"{'Pred HMM':<15} │ "
        f"{'Score HMM':>12}"
    )
    print("─" * 90)

    for path in example_files:
        pred_nn_word, pred_nn_conf = recognize(
            path,
            nn_model,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            use_deltas=True,
            use_vector=True
        )

        pred_hmm_word, pred_hmm_conf = recognize(
            path,
            hmm_model,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            use_deltas=True,
            use_vector=False
        )

        real = os.path.basename(os.path.dirname(path))

        print(
            f"{real:<15} │ "
            f"{pred_nn_word:<15} │ "
            f"{pred_nn_conf:>10.4f} │ "
            f"{pred_hmm_word:<15} │ "
            f"{pred_hmm_conf:>12.2f}"
        )

    print("═" * 90)

    print("\nPipeline completado. Resultados en /results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
