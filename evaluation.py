"""
Módulo 5: Evaluación de Modelos
─────────────────────────────────
- Accuracy (tasa de aciertos)
- WER — Word Error Rate
- Matriz de confusión
- Reporte por clase (precision, recall, F1)
- Comparativa entre modelos con gráficas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # Renderizado sin pantalla (servidor / CI)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ──────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """
    Calcula el Word Error Rate (WER) a nivel de palabra.

    WER = (S + D + I) / N
    donde:
        S = sustituciones
        D = eliminaciones
        I = inserciones
        N = total de palabras reales

    Para reconocimiento de palabras aisladas (vocabulario pequeño),
    WER ≈ 1 - Accuracy, ya que cada "oración" es una sola palabra.
    Se incluye el cálculo completo para demostración académica.

    Args:
        references:  Lista de palabras/frases reales
        hypotheses:  Lista de palabras/frases predichas

    Returns:
        WER como flotante en [0, ∞)
    """
    total_words, total_errors = 0, 0

    for ref, hyp in zip(references, hypotheses):
        r = ref.lower().split()
        h = hyp.lower().split()
        n = len(r)
        total_words += n

        # Programación dinámica: distancia de edición
        dp = np.zeros((n + 1, len(h) + 1), dtype=int)
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(len(h) + 1):
            dp[0, j] = j

        for i in range(1, n + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,       # eliminación
                    dp[i, j - 1] + 1,       # inserción
                    dp[i - 1, j - 1] + cost # sustitución
                )
        total_errors += dp[n, len(h)]

    return total_errors / max(total_words, 1)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    name: str = "Modelo",
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Evalúa un modelo completo y genera reporte + gráficas.

    Args:
        model: Modelo con método .predict()
        X_test: Características de prueba
        y_test: Etiquetas reales
        labels: Nombres de las clases
        name: Nombre del modelo (para títulos)
        save_path: Directorio donde guardar figuras (None = no guardar)
        verbose: Mostrar resultados en consola

    Returns:
        Diccionario con métricas: accuracy, wer, report, confusion_matrix
    """
    y_pred = model.predict(X_test)

    # Convertir índices a nombres de clase
    refs = [labels[i] for i in y_test]
    hyps = [labels[i] for i in y_pred]

    acc = accuracy_score(y_test, y_pred)
    wer = compute_wer(refs, hyps)
    cm  = confusion_matrix(y_test, y_pred)
    rep_str = classification_report(y_test, y_pred, target_names=labels)
    rep_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  Accuracy : {acc:.4f}  ({acc:.2%})")
        print(f"  WER      : {wer:.4f}  ({wer:.2%})")
        print(f"\n{rep_str}")

    # ── Gráficas ──────────────────────────────────────────────────────────
    if save_path:
        _plot_confusion_matrix(cm, labels, name, save_path)
        if hasattr(model, "history"):
            _plot_training_history(model.history, name, save_path)

    return {
        "name":             name,
        "accuracy":         acc,
        "wer":              wer,
        "report":           rep_dict,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
    }


def plot_model(
    model_metrics: dict,
    save_path: str = None,
) -> None:

    report = model_metrics["report"]

    labels = []
    f1_scores = []

    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue

        labels.append(label)
        f1_scores.append(metrics["f1-score"])

    plt.figure(figsize=(12, 6))
    plt.bar(labels, f1_scores)

    plt.title(f"F1-Score por clase - {model_metrics['name']}")
    plt.xlabel("Clase")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        f_name = f"f1_scores_{model_metrics['name'].lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_path, f_name), dpi=300, bbox_inches="tight")


def compare_models(
    *metrics_dicts,
    save_path: str = None,
) -> None:
    """
    Genera una tabla y gráfica comparativa entre múltiples modelos.

    Args:
        *metrics_dicts: Resultados de evaluate_model() para cada modelo
        save_path: Directorio donde guardar la figura comparativa
    """
    print(f"\n{'═'*50}")
    print("  COMPARATIVA DE MODELOS")
    print(f"{'═'*50}")
    print(f"  {'Modelo':<20} {'Accuracy':>10} {'WER':>10}")
    print(f"  {'─'*40}")
    for m in metrics_dicts:
        print(f"  {m['name']:<20} {m['accuracy']:>10.2%} {m['wer']:>10.2%}")
    print(f"{'═'*50}")

    if save_path:
        _plot_comparison(metrics_dicts, save_path)

# ──────────────────────────────────────────────────────────────
# Funciones de graficación (internas)
# ──────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    model_name: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fname = f"confusion_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(save_path, fname), dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {fname}")


def _plot_training_history(
    history: dict,
    model_name: str,
    save_path: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title(f"Curva de pérdida — {model_name}")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy validación
    ax2.plot(epochs, history["val_acc"], color="green", label="Val accuracy")
    ax2.set_title(f"Accuracy de validación — {model_name}")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"history_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(save_path, fname), dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {fname}")


def _plot_comparison(metrics_list: tuple, save_path: str) -> None:
    names    = [m["name"]     for m in metrics_list]
    accs     = [m["accuracy"] for m in metrics_list]
    wers     = [m["wer"]      for m in metrics_list]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, wers, w, label="WER",      color="#DD8452")

    ax.set_ylabel("Valor")
    ax.set_title("Comparativa de Modelos ASR")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fname = "comparativa_modelos.png"
    fig.savefig(os.path.join(save_path, fname), dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {fname}")
