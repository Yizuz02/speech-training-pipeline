"""
Módulo 4A: Modelo HMM / GMM
─────────────────────────────
Hidden Markov Model con emisiones Gaussian Mixture Model.

Intuición:
    - Cada palabra es una secuencia de estados ocultos (fonemas aproximados).
    - Cada estado emite vectores de características según una distribución
      gaussiana (o mezcla de gaussianas).
    - El reconocimiento busca el modelo que maximiza P(observaciones | modelo).

Dependencia: pip install hmmlearn
"""

import os
import pickle
import numpy as np
from hmmlearn import hmm


class HMMRecognizer:
    """
    Reconocedor ASR basado en HMM con emisiones gaussianas.
    Entrena un HMM independiente por clase (palabra).

    Attributes:
        n_components: Número de estados ocultos por HMM (≈ fonemas)
        n_iter: Iteraciones del algoritmo Baum-Welch (EM)
        models: Diccionario {clase: GaussianHMM entrenado}
        labels: Lista de nombres de clases
    """

    def __init__(self, n_components: int = 5, n_iter: int = 100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.models: dict[int, hmm.GaussianHMM] = {}
        self.labels: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels: list[str],
    ) -> "HMMRecognizer":
        """
        Entrena un HMM por clase usando el algoritmo Baum-Welch (EM).

        Para HMMs con vectores de características de longitud fija,
        tratamos cada vector como una "secuencia" de longitud 1.
        Para usar secuencias temporales reales, pasar as_vector=False
        en extract_full_features.

        Args:
            X: Matriz de características (n_samples, n_features)
            y: Etiquetas numéricas (n_samples,)
            labels: Nombres de las clases

        Returns:
            self (encadenamiento de métodos)
        """
        self.labels = labels
        classes = np.unique(y)

        for cls in classes:
            cls_mask = y == cls
            X_cls = X[cls_mask]

            # hmmlearn espera secuencias concatenadas con lengths
            # Aquí cada muestra es un "frame" de observación
            sequences = X_cls                # (n_samples, n_features)
            lengths   = [1] * len(X_cls)    # cada muestra = secuencia de 1

            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",  # covarianza diagonal (eficiente)
                n_iter=self.n_iter,
                random_state=42,
            )

            try:
                model.fit(sequences, lengths)
                self.models[int(cls)] = model
            except Exception as e:
                print(f"  [WARN] HMM clase {labels[cls]}: {e}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de cada muestra usando el criterio MAP:
        argmax_k P(x | HMM_k)

        Args:
            X: Matriz de características (n_samples, n_features)

        Returns:
            Array de etiquetas predichas (n_samples,)
        """
        predictions = []
        for x in X:
            obs = x.reshape(1, -1)  # (1, n_features)
            scores = {}
            for cls, model in self.models.items():
                try:
                    score = model.score(obs, [1])
                except Exception:
                    score = -np.inf
                scores[cls] = score
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

    def predict_single(self, x: np.ndarray) -> tuple[str, float]:
        """
        Predice la clase de una sola muestra.

        Args:
            x: Vector de características (n_features,)

        Returns:
            (nombre_clase, log_verosimilitud)
        """
        obs = x.reshape(1, -1)
        best_cls, best_score = None, -np.inf
        for cls, model in self.models.items():
            try:
                score = model.score(obs, [1])
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_cls = cls
        return self.labels[best_cls], best_score

    def save(self, path: str) -> None:
        """Guarda el modelo entrenado en disco."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modelo HMM guardado: {path}")

    @classmethod
    def load(cls, path: str) -> "HMMRecognizer":
        """Carga un modelo previamente guardado."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── Demo rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import generate_dummy_dataset, load_dataset, train_test_split_dataset

    generate_dummy_dataset(n_per_class=30)
    X, y, labels = load_dataset("data", verbose=False)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    model = HMMRecognizer(n_components=3, n_iter=50)
    model.fit(X_train, y_train, labels)

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"HMM Accuracy en test: {acc:.2%}")
