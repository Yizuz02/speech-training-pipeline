"""
Módulo 4B: Modelo Red Neuronal (MLP / Feedforward)
────────────────────────────────────────────────────
Red neuronal densa implementada en PyTorch.

Arquitectura:
    Input (n_features) → FC(128) → ReLU → Dropout(0.3)
                       → FC(64)  → ReLU → Dropout(0.3)
                       → FC(n_clases) → Softmax

Alternativa comentada al final: RNN / GRU para secuencias temporales.

Dependencia: pip install torch
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ──────────────────────────────────────────────
# Arquitectura de la red
# ──────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Red neuronal feedforward (MLP) para clasificación de palabras.

    Args:
        input_dim: Dimensión del vector de entrada (n_mfcc * 3 * 2)
        hidden_dim: Neuronas en capas ocultas
        output_dim: Número de clases (palabras)
        dropout: Probabilidad de dropout (regularización)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Capa 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Capa 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Capa de salida
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Wrapper entrenador / predictor
# ──────────────────────────────────────────────

class NeuralRecognizer:
    """
    Wrapper de alto nivel para entrenar y usar el MLP.

    Maneja:
    - Normalización de datos (StandardScaler manual)
    - Ciclo de entrenamiento con validación
    - Early stopping
    - Guardado / carga del modelo

    Args:
        input_dim: Dimensión del vector de características
        hidden_dim: Neuronas en capa oculta
        output_dim: Número de clases
        epochs: Máximo de épocas de entrenamiento
        batch_size: Tamaño de mini-lote
        lr: Tasa de aprendizaje inicial
        patience: Épocas sin mejora antes de detener entrenamiento
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
    ):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels: list[str] = []

        # Parámetros de normalización (calculados en fit)
        self._mean: np.ndarray = None
        self._std:  np.ndarray = None

        self.model: MLPClassifier = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ── Normalización ──────────────────────────────────────────────────────

    def _fit_scaler(self, X: np.ndarray) -> None:
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-8  # evitar /0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    # ── Entrenamiento ──────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.15,
        labels: list[str] = None,
        verbose: bool = True,
    ) -> "NeuralRecognizer":
        """
        Entrena la red con el algoritmo de retropropagación (Adam + ReduceLROnPlateau).

        Args:
            X: Características de entrenamiento (n_samples, n_features)
            y: Etiquetas (n_samples,)
            val_split: Fracción para validación interna
            labels: Nombres de las clases (opcional)
            verbose: Mostrar progreso por época

        Returns:
            self
        """
        if labels:
            self.labels = labels

        # Normalizar
        self._fit_scaler(X)
        X_norm = self._transform(X).astype(np.float32)
        y      = y.astype(np.int64)

        # Split validación
        n_val  = max(1, int(len(X_norm) * val_split))
        idx    = np.random.permutation(len(X_norm))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, X_val = X_norm[train_idx], X_norm[val_idx]
        y_train, y_val = y[train_idx],      y[val_idx]

        # Tensores
        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        # Modelo
        self.model = MLPClassifier(
            input_dim=X_norm.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = np.inf
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ── Entrenamiento ──
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_train)

            # ── Validación ──
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss   = criterion(val_logits, y_val_t).item()
                val_preds  = val_logits.argmax(dim=1)
                val_acc    = (val_preds == y_val_t).float().mean().item()

            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"      Early stopping en época {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"      Época {epoch:3d}/{self.epochs} | "
                    f"loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                    f"val_acc: {val_acc:.2%} | lr: {lr_now:.5f}"
                )

        if best_state:
            self.model.load_state_dict(best_state)

        return self

    # ── Predicción ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase para cada muestra.

        Args:
            X: Características (n_samples, n_features)

        Returns:
            Array de etiquetas predichas (n_samples,)
        """
        self.model.eval()
        X_norm = self._transform(X).astype(np.float32)
        X_t    = torch.from_numpy(X_norm).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds  = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por clase (softmax)."""
        self.model.eval()
        X_norm = self._transform(X).astype(np.float32)
        X_t    = torch.from_numpy(X_norm).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_single(self, x: np.ndarray) -> tuple[str, float]:
        """
        Predice una sola muestra.

        Returns:
            (nombre_clase, confianza)
        """
        proba = self.predict_proba(x.reshape(1, -1))[0]
        cls   = int(np.argmax(proba))
        name  = self.labels[cls] if self.labels else str(cls)
        return name, float(proba[cls])

    # ── Persistencia ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Guarda el modelo completo (pesos + normalizador + metadatos)."""
        state = {
            "model_state": self.model.state_dict(),
            "input_dim":   self.input_dim,
            "hidden_dim":  self.hidden_dim,
            "output_dim":  self.output_dim,
            "labels":      self.labels,
            "mean":        self._mean,
            "std":         self._std,
            "history":     self.history,
        }
        torch.save(state, path)
        print(f"  Modelo NN guardado: {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralRecognizer":
        """Carga un modelo guardado."""
        state = torch.load(path, map_location="cpu")
        rec = cls(
            input_dim=state["input_dim"],
            hidden_dim=state["hidden_dim"],
            output_dim=state["output_dim"],
        )
        rec.labels  = state["labels"]
        rec._mean   = state["mean"]
        rec._std    = state["std"]
        rec.history = state["history"]
        rec.model   = MLPClassifier(
            input_dim=rec.input_dim,
            hidden_dim=rec.hidden_dim,
            output_dim=rec.output_dim,
        )
        rec.model.load_state_dict(state["model_state"])
        rec.model.eval()
        return rec


# ══════════════════════════════════════════════════════════════════
#  ALTERNATIVA: RNN / GRU para secuencias temporales (comentado)
# ══════════════════════════════════════════════════════════════════
#
# class GRUClassifier(nn.Module):
#     """RNN bidireccional para clasificar secuencias de MFCCs."""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_dim, hidden_dim, num_layers=num_layers,
#             batch_first=True, bidirectional=True, dropout=dropout
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, x):
#         # x: (batch, T, n_mfcc)
#         out, _ = self.gru(x)
#         # Pooling temporal promedio
#         out = out.mean(dim=1)
#         return self.classifier(out)


# ─── Demo rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import generate_dummy_dataset, load_dataset, train_test_split_dataset

    generate_dummy_dataset(n_per_class=30)
    X, y, labels = load_dataset("data", verbose=False)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)
    X_train, X_test = np.stack(X_train), np.stack(X_test)

    model = NeuralRecognizer(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        output_dim=len(labels),
        epochs=30,
    )
    model.fit(X_train, y_train, labels=labels)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"NN Accuracy en test: {acc:.2%}")
