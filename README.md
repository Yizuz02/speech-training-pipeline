# speech-training-pipeline
Lightweight speech training pipeline that uses MFCCs and neural networks for recognizing and classifying 20 spoken commands.

---

## Estructura del proyecto

```
asr_system/
├── main.py              ← Punto de entrada: ejecuta el pipeline completo
├── preprocessing.py     ← Módulo 1: preprocesamiento de audio
├── feature_extraction.py← Módulo 2: extracción de MFCCs y features
├── dataset.py           ← Módulo 3: carga y división del dataset
├── model_hmm.py         ← Módulo 4A: reconocedor HMM/GMM
├── model_nn.py          ← Módulo 4B: red neuronal feedforward (PyTorch)
├── evaluation.py        ← Módulo 5: métricas, matrices de confusión
├── inference.py         ← Módulo 6: inferencia por archivo o micrófono
├── requirements.txt     ← Dependencias
└── data/                ← Dataset (crear manualmente o con script)
    ├── si/
    │   ├── si_001.wav
    │   └── ...
    ├── no/
    └── arriba/
```

---

## Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Preparar el dataset

**Opción A — Dataset sintético (para probar el pipeline)**
```python
from dataset import generate_dummy_dataset
generate_dummy_dataset(words=["si","no","arriba","abajo"], n_per_class=50)
```

**Opción B — Grabar tus propios audios**
Crea subcarpetas en `data/` con el nombre de cada palabra.
Coloca archivos `.wav` de ~1-2 segundos grabados a 16 kHz (mono).

**Opción C — Google Speech Commands (corpus libre)**
```bash
# Descarga una versión reducida
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C data/
```

---

## Ejecución

```bash
# Pipeline completo (entrena, evalúa, infiere)
python main.py

# Módulos individuales
python preprocessing.py    data/si/si_001.wav
python feature_extraction.py data/si/si_001.wav
python dataset.py

# Inferencia con archivo
python inference.py models/hmm.pkl data/no/no_001.wav

# Inferencia con micrófono
python inference.py models/nn_model.pt mic
python inference.py models/hmm_model.pkl mic
```

---

## Flujo del pipeline

```
Audio (.wav)
    ↓
[preprocessing.py]
  • Carga mono @ 16 kHz
  • Recorte de silencios
  • Filtro de Wiener (ruido)
  • Normalización
  • Pre-énfasis
    ↓
[feature_extraction.py]
  • MFCC (13 coef.)
  • Delta + Delta-delta
  • Vector de estadísticas (media + std)
    ↓
[model_hmm.py]          [model_nn.py]
  • GaussianHMM           • MLP (PyTorch)
  • 1 modelo/clase        • FC(128)→FC(64)→FC(n)
  • Baum-Welch (EM)       • Adam + Early stopping
    ↓                         ↓
[evaluation.py]
  • Accuracy, WER
  • Matriz de confusión
  • Comparativa
    ↓
[inference.py]
  • Archivo o micrófono
  • Palabra predicha
```

---

## Métricas

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | % de palabras correctamente clasificadas |
| **WER** | Word Error Rate: errores de sustitución, inserción, eliminación |
| **Matriz de confusión** | Qué palabras se confunden entre sí |
| **Precision / Recall / F1** | Métricas por clase |

---

## Referencias

- Rabiner, L. (1989). A tutorial on HMMs and selected applications in speech recognition. *Proc. IEEE, 77*(2), 257–286.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Trans. Acoust., 28*(4), 357–366.
- McFee, B. et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy*, 18–25.
- Google Speech Commands Dataset: https://arxiv.org/abs/1804.03209
