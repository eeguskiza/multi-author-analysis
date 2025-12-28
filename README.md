# Multi-Author Change Detection

Proyecto para detectar cambios de autor en texto corto (PAN @ CLEF 2025). El repo contiene los notebooks y scripts de cada entrega:

- **E1** — planteamiento y plan experimental.
- **E2** — pipeline de preprocesado, TF‑IDF, Word2Vec y DistilBERT.
- **E3** — construcción del dataset de fronteras y baselines shallow (TF‑IDF y Word2Vec).
- **E4** — prototipos neuronales ligeros (MLP, LSTM, CNN) y comparación por nivel.
- **E5** — modelos de lenguaje grandes: In-Context Learning (ICL) con Qwen3 y fine-tuning (full FT + QLoRA).

Cada notebook sigue la convención `XX_nombre.ipynb` y todo el cómputo pesado se delega en `scripts/`.

## Estructura
- `data/raw/{easy,medium,hard}/{train,validation}/` — corpus original (no versionado).
- `data/processed/` — frases normalizadas, boundaries, deltas, secuencias.
- `features/` — TF-IDF (`tfidf/`), Word2Vec (`embeddings_static/`) y DistilBERT (`embeddings_contextual/`).
- `checkpoints/` — modelos entrenados (fine-tuning con full FT y QLoRA).
- `scripts/` — scripts CLI (`08_*`, `09/*`, etc.) para precomputar y entrenar sin bloquear los notebooks.
- `notebooks/` — análisis paso a paso (ver `notebooks/README.md` para el detalle).
- `reports/` — figuras, tablas y métricas JSON.

## Quickstart

### Requisitos previos
- Python 3.9+ con conda/virtualenv
- GPU con CUDA (opcional para notebooks 01-10, **requerida** para notebooks 11-12)
- VRAM recomendada: mínimo 8GB, ideal 16GB+ para modelos grandes

### Instalación
1. Descargar los datos RAW a `data/raw/{nivel}/{split}/`.
2. Crear el entorno e instalar dependencias:
   ```bash
   conda create -n multi-author python=3.10
   conda activate multi-author
   pip install -r requirements.txt
   ```

### Ejecución de notebooks

**E1-E2: Preprocesado y features**
1. `00_analisis_inicial` → `01_preprocesamiento` → `02_analisis_datos`
2. `03_repr_tradicionales` → `04_embeddings_estaticos` → `05_embeddings_contextuales`
3. `06_analisis_final`

**E3-E4: Modelos tradicionales y shallow**
4. `07_boundaries_y_baselines` → `08_tfidf_ventanas_y_shallow`
5. `09_word2vec_lstm_cnn` → `10_analisis_modelos_por_nivel`

**E5: Modelos de lenguaje grandes (requiere GPU)**
6. `11_icl_qwen3_scaling` — In-Context Learning con Qwen3 (0.6B a 8B)
7. `12_finetuning_transformers` — Fine-tuning con DistilBERT, Qwen3 (full FT + QLoRA)

**Nota importante:** Los notebooks 08-10 requieren ejecutar scripts previos:
```bash
python scripts/08_compute_sw.py
python scripts/08_train_shallow.py
python scripts/09/09_compute_deltas.py
python scripts/09/09_train_models.py
```

Los notebooks 11-12 se ejecutan directamente (el entrenamiento está integrado en el notebook).

## Dataset y enlaces
- Fuente oficial RAW: [Style Change Detection — PAN @ CLEF 2025](https://pan.webis.de/clef25/pan25-web/style-change-detection.html)
- Dump procesado (frases, features y deltas) para evitar recomputar todo: [Drive Processed Data - Erik Alex](https://drive.google.com/drive/folders/1_TDplYt0EUvZHNB-Q61mLUXOoVNDhByn?usp=sharing)

## Resultados y outputs

### Métricas guardadas
- **Modelos tradicionales (E3-E4):** `reports/08_*.json`, `reports/09_metrics.json`, `reports/10_*.json`
- **In-Context Learning (E5):** `reports/11_icl_metrics_final.json` (desglose por modelo, shots y nivel)
- **Fine-tuning (E5):** `reports/12_finetuning_metrics_final.json` (agregado + por nivel)

### Modelos entrenados
- **Checkpoints fine-tuning:** `checkpoints/finetuning/{model_key}/best_model/`
  - `distilbert` (66M params, full FT)
  - `qwen3-0.6b` (600M params, full FT)
  - `qwen3-1.7b` (1.7B params, QLoRA 4-bit)
  - `qwen3-4b` (4B params, QLoRA 4-bit)

### Visualizaciones
- **Análisis general:** `reports/02_*.png`, `reports/06_*.png`, `reports/10_*.png`
- **ICL scaling:** `reports/11_scaling_*.png`
- **Fine-tuning:** `reports/12_f1_*.png`, `reports/12_disagreement_heatmap_{level}.png`

## Referencias adicionales
- `requirements.txt` incluye las versiones usadas en los notebooks.
- `notebooks/README.md` describe cada cuaderno (00–12) con el objetivo y salidas detalladas.
- Para reproducir resultados por nivel (notebook 10) asegúrate de tener `data/processed/predictions/*.npz`, generados al final del script `09/09_train_models.py`.
- **Demo interactiva (notebook 12, sección 13):** Permite cargar los checkpoints y probar predicciones en ejemplos personalizados, con análisis de desacuerdos entre modelos por nivel de dificultad.
