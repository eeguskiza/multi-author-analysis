# Multi-Author Change Detection

Proyecto para detectar cambios de autor en texto corto (PAN @ CLEF 2025). El repo contiene los notebooks y scripts de cada entrega:

- **E1** — planteamiento y plan experimental.
- **E2** — pipeline de preprocesado, TF‑IDF, Word2Vec y DistilBERT.
- **E3** — construcción del dataset de fronteras y baselines shallow (TF‑IDF y Word2Vec).
- **E4** (en progreso) — prototipos neuronales ligeros y comparación por nivel.

Cada notebook sigue la convención `XX_nombre.ipynb` y todo el cómputo pesado se delega en `scripts/`.

## Estructura
- `data/raw/{easy,medium,hard}/{train,validation}/` — corpus original (no versionado).
- `data/processed/` — frases normalizadas, boundaries, deltas, secuencias.
- `features/` — TF-IDF (`tfidf/`), Word2Vec (`embeddings_static/`) y DistilBERT (`embeddings_contextual/`).
- `scripts/` — scripts CLI (`08_*`, `09/*`, etc.) para precomputar y entrenar sin bloquear los notebooks.
- `notebooks/` — análisis paso a paso (ver `notebooks/README.md` para el detalle).
- `reports/` — figuras, tablas y métricas JSON.

## Quickstart
1. Descargar los datos RAW a `data/raw/{nivel}/{split}/`.
2. Crear el entorno (por ejemplo con conda) e instalar dependencias: `pip install -r requirements.txt`.
3. Ejecutar los notebooks en orden:
   1. `00_analisis_inicial` → `01_preprocesamiento`
   2. `02_analisis_datos` → `03_repr_tradicionales` → `04_embeddings_estaticos` → `05_embeddings_contextuales`
   3. `06_analisis_final` para cerrar la E2.
   4. `07_boundaries_y_baselines` → `08_tfidf_ventanas_y_shallow`
   5. `09_word2vec_lstm_cnn` → `10_analisis_modelos_por_nivel`
4. Para los cuadernos 08–10, lanzar antes los scripts correspondientes:
   ```bash
   conda run -n work python scripts/08_compute_sw.py
   conda run -n work python scripts/08_train_shallow.py
   conda run -n work python scripts/09/09_compute_deltas.py
   conda run -n work python scripts/09/09_train_models.py
   ```
   Estos scripts generan los `.parquet`, `.npz` y JSON que consumen los notebooks.

## Dataset y enlaces
- Fuente oficial RAW: [Style Change Detection — PAN @ CLEF 2025](https://pan.webis.de/clef25/pan25-web/style-change-detection.html)
- Dump procesado (frases, features y deltas) para evitar recomputar todo: [Drive Processed Data - Erik Alex](https://drive.google.com/drive/folders/1_TDplYt0EUvZHNB-Q61mLUXOoVNDhByn?usp=sharing)

## Referencias adicionales
- `requirements.txt` incluye las versiones usadas en los notebooks.
- `notebooks/README.md` describe cada cuaderno (00–10) con el objetivo y salidas.
- Para reproducir resultados por nivel (notebook 10) asegúrate de tener `data/processed/predictions/*.npz`, generados al final del script `09/09_train_models.py`.
