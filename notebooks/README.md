# Notebooks — Análisis Multi-Autor

## Entregas 1-3: Pipeline base y modelos tradicionales

**00_analisis_inicial_raw.ipynb.** Audita el corpus bruto por nivel y split para asegurar que los datos están completos y en el formato esperado. Recorre `data/raw/{easy,medium,hard}/{train,validation}`, cuenta documentos, estima frases y tokens de forma heurística y detecta vacíos o duplicados. Genera tablas de control y algunas figuras simples en `reports/` para respaldar decisiones del preprocesado.

**01_preprocesamiento.ipynb.** Convierte el RAW en frases normalizadas y reutilizables. Segmenta por signos de fin de oración con excepciones inglesas, normaliza (minúsculas, sustitución de URL, emails y números) y tokeniza; opcionalmente permite stemming (desactivado por defecto). La salida es un `sentences.jsonl` por nivel y split en `data/processed/...` con los campos `doc_id, sent_id, level, split, text_norm, n_tokens, is_boundary`.

**02_analisis_datos.ipynb.** Analiza el conjunto ya procesado para validar que el pipeline ha sido consistente y para caracterizar el corpus. Calcula distribuciones por nivel y split (tokens por frase, frases por documento), detecta outliers, y produce gráficos de apoyo. Deja las tablas agregadas y las figuras en `reports/` para que puedan consultarse sin ejecutar todo el flujo.

**03_representaciones_tradicionales.ipynb.** Construye representaciones TF-IDF a partir de las frases normalizadas. Ajusta los vectorizadores con `train` y transforma `train` y `validation` sin fugas de información. Incluye una vista de términos destacados y guarda matrices dispersas y metadatos en `features/tfidf/{word,char}`, junto con índices que permiten mapear cada fila a `(level, split, doc_id, sent_id)`.

**04_embeddings_estaticos.ipynb.** Entrena Word2Vec sobre `train` y proyecta cada frase como la media de sus vectores de palabra. Informa del tamaño del vocabulario y de la cobertura OOV para comprobar que el modelo captura la mayor parte del léxico. Guarda el modelo (`.kv`), el vocabulario y las matrices de embeddings de frase por split en `features/embeddings_static/`, además de un resumen con los parámetros y coberturas.

**05_embeddings_contextuales.ipynb.** Extrae embeddings por frase con `distilbert-base-uncased` usando mean pooling y `attention_mask`. Selecciona dispositivo automáticamente (CUDA/MPS/CPU) y registra la fracción de frases truncadas a `MAX_LEN`, junto con percentiles de longitud subpalabra para justificar el corte. Produce `C_train.npy` y `C_validation.npy` con sus índices y un `contextual_resumen.json` en `features/embeddings_contextual/`.

**06_analisis_final.ipynb.** Integra y valida todo lo generado en la E2. Carga resúmenes de TF-IDF, Word2Vec y embeddings contextuales, y cruza shapes e índices con `data/processed` para comprobar consistencia por nivel y split. Calcula métricas globales (densidad TF-IDF, tamaños y coberturas de embeddings, ratio de truncado y percentiles de longitud subpalabra) y produce 4–5 gráficos globales. Escribe un `06_resumen_global.md` y las figuras en `reports/` como cierre de la entrega.

**07_boundaries_y_baselines.ipynb.** Construye el dataset a nivel de frontera (entre frase y frase) combinando los índices de la E2 con las etiquetas `is_boundary`. Analiza la distribución de positivos por split y nivel, introduce las métricas de segmentación (Pk/WindowDiff) y define baselines sencillos (`never_change`, `random_p`). Deja el dataset en `data/processed/boundaries/` y un PDF resumen en `reports/07_boundaries_y_baselines.pdf`.

**08_tfidf_ventanas_y_shallow.ipynb.** Usa las matrices TF-IDF de caracteres para comparar dos enfoques: (1) ventanas deslizantes con similitud coseno y umbral, (2) modelos shallow (Regresión Logística y Linear SVM) sobre deltas TF-IDF. Todo el cómputo pesado vive en `scripts/08_*`, y el notebook solo orquesta ejecuciones, lee métricas (`reports/08_*`) y arma tablas/figuras para el informe.

**09_word2vec_lstm_cnn.ipynb.** Extiende la experimentación a embeddings densos: deltas con Word2Vec + modelos lineales y prototipos neuronales ligeros (MLP, LSTM, CNN) sobre deltas y secuencias DistilBERT. Igual que en el 08, delega el preprocesado y entrenamiento a `scripts/09/`, y el notebook se limita a revisar shapes, explicar los pasos, cargar `reports/09_metrics.json` y comparar con los baselines previos.

**10_analisis_modelos_por_nivel.ipynb.** Reúne todas las predicciones guardadas y recalcula métricas por nivel (`easy`, `medium`, `hard`) para cada modelo. Genera tablas resúmenes (F1, Pk, WindowDiff) y barplots comparativos guardados en `reports/10_*`. Sirve como panel de control para el informe: resalta qué modelos funcionan mejor por nivel y en global, incluyendo los baselines y los prototipos contextuales.

## Entregas 4-5: Modelos de lenguaje grandes

**11_icl_qwen3_scaling.ipynb.** Explora In-Context Learning (ICL) con la familia Qwen3 (0.6B, 1.7B, 4B, 8B) sin fine-tuning. Implementa prompting few-shot (0 a 10 ejemplos) para detectar cambios de autor. Analiza el scaling law: cómo varían las métricas (F1, accuracy) según el tamaño del modelo y el número de shots. Usa cuantización 4-bit (BitsAndBytes) para ejecutar modelos grandes en GPU con VRAM limitada. Guarda métricas en `reports/11_icl_metrics_final.json` con desglose por nivel (easy/medium/hard), número de shots y tamaño de modelo.

**12_finetuning_transformers.ipynb.** Fine-tuning end-to-end de modelos transformer para clasificación binaria (mismo autor vs cambio de autor). Compara dos técnicas:
- **Full fine-tuning** (DistilBERT-66M, Qwen3-0.6B): Entrena todos los parámetros.
- **QLoRA** (Qwen3-1.7B, Qwen3-4B): Parameter-efficient fine-tuning con cuantización 4-bit + adaptadores LoRA de bajo rango.

Incluye:
- Métricas agregadas y por nivel de dificultad guardadas en `reports/12_finetuning_metrics_final.json`.
- Checkpoints de modelos entrenados en `checkpoints/finetuning/{model_key}/best_model/`.
- **Sección 13: Demo interactiva** que carga los checkpoints y permite:
  - Probar predicciones en ejemplos del dataset (desglosados por nivel easy/medium/hard).
  - Entrada manual de pares de oraciones personalizados.
  - Análisis de casos de desacuerdo entre modelos (por nivel).
  - Visualizaciones: heatmaps de tasas de desacuerdo entre modelos guardadas en `reports/12_disagreement_heatmap_{level}.png`.

## Orden de ejecución recomendado

**Preprocesado y features (E1-E2):**
1. `00_analisis_inicial` → `01_preprocesamiento` → `02_analisis_datos`
2. `03_repr_tradicionales` → `04_embeddings_estaticos` → `05_embeddings_contextuales`
3. `06_analisis_final`

**Modelos tradicionales y shallow (E3):**
4. `07_boundaries_y_baselines` → `08_tfidf_ventanas_y_shallow` → `09_word2vec_lstm_cnn`
5. `10_analisis_modelos_por_nivel`

**Modelos de lenguaje grandes (E4-E5):**
6. `11_icl_qwen3_scaling` (In-Context Learning, sin entrenamiento)
7. `12_finetuning_transformers` (Fine-tuning con Full FT y QLoRA)

**Nota:** Los notebooks 08-09 requieren ejecutar scripts de `scripts/` previamente. Los notebooks 11-12 requieren GPU con VRAM suficiente (mínimo 8GB, recomendado 16GB+ para modelos grandes).
