# Multi-Author Change Detection

Proyecto para detectar cambios de autor en texto. Este repositorio contiene el código y los notebooks de las las dinstintas etapas en la investigación.La ***E1*** es un ***planteamiento inicial*** de como vamos a hacer nuestar aproximación a este problema. La ***E2*** cubre ***preprocesamiento, representaciones tradicionales y word embeddings*** (estáticos y contextuales). Las demas entregas se irán desglosando a la vez que se iran haciendo. 


## Estructura
- `data/raw/{easy,medium,hard}/{train,validation}/` — datos de entrada (no se suben a git).
- `data/processed/.../sentences.jsonl` — frases normalizadas que se generan en E2.
- `features/...` — TF-IDF, Word2Vec y embeddings contextuales (se generan).
- `notebooks/` — notebooks por entrega.
- `reports/` — tablas y figuras.

## Quickstart
1. Coloca los datos en `data/raw/{nivel}/{split}/`.
2. `pip install -r requirements.txt`
3. Ejecuta en este orden: `00_analisis_inicial` → `01_preprocesamiento` → `05_analisis_datos` → `02_repr_tradicionales` → `03_embeddings_estaticos` → `04_embeddings_contextuales` -> `06_analisis_final.ipynb`.

## Entregas
- **E1**: propuesta y plan experimental.
- **E2**: preprocesado + TF-IDF + Word2Vec + embeddings contextuales + EDA en processed.

## Dataset y referencias
En el siguiente enlace se encuentran los datos (raw) usados para este trabajo.
- Fuente oficial: [Style Change Detection — PAN @ CLEF 2025](https://pan.webis.de/clef25/pan25-web/style-change-detection.html)

Y en este otro enlace s epueden descragar los datos ya procesados y features para no tener que relanzar todos los notebooks. 
- Fuente : [Drive Processed Data - Erik Alex](https://drive.google.com/drive/folders/1_TDplYt0EUvZHNB-Q61mLUXOoVNDhByn?usp=sharing)