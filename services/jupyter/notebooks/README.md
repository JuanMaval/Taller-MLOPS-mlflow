# Notebook de entrenamiento · Cubierta Forestal (S4 Jupyter)

Entrena **un solo tipo de modelo** (AutoGluon · LightGBM) con **25 configuraciones de
hiperparámetros** (25 experimentos) leyendo los datos **ya cargados** en PostgreSQL
(`training.covertype_features`) y deja los resultados **listos para MLflow**.

> No implementa MLflow, MinIO ni FastAPI: esos los integran otros servicios. Aquí se produce el
> bloque **S5 (PostgreSQL) → S4 (Jupyter)** y la salida preparada para **S2 (MLflow)**.

## Fuente de datos
`training.covertype_features` (snapshot versionado, con columna `split` ya decidida). El notebook
respeta ese split (test aislado). Orden de resolución de la fuente:
1. `training.covertype_features` (versión más reciente, o la fijada en `DATASET_VERSION`).
2. `processed.covertype` (respaldo dentro de la base).
3. `sklearn.datasets.fetch_covtype` (último recurso, para correr fuera del stack).

## Cómo ejecutarlo

### A) Dentro del stack (recomendado)
El servicio `jupyter` del `docker-compose` ya monta esta carpeta en `/workspace/notebooks` y trae
inyectadas las variables `DATA_DB_HOST=postgres`, `DATA_DB_NAME=cubierta_forestal`, etc., así que la
conexión funciona **sin configurar nada**:
```bash
docker compose up -d postgres jupyter
# abre Jupyter y ejecuta services/jupyter/notebooks/cubierta_forestal_experimentos.ipynb
```

### B) Local (VS Code)
```bash
pip install -r requirements.txt
```
Define las variables `DATA_DB_*` (o un `.env`) apuntando a tu PostgreSQL (si está en Docker,
expón el puerto 5432 y usa `DATA_DB_HOST=localhost`). Si no hay base, el notebook usa
`fetch_covtype` como respaldo. Para una prueba rápida, pon `QUICK_SMOKE = True` en la sección 14.

## Salidas (en `artifacts/`, junto al notebook)
```
artifacts/
├── configurations.json        # las 25 configuraciones
├── environment.json           # versiones
├── selected_model.json        # configuración ganadora + métricas de test
├── models/config_01 … 25/     # un modelo por configuración
├── results/results.csv|.parquet   # tabla consolidada (25 filas)
└── mlflow_payloads/config_01 … 25.json   # params+metrics+rutas, listos para MLflow
```
(Esta carpeta `artifacts/` está en `.gitignore`: son productos de ejecución, no van al repo.)

## Preparación para MLflow
Cada experimento deja un payload JSON con `params`, `metrics`, `tags` y rutas. La sección 20 del
notebook muestra el bucle `mlflow.start_run()/log_params/log_metrics/log_artifact(s)` que el
servicio de MLflow (S2) añadirá.
