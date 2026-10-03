# s2 · MLflow tracking server + model registry.
#
# The official image ships MLflow only. Add the two client libraries the
# server needs to talk to the backend store (Postgres) and the artifact
# store (MinIO through the S3 API).
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.10 \
    boto3==1.40.49

EXPOSE 5000
