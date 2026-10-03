from fastapi import FastAPI

from inference_api import __version__

app = FastAPI(
    title="Cubierta Forestal · Inference API",
    version=__version__,
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by the container healthcheck."""
    return {"status": "ok"}
