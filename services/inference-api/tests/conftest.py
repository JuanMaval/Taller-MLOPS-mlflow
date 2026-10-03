import os

# Tests never talk to a tracking server: the app starts without a model and
# each test injects what it needs through FastAPI dependency overrides.
os.environ["LOAD_MODEL_ON_STARTUP"] = "false"
