"""MLflow pyfunc wrapper around an AutoGluon ``TabularPredictor``.

MLflow has no AutoGluon flavor, so the predictor directory is logged as an
artifact and this class loads it back. It lives in the shared package so the
training notebook (Jupyter) and the Inference API import the same class.
"""

from typing import Any

import mlflow.pyfunc
import pandas as pd

PREDICTOR_ARTIFACT = "predictor"


class AutoGluonModel(mlflow.pyfunc.PythonModel):
    """Serves ``TabularPredictor.predict`` through the pyfunc interface."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from autogluon.tabular import TabularPredictor

        self.predictor = TabularPredictor.load(
            context.artifacts[PREDICTOR_ARTIFACT],
            require_py_version_match=False,
            check_packages=False,
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.Series:
        return self.predictor.predict(model_input)
