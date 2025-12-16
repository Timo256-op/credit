from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import traceback
import logging
import os

app = FastAPI()

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# serve static files from ./static
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
scaler = None
try:
    model = joblib.load("credit_scoring_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("Model/scaler not loaded:", e)


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "model_loaded": model is not None}


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# Allow CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic wrapper to validate incoming JSON is an object
class AnyDict(BaseModel):
    __root__: Dict[str, Any]


@app.post("/predict")
def predict_default(payload: AnyDict):
    try:
        features = payload.__root__
        if not isinstance(features, dict):
            raise HTTPException(status_code=400, detail="payload must be a JSON object of feature:value pairs")
        data = pd.DataFrame([features])
        logger.info("Received predict request with %d features", len(features))

        # Determine expected feature names from scaler or model if available
        expected = None
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            expected = list(getattr(scaler, "feature_names_in_"))
        elif model is not None and hasattr(model, "feature_names_in_"):
            expected = list(getattr(model, "feature_names_in_"))

        if expected is not None:
            # build a single-row dict with expected columns, filling missing with 0
            base = {col: 0 for col in expected}
            # copy incoming values where available
            for col in data.columns:
                if col in expected:
                    # assign the first-row value from incoming data
                    base[col] = data.iloc[0].loc[col]
            data_to_scale = pd.DataFrame([base])[expected]
        else:
            # no metadata available; attempt best-effort: use incoming order
            data_to_scale = data

        # Impute any remaining NaNs before scaling/prediction
        if data_to_scale.isnull().any().any():
            # Prefer using scaler mean (proxied from training) if available
            if scaler is not None and hasattr(scaler, "mean_") and expected is not None:
                try:
                    fill_values = {col: float(scaler.mean_[i]) for i, col in enumerate(expected)}
                    data_to_scale = data_to_scale.fillna(value=fill_values)
                except Exception:
                    pass

            # Fallback per-column numeric median / categorical mode or 0
            if data_to_scale.isnull().any().any():
                # numeric columns
                num_cols = data_to_scale.select_dtypes(include=[np.number]).columns
                for c in num_cols:
                    if data_to_scale[c].isnull().any():
                        non_null = data_to_scale[c].dropna()
                        data_to_scale[c] = data_to_scale[c].fillna(non_null.median() if not non_null.empty else 0)

                # non-numeric columns
                obj_cols = data_to_scale.select_dtypes(exclude=[np.number]).columns
                for c in obj_cols:
                    if data_to_scale[c].isnull().any():
                        try:
                            mode = data_to_scale[c].mode()[0]
                            data_to_scale[c] = data_to_scale[c].fillna(mode)
                        except Exception:
                            data_to_scale[c] = data_to_scale[c].fillna("")

        if scaler is not None:
            data_scaled = scaler.transform(data_to_scale)
        else:
            data_scaled = data_to_scale.values
        if model is None:
            return {"error": "model not loaded"}
        prediction = int(model.predict(data_scaled)[0])
        # Compute probability if available; otherwise fallback to sigmoid(decision_function)
        probability = None
        try:
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(data_scaled)[0][1])
        except Exception:
            probability = None
        if probability is None:
            try:
                if hasattr(model, "decision_function"):
                    score = float(model.decision_function(data_scaled)[0])
                    probability = float(1.0 / (1.0 + np.exp(-score)))
            except Exception:
                probability = None

        return {
            "default_prediction": prediction,
            "default_probability": probability
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
