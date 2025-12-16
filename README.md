# Credit Scoring — Loan Default Prediction API

## Summary
- Cleaned dataset and trained a Logistic Regression model for predicting loan default risk.
- FastAPI application serving predictions via `/predict` endpoint with a modern web UI.
- Deployable to Render, AWS, or other cloud platforms without Docker.

## Files
- `app.py` — FastAPI application with CORS, logging, input validation, and `/health` endpoint.
- `static/index.html` — responsive web UI with organized input form.
- `requirements.txt` — pinned Python dependencies.
- `example_payload.json` — sample request payload for testing.
- `README.md` — this file with deployment instructions.
- `.gitignore` — excludes model files (stored in GitHub releases), data files, and virtual environment.

**Model files (stored in GitHub Releases, not in repo):**
- `credit_scoring_model.pkl`
- `scaler.pkl`

## Run Locally

### 1. Clone and set up
```powershell
git clone https://github.com/YOUR_USERNAME/credit-scoring.git
cd credit-scoring
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download model files from GitHub Releases
```powershell
# Download credit_scoring_model.pkl and scaler.pkl from the GitHub release
# Place them in the project root directory
```

### 3. Start the API
```powershell
# Development (with auto-reload)
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Production (no reload)
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Access the UI
Open http://127.0.0.1:8000 in your browser. Fill in the form and click "🚀 Predict Default".

## Deploy to Render

### 1. Push code to GitHub (without model files)
```powershell
git add .
git commit -m "Add FastAPI credit scoring app"
git push
```

### 2. Upload model files to GitHub Releases
1. Go to your GitHub repo → **Releases** → **Create a new release**
2. Tag: `v1.0.0` (or your version)
3. Upload `credit_scoring_model.pkl` and `scaler.pkl` as release assets
4. Publish the release

### 3. Update `app.py` to download models from GitHub Release
The app currently loads `.pkl` files from the current directory. On Render, you have two options:

**Option A (Simpler):** Update `app.py` to download models on startup:
```python
# Add to top of app.py after imports
import urllib.request
import os

RELEASE_URL = "https://github.com/YOUR_USERNAME/credit-scoring/releases/download/v1.0.0"

def download_model(filename):
    if not os.path.exists(filename):
        url = f"{RELEASE_URL}/{filename}"
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

download_model("credit_scoring_model.pkl")
download_model("scaler.pkl")
```

**Option B (Using environment variable):** Store models in a cloud bucket (S3) and configure the URL in environment variables.

### 4. Create a Render Web Service
1. Go to https://render.com → **New** → **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Name:** `credit-scoring-api`
   - **Environment:** `Python 3.10`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port 8000`
   - **Free tier** available (with limitations)
4. Click **Create Web Service**

### 5. Monitor and test
- Once deployed, your app will be at: `https://credit-scoring-api.onrender.com`
- Test health: `https://credit-scoring-api.onrender.com/health`
- Visit UI: `https://credit-scoring-api.onrender.com`
- POST predictions: `https://credit-scoring-api.onrender.com/predict`

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Make a prediction on loan default risk.

**Request (JSON):**
```json
{
  "Amount": 2500,
  "TransactionCount": 3,
  "Value": 2600
}
```

**Response:**
```json
{
  "default_prediction": 0,
  "default_probability": 0.2936043818025309
}
```
- `default_prediction`: 0 = low risk, 1 = high risk
- `default_probability`: probability of default (0–1)
