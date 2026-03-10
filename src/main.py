from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import duckdb
import os

app = FastAPI(title="Climate Bond Risk API")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "climate_risk.duckdb")
CSV_PATH = os.path.join(BASE_DIR, "data", "bonds_scored.csv")

# Connect to DB (Read Only)
con = duckdb.connect(DB_PATH, read_only=True)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "climate-bond-risk-api"}

@app.get("/api/bonds")
def get_bonds():
    """Returns all bonds with multi-hazard risk scores."""
    try:
        df = pd.read_csv(CSV_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stats")
def get_stats():
    """Returns aggregate portfolio risk metrics."""
    df = pd.read_csv(CSV_PATH)
    total_exposure = df['outstanding_amount'].sum()
    total_var = df['VaR_Amount'].sum()

    return {
        "total_exposure":   float(total_exposure),
        "total_var":        float(total_var),
        "avg_risk_score":   float(df['composite_score'].mean()) if 'composite_score' in df.columns else float(df['risk_score'].mean()),
        "high_risk_bonds":  int(len(df[df.get('composite_score', df.get('risk_score', 0)) > 0.7])),
        "avg_spread_bps":   float(df['climate_spread_bps'].mean()) if 'climate_spread_bps' in df.columns else 0.0,
        "num_bonds":        int(len(df)),
        "avg_wildfire":     float(df['wildfire_score'].mean()) if 'wildfire_score' in df.columns else 0.0,
        "avg_flood":        float(df['flood_score'].mean()) if 'flood_score' in df.columns else 0.0,
        "avg_earthquake":   float(df['earthquake_score'].mean()) if 'earthquake_score' in df.columns else 0.0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
