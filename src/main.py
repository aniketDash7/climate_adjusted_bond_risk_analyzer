from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import duckdb

app = FastAPI(title="Climate Bond Risk API")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to DB (Read Only)
con = duckdb.connect("data/climate_risk.duckdb", read_only=True)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "climate-bond-risk-api"}

@app.get("/api/bonds")
def get_bonds():
    """Returns all bonds with risk scores."""
    # We can read from the CSV for speed/simplicity or valid SQL
    try:
        df = pd.read_csv("data/bonds_scored.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stats")
def get_stats():
    """Returns aggregate portfolio risk metrics."""
    df = pd.read_csv("data/bonds_scored.csv")
    total_exposure = df['outstanding_amount'].sum()
    total_var = df['VaR_Amount'].sum()
    avg_risk = df['risk_score'].mean()
    
    return {
        "total_exposure": float(total_exposure),
        "total_var": float(total_var),
        "avg_risk_score": float(avg_risk),
        "high_risk_bonds": int(len(df[df['risk_score'] > 0.7]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
