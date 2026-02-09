# Climate-Adjusted Municipal Bond Risk Analyzer

## Business Context (The "Why")
Municipal bonds ("munis") are debt securities issued by local governments to fund public projects. Traditionally, credit ratings for these bonds rely on financial metrics (tax base, outstanding debt). However, climate change poses a material risk to the underlying assets and tax base—a town destroyed by wildfire cannot repay its debts.

**The Problem**: Standard credit models often lag in pricing physical climate risks.
**The Solution**: This tool ingests climate hazard data (Wildfire/Flood) and overlays it with municipal bond issuer locations to calculate a "Climate-Adjusted Risk Score" and estimate financial impact (Value-at-Risk).

## Technical Architecture
- **Data Layer**: DuckDB for efficient analytical querying.
- **Geospatial**: Rasterio (Raster analysis) & Geopandas (Vector analysis).
- **Backend**: FastAPI for serving risk metrics.
- **Frontend**: React (Planned) / Streamlit (Prototype) for visualization.

## Setup & Run

### 1. Backend (FastAPI)
Open a terminal in the project root:
```bash
# Install dependencies
pip install -r requirements.txt

# Run Data Generation (First time only)
python src/generate_data.py
python src/analysis.py

# Start API Server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (React)
Open a NEW terminal:
```bash
cd frontend
npm.cmd install  # or npm install if using Bash
npm.cmd run dev
```

## Features
- **Geospatial Risk Mapping**: Synthetic Wildfire Risk Raster (0.0 - 1.0) overlaid on CA.
- **Financial Modeling**: Value-at-Risk (VaR) estimation based on hazard exposure.
- **Interactive Dashboard**: React + Leaflet visualization of 100+ municipal bonds.

