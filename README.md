# Climate-Adjusted Municipal Bond Risk Analyzer

## Business Context (The "Why")
Municipal bonds ("munis") are debt securities issued by local governments to fund public projects. Traditionally, credit ratings for these bonds rely on financial metrics (tax base, outstanding debt). However, climate change poses a material risk to the underlying assets and tax base—a town destroyed by wildfire cannot repay its debts.

**The Problem**: Standard credit models often lag in pricing physical climate risks.
**The Solution**: This tool ingests climate hazard data (Wildfire/Flood) and overlays it with municipal bond issuer locations to calculate a "Climate-Adjusted Risk Score" and estimate financial impact (Value-at-Risk).

# Start API Server
uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
```

### 3. Data Pipeline (Optional - To Refresh Real Data)
```bash
python src/source_financial_data.py
python src/fetch_real_data.py
python src/analysis.py
```

## Features
- **NASA FIRMS Satellite Integration**: 524k+ real-world fire records from VIIRS 375m sensors (2019-2023).
- **Random Forest ML Pipeline**: Trained a spatial wildfire risk classifier on satellite-weather-terrain vectors.
- **Verified Metrics**: 0.93 ROC-AUC on Northern-to-Southern California spatial generalisation tests.
- **Financial Risk Modeling**: Climate Yield Spread and Fair Value pricing for $1.2B in municipal assets.
- **Interactive Dashboard**: High-fidelity React + FastAPI + Leaflet analytical suite.

---
Detailed documentation on the app's inception and technical architecture can be found in [origin_and_architecture.md](./origin_and_architecture.md).

