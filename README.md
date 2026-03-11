# Climate-Adjusted Municipal Bond Risk Analyzer

## Executive Summary
This platform is an institutional-grade Geospatial ML suite designed to solve the mispricing of physical climate risk in the $4 trillion municipal bond market. By synthesizing real-time satellite imagery, daily weather vectors, and historical climate patterns, it quantifies the specific "Climate Spread" (yield penalty) per bond across a $1.2B portfolio.

---

## 🛠 Features & Architecture

### 1. Multi-Hazard Analytical Engine
- **Wildfire (RF Model)**: Trained a Scikit-Learn Random Forest on 524k+ NASA FIRMS records with 0.93 AUC.
- **Fire Path Prediction (PyTorch)**: Implemented a 1D CNN + LSTM temporal model trained on 7-day historical weather sequences to predict imminent spread probabilities.
- **Vegetation Health (NDVI)**: Ingests real-time Sentinel-2 L2A satellite imagery via Microsoft Planetary Computer (STAC/COG) to dynamically modulate fire risk based on current vegetation dryness.
- **FEMA Integration**: Spatial joins for county-level flood and earthquake risk using the FEMA National Risk Index.

### 2. Real-Time Alerting (WebSockets)
- A standalone FastAPI service that polls NASA FIRMS active fire detections every 30 seconds.
- Emits live "Red Alerts" to the dashboard when a fire is detected within 5km of an asset, complete with animated fire path trajectory highlights on the map.

### 3. Institutional Dashboard
- Professional, emoji-free interface with tooltips explaining financial metrics like Basis Points (bps) and Normalized Risk Scores.
- Interactive map showing asset-level risk layers and live alert streams.

---

## One-Click Execution

### 1. Full Production Stack (Recommended)
If you have Docker installed:
```bash
docker-compose up --build
```
- **Dashboard**: [http://localhost:80](http://localhost:80)
- **API Backend**: Port 8000
- **Alert Stream**: Port 5001

### 2. Data Science Pipeline (Automation)
To re-run the entire data fetch, NDVI processing, and ML training sequence:
```bash
python src/run_pipeline.py
```

---

## Project Structure
- `src/run_pipeline.py`: Master orchestration script.
- `src/train_dl_model.py`: PyTorch CNN+LSTM training.
- `src/alert_engine.py`: WebSocket live alert server.
- `src/analysis.py`: Portfolio scoring and yield-spread engine.
- `data/bonds_scored.csv`: Final risk-priced municipal portfolio.

---
Detailed technical breakdowns and roadmap logs can be found in the [brain/walkthrough.md](./brain/walkthrough.md) and [tasks.md](./brain/tasks.md) artifacts.
