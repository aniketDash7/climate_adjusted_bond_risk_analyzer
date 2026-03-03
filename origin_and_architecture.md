# ICE Climate Bond Risk Analytics: Origin & Architecture

### **The Genesis**
This application was born out of a specific need: to demonstrate a master-level intersection between **Climate Science**, **Geospatial Engineering**, and **Financial Risk Modeling** for the *ICE Climate Data Scientist* role. The primary challenge was bridging the gap between "Raw Weather Data" (pixels) and "Credit Risk" (price). While many apps show a simple map of fires, this project was designed to answer the *financial* question: *"By how many basis points is this $100M municipal bond mispriced due to live atmospheric hazards?"*

---

### **The Workflow: From Atom to Analytics**

The application functions as a data pipeline that flows from raw external API ingestion to a premium interactive dashboard.

#### **1. Asset Generation (`src/source_financial_data.py`)**
The pipeline starts by defining the "exposure" layer. This script aggregates a list of the **Top 50 California Cities** with their real-world centroids. It assigns proxy financial identities—including **Credit Ratings** (sourced from S&P/Moody's historical averages) and approximated **Outstanding Debt** amounts. It transforms these into a `municipal_bonds.geojson` file, representing our starting portfolio of fixed-income assets.

#### **2. Real-Time Hazard Layer (`src/fetch_real_data.py`)**
This is the app's "nervous system." It reaches out to the **OpenMeteo API** and the **CodeforAmerica GeoJSON repository** to fetch real-world California county boundaries. For every county, the script performs a live weather query (Temperature, Humidity, Wind Speed) at the centroid. 
*   **The Brain**: It then applies a custom **Fire Weather Index (FWI)** proxy formula: 
    *   $FWI = (Temp \times 0.4) + (Wind \times 0.3) + (1 - Humid) \times 0.3$
The result is `firms_enriched.csv`, a large-scale supervised learning dataset.

#### **3. The Random Forest Pipeline (`src/train_model.py`)**
This is the machine learning backbone. It replaces simple heuristics with an **ML-native approach**:
*   **Feature Vectors**: Combines satellite firing radiosity, OpenMeteo weather history, and spatial seasonality for 500k+ fire records.
*   **Spatial Split**: Evaluates the model by training on Northern CA and testing on Southern CA to ensure generalisation.
*   **Result**: A robust model with **0.93 ROC-AUC**, saving a serialized `.pkl` for real-time inference.

#### **4. The Analytical Engine (`src/analysis.py`)**
Using the trained Random Forest, the engine:
*   Ingests the bonds into a unified SQL environment.
*   **ML Inference**: For each bond location, it constructs a feature vector (Live Weather + Seasonality) and feeds it into the classifier. 
*   **Financial Model**: It calculates the **Climate Yield Spread** using the probability of fire (P-fire). If a bond's location has a high ML-predicted risk, it adds a risk premium (+X bps) to the original coupon.

#### **4. The Data Bridge (`src/main.py`)**
A **FastAPI** backend acts as the gateway. It serves a RESTful API that delivers the final `bonds_scored.csv` to the frontend. It calculates real-time aggregate stats (Portfolio Exposure, Total VaR, Average Risk Spread) so the dashboard can provide a high-level executive summary.

#### **5. The Command Center (`frontend/src/App.jsx`)**
Finally, the **React-Leaflet** dashboard brings the data to life. It renders the bonds as interactive markers, color-coded by their live climate risk. The sidebar provides the "Portfolio Scorecard," while the custom popups allow an analyst to drill down into a specific issuer's rating, climate spread, and fair-value pricing.

---

### **Summary of Key Files**
| File | Role | Tech Stack |
| :--- | :--- | :--- |
| `src/source_financial_data.py` | Asset Layer | GeoPandas, Shapely |
| `src/fetch_real_data.py` | Hazard Layer | Requests, OpenMeteo API |
| `src/analysis.py` | Risk Modeling | DuckDB, GeoPandas (sjoin) |
| `src/main.py` | Backend API | FastAPI, Uvicorn |
| `frontend/src/App.jsx` | Visualization | React, Leaflet, Axios |
