# Climate Bond Risk Analyzer: Technical Working & Business Impact

## 1. Technical "Working" of the Project (End-to-End)

The project is built as a specialized **Geospatial ML Pipeline** for institutional risk management:

### **Phase 1: Multi-Source Data Ingestion**
-   **Satellite Fire Data**: Ingests 5+ years of **NASA FIRMS (VIIRS 375m)** active fire detections (524k+ records) to train core risk models.
-   **Vegetation Health (Sentinel-2)**: Queries the **Microsoft Planetary Computer** for real-time multispectral imagery (B04/B08) to calculate **NDVI**. This allows the model to "see" how dry the brush is surrounding a specific municipal bond asset.
-   **Multi-Hazard Layers**: Spatial joins for **FEMA Flood Risk** and **USGS Earthquake** datasets using the FEMA National Risk Index, providing a holistic climate profile.

### **Phase 2: Hybrid ML Architecture**
-   **Static Risk (Random Forest)**: A Scikit-Learn classifier (0.93 AUC) that evaluates geographic vulnerability based on weather, terrain, and vegetation.
-   **Temporal Prediction (CNN+LSTM)**: A PyTorch Deep Learning model trained on **real 7-day historical weather sequences** from the OpenMeteo Archive API. It predicts the "trajectory" and probability of fire path movement for imminent threats.

### **Phase 3: Real-Time Alerting & Portfolio Quant**
-   **WebSockets Engine**: A live polling service that checks NASA FIRMS every 30 seconds and broadcasts alerts to the dashboard for assets within a 5km radius.
-   **Financial Modeling**: The pipeline calculates the **Climate Yield Spread** (Risk Score × 100 bps) and **Value-at-Risk (VaR)**, translating raw climate physics into basis points for bond traders.

---

## 2. Business Impact (The "So What?")

### **A. Alpha Generation & Price Discovery**
Municipal bond markets are traditionally slow to price in climate externalities. By identifying bonds where the "ML-predicted risk" is significantly higher than the traditional yield reflects, an investor can avoid over-valuation or discover high-yield opportunities before the market adjusts.

### **B. Predictive Portfolio Protection**
Most firms measure risk *after* a disaster happens. This system measures **exposure probability** in real-time. A portfolio manager can see exactly which **$50M position** is most exposed to *today's* specific heatwave and low-humidity anomalies across California.

### **C. ESG Compliance & Reporting**
Regulatory requirements (like CA SB 253) now demand scientifically backed transparency. This tool provides a **defensible, satellite-validated quantitative proof** of climate exposure for institutional ESG auditing.

---

## 3. Production Readiness & Orchestration

The prototype has been hardened into an **Enterprise-Ready Stack**:
1.  **Orchestration**: A master pipeline script (`run_pipeline.py`) automates the entire flow from data ingestion to model retraining.
2.  **Containerization**: Fully Dockerized with `docker-compose`, providing one-click deployment for the Frontend, API Backend, and Alerting services.
3.  **Roadmap**: Future cloud scaling involves migrating local DuckDB files to a **Snowflake** or **PostgreSQL+PostGIS** warehouse and deploying the services via **AWS Fargate**.
