import duckdb
import geopandas as gpd
import pandas as pd
import requests
import numpy as np
import pickle
import os
import sys
import time

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "climate_risk.duckdb")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# DB Connection
con = duckdb.connect(DB_PATH)

RF_MODEL_PATH = os.path.join(MODELS_DIR, "wildfire_rf_model.pkl")
HAZARD_CSV    = os.path.join(DATA_DIR, "hazard_layers.csv")

FEATURES = [
    "latitude", "longitude",
    "temperature", "humidity", "wind_speed", "precipitation",
    "month", "day_of_year",
]

# Composite weights
W_FIRE  = 0.50
W_FLOOD = 0.30
W_QUAKE = 0.20

def ingest_data():
    """Loads GeoJSON and Metadata into DuckDB."""
    print("Ingesting data into DuckDB...")
    bound_path = os.path.join(DATA_DIR, "municipal_bonds.geojson")
    gdf = gpd.read_file(bound_path)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    con.execute("DROP TABLE IF EXISTS bonds")
    con.execute("CREATE TABLE bonds AS SELECT * FROM df")
    print(f"Ingested {len(df)} bonds.")

def fetch_live_weather_for_bonds(df):
    """Fetch live weather for each bond's location via OpenMeteo."""
    temps, winds, humids, precips = [], [], [], []
    for _, row in df.iterrows():
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={row['lat']}&longitude={row['lon']}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        )
        try:
            r = requests.get(url, timeout=10).json()
            c = r.get('current', {})
            temps.append(c.get('temperature_2m', 20))
            winds.append(c.get('wind_speed_10m', 10))
            humids.append(c.get('relative_humidity_2m', 50))
            precips.append(c.get('precipitation', 0))
        except:
            temps.append(20); winds.append(10); humids.append(50); precips.append(0)
        time.sleep(0.05)
    df = df.copy()
    df['temperature']   = temps
    df['wind_speed']    = winds
    df['humidity']      = humids
    df['precipitation'] = precips
    return df

def calculate_risk():
    """Score bonds using RF wildfire model + FEMA flood/earthquake -> composite."""
    print("Calculating Multi-Hazard Risk Scores...")
    df = con.execute("SELECT * FROM bonds").fetchdf()

    # --- 1. Wildfire (RF Model) ---
    print("Fetching live weather for bond locations...")
    df = fetch_live_weather_for_bonds(df)

    from datetime import datetime
    now = datetime.utcnow()
    df['month']       = now.month
    df['day_of_year'] = now.timetuple().tm_yday

    if os.path.exists(RF_MODEL_PATH):
        print("Using trained Random Forest model for wildfire scoring...")
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
        df['latitude']  = df['lat']
        df['longitude'] = df['lon']
        X = df[FEATURES]
        df['wildfire_score'] = rf.predict_proba(X)[:, 1]
        print(f"  RF wildfire scores -> min={df['wildfire_score'].min():.3f}  max={df['wildfire_score'].max():.3f}")
    else:
        print("RF model not found - using default wildfire score...")
        df['wildfire_score'] = 0.30

    # --- 2. Flood + Earthquake (from FEMA NRI hazard layers) ---
    if os.path.exists(HAZARD_CSV):
        print(f"Loading hazard layers from {HAZARD_CSV}...")
        hazard_df = pd.read_csv(HAZARD_CSV)
        # Merge on bond_id
        merge_cols = ['bond_id', 'flood_score', 'earthquake_score']
        merge_cols = [c for c in merge_cols if c in hazard_df.columns]
        df = df.merge(hazard_df[merge_cols], on='bond_id', how='left', suffixes=('', '_hazard'))
        df['flood_score']      = df['flood_score'].fillna(0.30)
        df['earthquake_score'] = df['earthquake_score'].fillna(0.30)
        print(f"  Flood scores   -> min={df['flood_score'].min():.3f}  max={df['flood_score'].max():.3f}")
        print(f"  Earthquake     -> min={df['earthquake_score'].min():.3f}  max={df['earthquake_score'].max():.3f}")
    else:
        print("WARNING: hazard_layers.csv not found. Run fetch_hazard_layers.py first.")
        df['flood_score']      = 0.30
        df['earthquake_score'] = 0.30

    # --- 3. Weighted Composite Score ---
    df['composite_score'] = (
        W_FIRE  * df['wildfire_score'] +
        W_FLOOD * df['flood_score'] +
        W_QUAKE * df['earthquake_score']
    )
    df['risk_score'] = df['composite_score']   # backward compat

    print(f"\n  Composite score -> min={df['composite_score'].min():.3f}  max={df['composite_score'].max():.3f}")
    print(f"  Weights: Fire={W_FIRE}, Flood={W_FLOOD}, Earthquake={W_QUAKE}")

    # --- Financial Impact Model ---
    df['climate_spread_bps'] = df['composite_score'] * 100
    df['fair_value_yield']   = df['coupon_rate'] + (df['climate_spread_bps'] / 100)
    df['mispricing_bps']     = (df['fair_value_yield'] - df['coupon_rate']) * 100
    df['VaR_Amount']         = df['outstanding_amount'] * df['composite_score'] * 0.4

    # Save
    con.execute("DROP TABLE IF EXISTS bonds")
    con.execute("CREATE TABLE bonds AS SELECT * FROM df")
    out_csv = os.path.join(DATA_DIR, "bonds_scored.csv")
    df.to_csv(out_csv, index=False)

    cols = ['issuer', 'wildfire_score', 'flood_score', 'earthquake_score', 'composite_score']
    cols = [c for c in cols if c in df.columns]
    print("\nMulti-Hazard Risk Calculation Complete.")
    print(df[cols].head(10).to_string(index=False))
    print(f"\nExported {len(df)} scored bonds -> {out_csv}")

if __name__ == "__main__":
    ingest_data()
    calculate_risk()
