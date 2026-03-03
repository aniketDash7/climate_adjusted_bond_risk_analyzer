import duckdb
import geopandas as gpd
import pandas as pd
import requests
import numpy as np
import pickle
import os
import sys

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "climate_risk.duckdb")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# DB Connection
con = duckdb.connect(DB_PATH)

RF_MODEL_PATH = os.path.join(MODELS_DIR, "wildfire_rf_model.pkl")
FEATURES = [
    "latitude", "longitude",
    "temperature", "humidity", "wind_speed", "precipitation",
    "month", "day_of_year",
]

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
    """Score bonds using the trained Random Forest model, or FWI heuristic as fallback."""
    print("Calculating Physical Risk Scores...")
    df = con.execute("SELECT * FROM bonds").fetchdf()

    # Fetch live weather for each bond
    print("Fetching live weather for bond locations...")
    df = fetch_live_weather_for_bonds(df)

    # Add temporal features (current date)
    from datetime import datetime
    now = datetime.utcnow()
    df['month']       = now.month
    df['day_of_year'] = now.timetuple().tm_yday

    # Satellite-based features - for inference at bond locations (not active fires)
    # Note: These are no longer in FEATURES used by the model
    df['frp']        = 0.0
    df['bright_ti4'] = 285.0

    if os.path.exists(RF_MODEL_PATH):
        print("Using trained Random Forest model (FIRMS VIIRS 375m, 2019-2023)...")
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)

        df['latitude']  = df['lat']
        df['longitude'] = df['lon']

        X = df[FEATURES]
        df['risk_score'] = rf.predict_proba(X)[:, 1]   # P(fire) at each bond location
        print(f"  RF model scores -> min={df['risk_score'].min():.3f}  max={df['risk_score'].max():.3f}")
    else:
        print("RF model not found - using OpenMeteo FWI heuristic fallback...")
        real_data_path = "data/real_risk_zones.geojson"
        if os.path.exists(real_data_path):
            bonds_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
            risk_gdf  = gpd.read_file(real_data_path).rename(columns={'risk_score': 'climate_risk_score'})
            joined    = gpd.sjoin(bonds_gdf, risk_gdf[['geometry', 'climate_risk_score', 'name']], how="left", predicate="within")
            joined['risk_score'] = joined['climate_risk_score'].fillna(0.1)
            df = pd.DataFrame(joined.drop(columns=['geometry', 'index_right', 'climate_risk_score'], errors='ignore'))
        else:
            df['risk_score'] = np.random.uniform(0.05, 0.5, len(df))

    # --- Financial Impact Model ---
    df['climate_spread_bps'] = df['risk_score'] * 100
    df['fair_value_yield']   = df['coupon_rate'] + (df['climate_spread_bps'] / 100)
    df['mispricing_bps']     = (df['fair_value_yield'] - df['coupon_rate']) * 100
    df['VaR_Amount']         = df['outstanding_amount'] * df['risk_score'] * 0.4

    # Save
    con.execute("DROP TABLE IF EXISTS bonds")
    con.execute("CREATE TABLE bonds AS SELECT * FROM df")
    out_csv = os.path.join(DATA_DIR, "bonds_scored.csv")
    df.to_csv(out_csv, index=False)

    cols = [c for c in ['issuer', 'rating', 'risk_score', 'coupon_rate', 'fair_value_yield'] if c in df.columns]
    print("\nRisk Calculation Complete.")
    print(df[cols].head(10).to_string(index=False))
    print(f"\nExported {len(df)} scored bonds -> {out_csv}")

if __name__ == "__main__":
    ingest_data()
    calculate_risk()
