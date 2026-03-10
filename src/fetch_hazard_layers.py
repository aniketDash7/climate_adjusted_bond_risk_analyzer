"""
Multi-Hazard Layer Fetcher
Sources:
  1. FEMA National Risk Index (NRI) -- County-level scores for Flood + Earthquake (0-100)
  2. USGS Earthquake Catalog (fallback) -- Historical M>=4.0 event counts within 50km
Output:
  data/hazard_layers.csv  (bond_id, flood_score, earthquake_score)
"""
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# 1. FEMA NRI (National Risk Index) -- County-level scores
# ============================================================

def fetch_fema_nri_ca():
    """
    Query FEMA NRI ArcGIS REST API for ALL CA counties.
    Returns dict: { county_name -> {flood_score, earthquake_score, wildfire_score} }
    Scores are 0-100 from NRI, normalized to 0-1.
    """
    print("Fetching FEMA NRI scores for CA counties...")

    url = (
        "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
        "National_Risk_Index_Counties/FeatureServer/0/query"
    )
    params = {
        "where": "STATEABBRV = 'CA'",
        "outFields": "COUNTY,IFLD_RISKS,CFLD_RISKS,ERQK_RISKS,WFIR_RISKS,RISK_SCORE",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 100,
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        data = resp.json()

        if "features" not in data or len(data["features"]) == 0:
            print("  WARNING: FEMA NRI returned no features.")
            return None

        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        print(f"  Retrieved NRI data for {len(gdf)} CA counties")

        # Combine inland + coastal flood scores (take max)
        gdf["IFLD_RISKS"] = pd.to_numeric(gdf["IFLD_RISKS"], errors="coerce").fillna(0)
        gdf["CFLD_RISKS"] = pd.to_numeric(gdf["CFLD_RISKS"], errors="coerce").fillna(0)
        gdf["flood_nri"]  = gdf[["IFLD_RISKS", "CFLD_RISKS"]].max(axis=1) / 100.0

        gdf["ERQK_RISKS"] = pd.to_numeric(gdf["ERQK_RISKS"], errors="coerce").fillna(0)
        gdf["earthquake_nri"] = gdf["ERQK_RISKS"] / 100.0

        gdf["WFIR_RISKS"] = pd.to_numeric(gdf["WFIR_RISKS"], errors="coerce").fillna(0)
        gdf["wildfire_nri"] = gdf["WFIR_RISKS"] / 100.0

        return gdf[["COUNTY", "geometry", "flood_nri", "earthquake_nri", "wildfire_nri"]]

    except Exception as e:
        print(f"  FEMA NRI query failed: {e}")
        return None


def assign_nri_scores(bonds_df, nri_gdf):
    """
    Spatial join: assign FEMA NRI flood + earthquake scores to each bond
    by determining which county polygon the bond falls within.
    """
    bonds_gdf = gpd.GeoDataFrame(
        bonds_df,
        geometry=gpd.points_from_xy(bonds_df.lon, bonds_df.lat),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(bonds_gdf, nri_gdf, how="left", predicate="within")

    # Fill NaN for bonds that didn't match a county
    joined["flood_nri"]      = joined["flood_nri"].fillna(0.20)
    joined["earthquake_nri"] = joined["earthquake_nri"].fillna(0.20)
    joined["wildfire_nri"]   = joined["wildfire_nri"].fillna(0.20)

    return joined


# ============================================================
# 2. USGS Earthquake (fallback if FEMA NRI fails)
# ============================================================

def fetch_earthquake_score(lat, lon, radius_km=50, min_mag=4.0):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    params = {
        "format": "text",
        "latitude": lat, "longitude": lon,
        "maxradiuskm": radius_km, "minmagnitude": min_mag,
        "starttime": "1994-01-01", "endtime": "2024-12-31",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        return int(resp.text.strip())
    except:
        return 0


def fetch_all_earthquake_scores(bonds_df):
    print("Fetching USGS earthquake counts (M>=4.0, 50km radius)...")
    counts = []
    for i, (_, row) in enumerate(bonds_df.iterrows()):
        count = fetch_earthquake_score(row['lat'], row['lon'])
        counts.append(count)
        if (i + 1) % 10 == 0:
            print(f"  Earthquake: {i+1}/{len(bonds_df)} bonds queried...")
        time.sleep(0.15)

    counts_arr = np.array(counts, dtype=float)
    log_counts = np.log1p(counts_arr)
    max_log = log_counts.max() if log_counts.max() > 0 else 1.0
    return (log_counts / max_log).tolist()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Hazard Layer Fetcher")
    print("=" * 60)

    # Load bonds
    bonds_path = os.path.join(DATA_DIR, "bonds_scored.csv")
    if not os.path.exists(bonds_path):
        bonds_path = os.path.join(DATA_DIR, "municipal_bonds.geojson")
        bonds_df = gpd.read_file(bonds_path)
        bonds_df['lon'] = bonds_df.geometry.x
        bonds_df['lat'] = bonds_df.geometry.y
        bonds_df = pd.DataFrame(bonds_df.drop(columns='geometry'))
    else:
        bonds_df = pd.read_csv(bonds_path)

    print(f"Loaded {len(bonds_df)} bonds\n")

    # --- FEMA NRI (Flood + Earthquake + Wildfire scores) ---
    nri_gdf = fetch_fema_nri_ca()

    if nri_gdf is not None and len(nri_gdf) > 0:
        joined = assign_nri_scores(bonds_df, nri_gdf)
        bonds_df["flood_score"]      = joined["flood_nri"].values
        bonds_df["earthquake_score"] = joined["earthquake_nri"].values
        bonds_df["wildfire_nri"]     = joined["wildfire_nri"].values
        bonds_df["fema_county"]      = joined["COUNTY"].values
        print("\n  FEMA NRI spatial join complete.")
    else:
        # Fallback: use USGS for earthquake, set flood to moderate default
        print("  Using USGS fallback for earthquake scores...")
        bonds_df["earthquake_score"] = fetch_all_earthquake_scores(bonds_df)
        bonds_df["flood_score"] = 0.30
        bonds_df["wildfire_nri"] = 0.50
        bonds_df["fema_county"] = "Unknown"

    # --- Save ---
    out_cols = ['bond_id', 'issuer', 'lat', 'lon', 'fema_county',
                'flood_score', 'earthquake_score', 'wildfire_nri']
    out_cols = [c for c in out_cols if c in bonds_df.columns]
    out_path = os.path.join(DATA_DIR, "hazard_layers.csv")
    bonds_df[out_cols].to_csv(out_path, index=False)

    print(f"\nSaved hazard layers -> {out_path}")
    print(bonds_df[['issuer', 'fema_county', 'flood_score', 'earthquake_score']].head(15).to_string(index=False))
    print("\nDone. Run src/analysis.py next to compute composite scores.")
