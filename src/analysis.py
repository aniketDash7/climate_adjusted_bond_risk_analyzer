import duckdb
import geopandas as gpd
import rasterio
import pandas as pd
import pandas as pd
import json

# DB Connection
con = duckdb.connect("data/climate_risk.duckdb")

def ingest_data():
    """Loads GeoJSON and Metadata into DuckDB."""
    print("Ingesting data into DuckDB...")
    
    # Load GeoJSON
    gdf = gpd.read_file("data/municipal_bonds.geojson")
    
    # Convert Geometry to WKT for SQL storage (DuckDB spatial extension setup is complex without internet, using WKT text for now)
    # Actually, we can just store lat/lon columns 
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    gdf['risk_score'] = 0.0 # Placeholder
    
    # Drop geometry for pure relational storage or keep as text
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    # Create Table
    con.execute("CREATE TABLE IF NOT EXISTS bonds AS SELECT * FROM df")
    # If table exists, replace (for dev iteration)
    con.execute("DROP TABLE IF EXISTS bonds")
    con.execute("CREATE TABLE bonds AS SELECT * FROM df")
    
    print(f"Ingested {len(df)} bonds.")

def calculate_risk():
    """Performs Spatial Join between Bonds (Points) and Real Risk Zones (Polygons)."""
    print("Calculating Physical Risk Scores (Real Data)...")
    
    # Read Bonds from DB
    df = con.execute("SELECT * FROM bonds").fetchdf()
    bonds_gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )
    
    # Read Real Risk Data (CA Counties with Live Weather)
    # Check if real data exists, else fallback
    import os
    real_data_path = "data/real_risk_zones.geojson"
    if not os.path.exists(real_data_path):
        print("Real data not found, using playback/mock mode.")
        # Logic to handle fallback if needed, but for now assuming it runs after fetch
        return 

    risk_gdf = gpd.read_file(real_data_path)
    
    # Spatial Join: Find which County (and thus Risk Score) each Bond is in
    # sjoin defaults to 'inner', so only bonds inside known counties will keep data
    # use 'left' to keep all bonds, filling NaN risk with default
    joined = gpd.sjoin(bonds_gdf, risk_gdf[['geometry', 'risk_score', 'name']], how="left", predicate="within")
    
    # Fill missing risk (e.g. bonds falling outside CA shapes)
    joined['risk_score'] = joined['risk_score'].fillna(0.1)
    
    # Update DataFrame (drop geometry to save back to SQL/CSV)
    df = pd.DataFrame(joined.drop(columns=['geometry', 'index_right']))
    
    print("Risk Calculation Complete.")
    
    # Financial Impact (Advanced Model)
    # 1. Climate Spread: How much EXTRA yield should they pay?
    # Logic: Risk Score (0-1) * 100 basis points (High risk = +1%)
    df['climate_spread_bps'] = df['risk_score'] * 100
    
    # 2. Adjusted Yield: Original Coupon + Risk Spread
    df['fair_value_yield'] = df['coupon_rate'] + (df['climate_spread_bps'] / 100)
    
    # 3. Mispricing: Is the bond OVERVALUED? (If Coupon < Fair Value Yield)
    df['mispricing_bps'] = (df['fair_value_yield'] - df['coupon_rate']) * 100
    
    # 4. VaR (Existing)
    df['VaR_Amount'] = df['outstanding_amount'] * df['risk_score'] * 0.4
    
    # Save final results to DB
    con.execute("DROP TABLE bonds")
    con.execute("CREATE TABLE bonds AS SELECT * FROM df")
    
    print(df[['issuer', 'rating', 'risk_score', 'coupon_rate', 'fair_value_yield']].head())
    
    # Export for Frontend
    df.to_csv("data/bonds_scored.csv", index=False)


if __name__ == "__main__":
    ingest_data()
    calculate_risk()
