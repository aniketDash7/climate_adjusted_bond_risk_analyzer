import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point
import time
import json
import os

DATA_DIR = "data"
COUNTIES_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

def fetch_counties():
    print("Fetching CA Counties GeoJSON...")
    resp = requests.get(COUNTIES_URL)
    filepath = os.path.join(DATA_DIR, "ca_counties.geojson")
    with open(filepath, "wb") as f:
        f.write(resp.content)
    print("Counties downloaded.")
    return filepath

def fetch_weather_risk(gdf):
    print("Fetching Live Weather from OpenMeteo...")
    
    risk_data = []
    
    # Iterate over counties to get centroid weather
    for idx, row in gdf.iterrows():
        # Simple centroid
        centroid = row.geometry.centroid
        lat = centroid.y
        lon = centroid.x
        
        # OpenMeteo Free API
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        
        try:
            w_resp = requests.get(url).json()
            curr = w_resp.get('current', {})
            
            temp = curr.get('temperature_2m', 20)
            humid = curr.get('relative_humidity_2m', 50)
            wind = curr.get('wind_speed_10m', 10)
            
            # Simple "Fire Weather Index" Proxy
            # High Temp + High Wind + Low Humidity = High Risk
            # Normalize inputs roughly: T(0-40), H(0-100), W(0-50)
            
            # Risk Formula: (T/40) * (W/20) * (1 - H/100)
            # This is a heuristic for demo purposes
            t_score = min(temp / 40, 1.0)
            w_score = min(wind / 20, 1.0)
            h_score = 1.0 - (humid / 100)
            
            # Weighted combo
            fwi = (t_score * 0.4) + (w_score * 0.3) + (h_score * 0.3)
            fwi = min(max(fwi, 0), 1) # Clip 0-1
            
            risk_data.append({
                "name": row['name'],
                "temperature": temp,
                "humidity": humid,
                "wind_speed": wind,
                "risk_score": fwi
            })
            
            # Be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error for {row['name']}: {e}")
            risk_data.append({"name": row['name'], "risk_score": 0.2}) # Default low risk

    # Merge back
    risk_df = pd.DataFrame(risk_data)
    result = gdf.merge(risk_df, on='name')
    
    output_path = os.path.join(DATA_DIR, "real_risk_zones.geojson")
    result.to_file(output_path, driver="GeoJSON")
    print(f"Real Risk Data Saved to {output_path}")

if __name__ == "__main__":
    counties_path = fetch_counties()
    gdf = gpd.read_file(counties_path)
    fetch_weather_risk(gdf)
