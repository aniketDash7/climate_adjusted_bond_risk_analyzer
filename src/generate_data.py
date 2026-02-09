import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
import os
import random

# Constants for California-ish bounds
MIN_LON, MAX_LON = -124.4, -114.1
MIN_LAT, MAX_LAT = 32.5, 42.0
RES = 0.1  # Degree resolution (approx 10km)

DATA_DIR = "data"
risk_raster_path = os.path.join(DATA_DIR, "wildfire_risk.tif")
bonds_path = os.path.join(DATA_DIR, "municipal_bonds.geojson")

def generate_risk_raster():
    """Generates a synthetic wildfire risk raster (0.0 to 1.0)."""
    print(f"Generating Risk Raster at {risk_raster_path}...")
    
    width = int((MAX_LON - MIN_LON) / RES)
    height = int((MAX_LAT - MIN_LAT) / RES)
    
    # Generate synthetic risk using simple Gaussian-like blobs
    transform = from_origin(MIN_LON, MAX_LAT, RES, RES)
    
    # Create a base of low risk
    data = np.random.normal(0.2, 0.1, (height, width))
    
    # Add "Hotspots" (High risk zones)
    for _ in range(5):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(5, 15)
        
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        mask = x*x + y*y <= radius*radius
        data[mask] += np.random.uniform(0.4, 0.8)

    # Normalize to 0-1
    data = np.clip(data, 0, 1)
    
    with rasterio.open(
        risk_raster_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    
    print("Risk Raster Generated.")

def generate_bonds():
    """Generates mock municipal bond data."""
    print(f"Generating Bond Data at {bonds_path}...")
    
    num_bonds = 100
    bonds = []
    
    issuers = ["City of San Francisco", "Los Angeles County", "Sacramento USD", "San Diego Water District", "Napa Valley Fire Dept"]
    
    for i in range(num_bonds):
        lon = random.uniform(MIN_LON, MAX_LON)
        lat = random.uniform(MIN_LAT, MAX_LAT)
        
        bond = {
            "bond_id": f"CUSIP-{random.randint(10000, 99999)}",
            "issuer": random.choice(issuers),
            "coupon_rate": round(random.uniform(1.5, 5.0), 2),
            "maturity_year": random.choice([2030, 2035, 2040, 2050]),
            "outstanding_amount": random.randint(1, 100) * 1_000_000,  # 1M to 100M
            "geometry": Point(lon, lat)
        }
        bonds.append(bond)
        
    gdf = gpd.GeoDataFrame(bonds, crs="EPSG:4326")
    gdf.to_file(bonds_path, driver="GeoJSON")
    print("Bond Data Generated.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_risk_raster()
    generate_bonds()
    print("Data Generation Complete.")
