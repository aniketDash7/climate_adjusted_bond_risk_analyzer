"""
Sentinel-2 NDVI Fetcher
Source: Microsoft Planetary Computer (free, no API key)
Collection: sentinel-2-l2a (Level 2A, 10m resolution)
Formula: NDVI = (B08 - B04) / (B08 + B04)

Output: data/ndvi_scores.csv  (bond_id, ndvi, fuel_class)
"""
import pandas as pd
import numpy as np
import os
import time

from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Buffer around bond point (~500m at CA latitudes)
BUFFER_DEG = 0.005


def classify_ndvi(ndvi):
    if ndvi < 0.15:
        return "Barren"
    elif ndvi < 0.25:
        return "Very Dry"
    elif ndvi < 0.40:
        return "Dry"
    elif ndvi < 0.55:
        return "Moderate"
    else:
        return "Green"


def fetch_ndvi_for_point(lat, lon, catalog, max_cloud=20):
    """
    Query Planetary Computer for the most recent cloud-free Sentinel-2 scene
    at (lat, lon), then compute NDVI from B04/B08 pixel values.
    Handles CRS reprojection from WGS84 to the COG's native UTM CRS.
    """
    bbox_wgs84 = [lon - BUFFER_DEG, lat - BUFFER_DEG,
                  lon + BUFFER_DEG, lat + BUFFER_DEG]

    try:
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox_wgs84,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby=["-datetime"],
            max_items=1,
        )
        items = list(search.items())
        if not items:
            return None, None, "No scene found"

        item = planetary_computer.sign(items[0])
        scene_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"

        b04_url = item.assets["B04"].href
        b08_url = item.assets["B08"].href

        # Read Red band & reproject bbox to COG's CRS
        with rasterio.open(b04_url) as src_r:
            # Reproject our WGS84 bbox into the COG's native CRS (UTM)
            native_bbox = transform_bounds("EPSG:4326", src_r.crs, *bbox_wgs84)
            window = rasterio.windows.from_bounds(*native_bbox, transform=src_r.transform)
            red = src_r.read(1, window=window).astype(float)

        with rasterio.open(b08_url) as src_n:
            native_bbox = transform_bounds("EPSG:4326", src_n.crs, *bbox_wgs84)
            window = rasterio.windows.from_bounds(*native_bbox, transform=src_n.transform)
            nir = src_n.read(1, window=window).astype(float)

        if red.size == 0 or nir.size == 0:
            return None, scene_date, "Empty pixel window"

        # Ensure same shape (10m bands should match)
        min_h = min(red.shape[0], nir.shape[0])
        min_w = min(red.shape[1], nir.shape[1])
        red = red[:min_h, :min_w]
        nir = nir[:min_h, :min_w]

        # Sentinel-2 L2A values are reflectance * 10000; mask nodata (0)
        valid = (red > 0) & (nir > 0)
        if valid.sum() == 0:
            return None, scene_date, "All pixels nodata"

        denom = nir[valid] + red[valid]
        ndvi_vals = (nir[valid] - red[valid]) / denom
        mean_ndvi = float(np.nanmean(ndvi_vals))

        return round(mean_ndvi, 4), scene_date, None

    except Exception as e:
        return None, None, str(e)


def main():
    print("=" * 60)
    print("Sentinel-2 NDVI Fetcher (Planetary Computer)")
    print("=" * 60)

    # Load bonds
    bonds_path = os.path.join(DATA_DIR, "bonds_scored.csv")
    if not os.path.exists(bonds_path):
        import geopandas as gpd
        bonds_path = os.path.join(DATA_DIR, "municipal_bonds.geojson")
        gdf = gpd.read_file(bonds_path)
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        bonds_df = pd.DataFrame(gdf.drop(columns='geometry'))
    else:
        bonds_df = pd.read_csv(bonds_path)

    print(f"Loaded {len(bonds_df)} bonds\n")

    # Connect to Planetary Computer STAC
    print("Connecting to Microsoft Planetary Computer...")
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    print("  Connected.\n")

    # Fetch NDVI for each bond
    results = []
    for i, (_, row) in enumerate(bonds_df.iterrows()):
        ndvi, scene_date, err = fetch_ndvi_for_point(row['lat'], row['lon'], catalog)
        if err or ndvi is None:
            print(f"  [{i+1:2d}/50] {row['issuer'][:25]:25s}  FALLBACK  (err: {(err or 'None')[:40]})")
            results.append({"ndvi": 0.35, "ndvi_date": "fallback", "fuel_class": "Moderate"})
        else:
            fc = classify_ndvi(ndvi)
            print(f"  [{i+1:2d}/50] {row['issuer'][:25]:25s}  NDVI={ndvi:.3f}  ({fc})  [{scene_date}]")
            results.append({"ndvi": ndvi, "ndvi_date": scene_date, "fuel_class": fc})
        time.sleep(0.3)

    res_df = pd.DataFrame(results)
    bonds_df['ndvi'] = res_df['ndvi'].values
    bonds_df['ndvi_date'] = res_df['ndvi_date'].values
    bonds_df['fuel_class'] = res_df['fuel_class'].values

    # Save
    out_cols = ['bond_id', 'issuer', 'lat', 'lon', 'ndvi', 'ndvi_date', 'fuel_class']
    out_cols = [c for c in out_cols if c in bonds_df.columns]
    out_path = os.path.join(DATA_DIR, "ndvi_scores.csv")
    bonds_df[out_cols].to_csv(out_path, index=False)

    success = sum(1 for r in results if r['ndvi_date'] != 'fallback')
    print(f"\n{'=' * 60}")
    print(f"Results: {success}/{len(bonds_df)} bonds with real Sentinel-2 NDVI")
    print(f"Avg NDVI: {bonds_df['ndvi'].mean():.3f}")
    print(f"Min NDVI: {bonds_df['ndvi'].min():.3f} ({bonds_df.loc[bonds_df['ndvi'].idxmin(), 'issuer']})")
    print(f"Max NDVI: {bonds_df['ndvi'].max():.3f} ({bonds_df.loc[bonds_df['ndvi'].idxmax(), 'issuer']})")
    print(f"Saved -> {out_path}")
    print(f"\nRun src/analysis.py next to integrate NDVI into composite scores.")


if __name__ == "__main__":
    main()
