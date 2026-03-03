"""
Phase 1: Satellite Fire Data Acquisition
Sources:
  - NASA FIRMS VIIRS 375m (Standard Processing) -- active fire detections 2019-2023
  - OpenMeteo Historical API -- weather at each fire detection date
Output:
  - data/firms_raw/viirs_{year}.csv   (raw satellite detections)
  - data/firms_enriched.csv           (fire detections + weather features)
"""
import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "firms_raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ----- CONFIG -----
FIRMS_KEY = "bf7fd4da4addb12d92f333e7f5588861"
# California bounding box: west, south, east, north
CA_BBOX  = "-124.5,32.5,-114.0,42.0"
YEARS    = [2019, 2020, 2021, 2022, 2023]
CHUNK_DAYS = 5

# ----- STEP 1: Download raw FIRMS data per-year -----

def date_chunks(year):
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    chunks = []
    cur = start
    while cur <= end:
        days = min(CHUNK_DAYS, (end - cur).days + 1)
        chunks.append((cur.strftime("%Y-%m-%d"), days))
        cur += timedelta(days=days)
    return chunks

def fetch_year(year):
    out_path = os.path.join(RAW_DIR, f"viirs_{year}.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if len(existing) > 100:
            print(f"  {year}: already downloaded ({len(existing)} detections), skipping.")
            return

    print(f"\nDownloading {year}...")
    frames = []
    chunks = date_chunks(year)

    for i, (date_str, n_days) in enumerate(chunks):
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{FIRMS_KEY}/VIIRS_SNPP_SP/{CA_BBOX}/{n_days}/{date_str}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and "latitude" in resp.text[:200]:
                from io import StringIO
                df_chunk = pd.read_csv(StringIO(resp.text))
                df_chunk = df_chunk[df_chunk['confidence'].isin(['h', 'n'])]
                frames.append(df_chunk)
            time.sleep(0.4)
        except Exception as e:
            print(f"    Error {date_str}: {e}")

        if (i+1) % 20 == 0:
            print(f"  {year}: {i+1}/{len(chunks)} chunks done...")

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {year}: saved {len(full):,} detections -> {size_mb:.1f} MB")
    else:
        print(f"  {year}: no data")

# ----- STEP 2: Enrich with historical weather -----

def fetch_weather_for_point(lat, lon, date_str):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean"
        f"&timezone=America%2FLos_Angeles"
    )
    try:
        r = requests.get(url, timeout=15).json()
        d = r.get("daily", {})
        return {
            "temperature":   d.get("temperature_2m_max", [None])[0],
            "precipitation": d.get("precipitation_sum", [None])[0],
            "wind_speed":    d.get("wind_speed_10m_max", [None])[0],
            "humidity":      d.get("relative_humidity_2m_mean", [None])[0],
        }
    except:
        return {"temperature": None, "precipitation": None, "wind_speed": None, "humidity": None}

def enrich_with_weather(df, sample_n=5000):
    print(f"\nEnriching {min(sample_n, len(df))} fire detections with weather...")
    df = df.sample(n=min(sample_n, len(df)), random_state=42).copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")

    temps, precips, winds, humids = [], [], [], []
    for i, row in df.iterrows():
        w = fetch_weather_for_point(row['latitude'], row['longitude'], row['acq_date'])
        temps.append(w['temperature'])
        precips.append(w['precipitation'])
        winds.append(w['wind_speed'])
        humids.append(w['humidity'])
        if len(temps) % 500 == 0 and len(temps) > 0:
            print(f"  Weather: {len(temps)}/{min(sample_n, len(df))}...")
        time.sleep(0.05)

    df['temperature']   = temps
    df['precipitation'] = precips
    df['wind_speed']    = winds
    df['humidity']      = humids
    return df.dropna(subset=['temperature', 'wind_speed', 'humidity'])

# ----- MAIN -----
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1a: Downloading NASA FIRMS VIIRS 375m data (CA)")
    print("=" * 60)
    for year in YEARS:
        fetch_year(year)

    print("\n" + "=" * 60)
    print("Phase 1b: Merging and enriching with weather data")
    print("=" * 60)
    all_files = [os.path.join(RAW_DIR, f"viirs_{y}.csv") for y in YEARS
                 if os.path.exists(os.path.join(RAW_DIR, f"viirs_{y}.csv"))]

    all_fires = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    print(f"Total fire detections (CA, 2019-2023): {len(all_fires):,}")

    total_bytes = sum(os.path.getsize(f) for f in all_files)
    print(f"Total satellite data size: {total_bytes/1e6:.1f} MB ({total_bytes/1e9:.3f} GB)")

    enriched = enrich_with_weather(all_fires, sample_n=6000)
    enriched_out = os.path.join(DATA_DIR, "firms_enriched.csv")
    enriched.to_csv(enriched_out, index=False)
    print(f"\nSaved {len(enriched):,} enriched fire points -> {enriched_out}")
    print("Phase 1 complete. Run src/train_model.py next.")
