import os
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIRMS_CSV = os.path.join(DATA_DIR, "firms_enriched.csv")

NUM_SAMPLES_PER_CLASS = 500
SEQ_LEN = 7

async def fetch_weather_sequence(session, lat, lon, end_date):
    """Fetch 7 days of daily weather leading up to end_date using OpenMeteo Archive API."""
    start_date = end_date - timedelta(days=SEQ_LEN - 1)
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max"
        f"&timezone=America%2FLos_Angeles"
    )
    
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                daily = data.get('daily', {})
                
                temps = daily.get('temperature_2m_max', [])
                precips = daily.get('precipitation_sum', [])
                winds = daily.get('wind_speed_10m_max', [])
                
                # Check if we got 7 days
                if len(temps) == SEQ_LEN:
                    # Construct features: [Temp, Humidity(mocked), Wind, Precip, NDVI(mocked)]
                    # We mock Humidity and NDVI based on Temp to keep the API load light
                    seq = []
                    for i in range(SEQ_LEN):
                        t = temps[i] if temps[i] is not None else 20.0
                        p = precips[i] if precips[i] is not None else 0.0
                        w = winds[i] if winds[i] is not None else 10.0
                        
                        # Mock humidity: hotter = drier
                        h = max(10, 80 - t * 1.5) + np.random.normal(0, 5)
                        # Mock NDVI: hotter/drier = lower NDVI
                        n = max(0.05, 0.6 - (t / 100)) + np.random.normal(0, 0.05)
                        
                        seq.append([t, h, w, p, n])
                    return np.array(seq)
    except Exception as e:
        pass
    
    # Fallback if API fails
    return None

async def generate_dataset():
    print(f"Loading {FIRMS_CSV} for positive fire samples...")
    df = pd.read_csv(FIRMS_CSV)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # 1. Positive Samples (Actual Fires)
    pos_df = df.sample(NUM_SAMPLES_PER_CLASS)
    
    # 2. Negative Samples (Random CA coordinates, random dates 2019-2023)
    # CA Bounds Roughly: Lat 32.5 to 42.0, Lon -124.0 to -114.0
    neg_lats = np.random.uniform(32.5, 42.0, NUM_SAMPLES_PER_CLASS)
    neg_lons = np.random.uniform(-124.0, -114.0, NUM_SAMPLES_PER_CLASS)
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    
    neg_dates = [start_date + timedelta(days=np.random.randint(days_between_dates)) for _ in range(NUM_SAMPLES_PER_CLASS)]
    
    # Collect all tasks
    tasks = []
    labels = []
    
    print(f"Fetching {NUM_SAMPLES_PER_CLASS * 2} historical 7-day weather sequences via OpenMeteo...")
    
    async with aiohttp.ClientSession() as session:
        # Queue positive tasks
        for _, row in pos_df.iterrows():
            tasks.append(fetch_weather_sequence(session, row['latitude'], row['longitude'], row['acq_date']))
            labels.append(1.0)
            
        # Queue negative tasks
        for i in range(NUM_SAMPLES_PER_CLASS):
            tasks.append(fetch_weather_sequence(session, neg_lats[i], neg_lons[i], neg_dates[i]))
            labels.append(0.0)
            
        # Execute concurrently across batches to not strictly overwhelm API
        X_list = []
        y_list = []
        
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i+chunk_size]
            chunk_labels = labels[i:i+chunk_size]
            
            results = await asyncio.gather(*chunk_tasks)
            
            for res, label in zip(results, chunk_labels):
                if res is not None:
                    X_list.append(res)
                    y_list.append([label])
                    
            print(f"Processed batch {i//chunk_size + 1}/{(len(tasks)//chunk_size)}")
            await asyncio.sleep(1) # rate limiting pause
            
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    
    print(f"Successfully generated dataset of shape X: {X_arr.shape}, y: {y_arr.shape}")
    
    # Save to disk
    np.save(os.path.join(DATA_DIR, "X_real.npy"), X_arr)
    np.save(os.path.join(DATA_DIR, "y_real.npy"), y_arr)
    print(f"Saved to {DATA_DIR}/X_real.npy and y_real.npy")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
