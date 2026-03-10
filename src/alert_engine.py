import asyncio
import json
import os
import time
from datetime import datetime
import pandas as pd
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from math import radians, sin, cos, sqrt, atan2

app = FastAPI(title="Climate Risk - Real-Time Alerting Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BONDS_CSV = os.path.join(DATA_DIR, "bonds_scored.csv")

# NASA FIRMS API (VIIRS SNPP, 24h trailing data)
FIRMS_MAP_KEY = "bf7fd4da4addb12d92f333e7f5588861"
FIRMS_URL = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP/USA/1"

# Alert radius in kilometers
ALERT_RADIUS_KM = 5.0

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_live_fires() -> pd.DataFrame:
    """Fetch active fires from NASA FIRMS (last 24 hours)."""
    try:
        # We fetch the CSV and parse it.
        # Format: latitude,longitude,acq_date,acq_time,...
        df = pd.read_csv(FIRMS_URL, usecols=['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence'])
        # Filter for recent California bounds roughly to minimize processing
        df = df[(df['latitude'] > 32.0) & (df['latitude'] < 42.5)]
        df = df[(df['longitude'] < -114.0) & (df['longitude'] > -124.5)]
        return df
    except Exception as e:
        print(f"Error fetching FIRMS data: {e}")
        return pd.DataFrame()

async def poll_firms_and_alert():
    """Background task to poll NASA FIRMS and emit alerts."""
    print("Alerting engine background task started...")
    seen_alerts = set() # Avoid spanning the same alert

    while True:
        if len(manager.active_connections) > 0:
            if not os.path.exists(BONDS_CSV):
                print(f"Waiting for {BONDS_CSV} to exist...")
                await asyncio.sleep(10)
                continue

            bonds_df = pd.read_csv(BONDS_CSV)
            print(f"[{datetime.now().isoformat()}] Polling NASA FIRMS for active fires...")
            fires_df = get_live_fires()

            if not fires_df.empty:
                print(f"  Found {len(fires_df)} active fires in CA bounding box.")
                for _, fire in fires_df.iterrows():
                    flat, flon = fire['latitude'], fire['longitude']
                    
                    # Check distance to all bonds
                    for _, bond in bonds_df.iterrows():
                        blat, blon = bond['lat'], bond['lon']
                        dist = haversine(flat, flon, blat, blon)
                        
                        if dist <= ALERT_RADIUS_KM:
                            alert_id = f"{bond['bond_id']}_{fire['acq_date']}_{fire['acq_time']}_{flat:.2f}_{flon:.2f}"
                            
                            if alert_id not in seen_alerts:
                                seen_alerts.add(alert_id)
                                alert_msg = {
                                    "type": "RED_ALERT",
                                    "timestamp": datetime.now().isoformat(),
                                    "bond_id": bond['bond_id'],
                                    "issuer": bond['issuer'],
                                    "distance_km": round(dist, 2),
                                    "fire_lat": flat,
                                    "fire_lon": flon,
                                    "confidence": fire['confidence']
                                }
                                print(f"🚨 RED ALERT! Fire {dist:.2f}km from {bond['issuer']}!")
                                await manager.broadcast(alert_msg)
            else:
                print("  No active fires found (or API error).")
                
            # Simulate a live fire for demonstration purposes if no real fires are near
            # We inject a fake fire near San Francisco or Los Angeles occasionally
            if len(seen_alerts) == 0 and int(time.time()) % 3 == 0:
                 fake_bond = bonds_df.sample(1).iloc[0]
                 dist = 3.1
                 alert_msg = {
                     "type": "SIMULATED_ALERT",
                     "timestamp": datetime.now().isoformat(),
                     "bond_id": fake_bond['bond_id'],
                     "issuer": fake_bond['issuer'],
                     "distance_km": dist,
                     "fire_lat": fake_bond['lat'] + 0.02, # Slightly offset
                     "fire_lon": fake_bond['lon'] + 0.02,
                     "confidence": "H"
                 }
                 print(f"🚨 SIMULATED ALERT! Fire {dist}km from {fake_bond['issuer']}!")
                 await manager.broadcast(alert_msg)

        # Poll every 30 seconds for demo purposes
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(poll_firms_and_alert())

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from the client, just keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    # Run on a different port than the main API
    uvicorn.run(app, host="0.0.0.0", port=5001)
