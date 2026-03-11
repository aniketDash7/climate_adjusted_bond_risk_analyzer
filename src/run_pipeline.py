import subprocess
import sys
import os
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

def run_script(script_name, args=None):
    script_path = os.path.join(SRC_DIR, script_name)
    print(f"\n{'='*20} RUNNING: {script_name} {'='*20}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed with exit code {result.returncode}")
        return False
    
    print(f"✅ COMPLETED: {script_name} in {end_time - start_time:.2f}s")
    return True

def main():
    print("="*60)
    print(f"CLIMATE BOND RISK - MASTER PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: Data Ingestion (NDVI and Hazard Layers)
    # We assume FIRMS data is already present or handled by alert_engine, 
    # but we can trigger specific fetchers here.
    if not run_script("fetch_hazard_layers.py"):
        sys.exit(1)
        
    if not run_script("fetch_ndvi.py"):
        sys.exit(1)

    # Step 2: Generate Deep Learning Dataset (Real Historical Data)
    if not run_script("generate_dl_dataset.py"):
        sys.exit(1)

    # Step 3: Train/Retrain Deep Learning Model
    if not run_script("train_dl_model.py"):
        sys.exit(1)

    # Step 4: Run Final Analytics & Scoring
    if not run_script("analysis.py"):
        sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE SUCCESSFUL: Portfolio Scored and Models Updated.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
