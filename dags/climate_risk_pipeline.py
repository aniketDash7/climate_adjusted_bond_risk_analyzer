from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Define standard default arguments for the DAG operations
default_args = {
    'owner': 'climate_bond_risk',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG: Runs daily at midnight to fetch new satellite data and score new bonds
with DAG(
    'climate_risk_ml_pipeline',
    default_args=default_args,
    description='End-to-end Geospatial ML pipeline for Municipal Bond Risk Analysis',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['finance', 'climate', 'ml'],
) as dag:

    # Task 1: Fetch live Satellite imagery and Geospatial hazard data
    fetch_data = BashOperator(
        task_id='fetch_satellite_and_hazard_data',
        bash_command='cd /app && python src/fetch_hazard_layers.py && python src/fetch_ndvi.py',
    )

    # Task 2: Build the Time-Series datasets (Historical Weather + Fire Anomalies)
    build_dataset = BashOperator(
        task_id='build_historical_dl_sequences',
        bash_command='cd /app && python src/generate_dl_dataset.py',
    )

    # Task 3: Retrain the PyTorch Deep Learning Model (CNN + LSTM) on the new data
    train_dl_model = BashOperator(
        task_id='train_temporal_fire_path_model',
        bash_command='cd /app && python src/train_dl_model.py',
    )

    # Task 4: Run the main Analysis engine to generate Financial Risk scores (Spread/VaR)
    score_portfolio = BashOperator(
        task_id='score_municipal_bond_portfolio',
        bash_command='cd /app && python src/analysis.py',
    )

    # Define the execution dependencies (The Pipeline Topology)
    fetch_data >> build_dataset >> train_dl_model >> score_portfolio
