from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

# Define the Python function to call the training script
def train_model():
    subprocess.run(["python", "text_classification/src/text_classification/train.py"], check=True)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 5),  # Set your desired start date
    'retries': 1,
}

# Define the DAG
with DAG('train_model_dag',
         default_args=default_args,
         schedule_interval=None,  # You can set a schedule interval or use 'None' for manual trigger
         catchup=False) as dag:

    # Define the task
    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
    )

    # Set the task order (here it's just a single task)
    train_task
s