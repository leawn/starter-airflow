from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['restack+publisher@restack.io'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlflow_train_model',
    default_args=default_args,
    description='A simple DAG to run MLflow training script',
    schedule_interval=timedelta(days=1),
)

# Make sure to use the absolute path to your train.py script
train_model_task = BashOperator(
    task_id='train_model',
    bash_command='python3.8 /opt/airflow/mlflow/train.py',
    env={'MLFLOW_TRACKING_URI': 'https://mlt6ur.clj5khk.gcp.restack.it', 'MLFLOW_TRACKING_USERNAME': 'Restack', 'MLFLOW_TRACKING_PASSWORD': 'd3stg6v3xb'},  # Set your MLflow tracking URI
    dag=dag,
)

train_model_task
