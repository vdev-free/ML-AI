import mlflow
import os
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('day84-mlflow-intro')
os.makedirs('artifacts/day84', exist_ok=True)

with mlflow.start_run(run_name='helo-mlflow'):
    mlflow.log_param('demo_param', 123)
    mlflow.log_metric('demo_metric', 0.77)

    plt.figure(figsize=(6, 4))
    plt.bar(['A', "B", "C"], [1, 3, 2])
    plt.tight_layout()

    path = 'artifacts/day84/hello_plot.png'
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)
    print("Saved artifact:", path)

with mlflow.start_run(run_name='test_2'):
    mlflow.log_param('demo_param', 500)
    mlflow.log_metric('demo_metric', 0.5)

    plt.figure(figsize=(6, 4))
    plt.bar(['A', "B", "C"], [3, 0.5, 1])
    plt.tight_layout()

    path = 'artifacts/day84/test_2.png'
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)
    print("Run test_2, artifact:", path)

