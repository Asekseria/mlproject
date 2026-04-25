# =========================================
# MLflow Full Demo (Drift + Training)
# =========================================

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

# =========================================
# 1. Set MLflow Experiment
# =========================================

mlflow.set_experiment("mlflow-devops-mlops-demo")

# =========================================
# 2. Create Dataset
# =========================================

X, y = make_classification(n_samples=1000, n_features=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Simulate drifted production data
X_prod = X_test + np.random.normal(0.5, 1.2, X_test.shape)

# =========================================
# 3. Train Model
# =========================================

with mlflow.start_run(run_name="model_training"):

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters & metrics
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# =========================================
# 4. Drift Detection (KS Test)
# =========================================

with mlflow.start_run(run_name="drift_detection"):

    ks_stat, p_value = ks_2samp(X_test[:, 0], X_prod[:, 0])

    mlflow.log_metric("ks_statistic", ks_stat)
    mlflow.log_metric("p_value", p_value)

    drift_detected = p_value < 0.05
    mlflow.log_param("drift_detected", drift_detected)

# =========================================
# 5. Visualization
# =========================================

plt.figure()
plt.hist(X_test[:, 0], alpha=0.5, label="Train/Test")
plt.hist(X_prod[:, 0], alpha=0.5, label="Production")
plt.legend()
plt.title("Feature Drift Visualization")

plt.savefig("drift_plot.png")

with mlflow.start_run(run_name="artifact_logging"):
    mlflow.log_artifact("drift_plot.png")

print("✅ All runs logged successfully!")