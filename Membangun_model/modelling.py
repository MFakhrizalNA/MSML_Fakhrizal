import sys, os, time
import mlflow
import joblib
import pandas as pd
import numpy as np
import dagshub
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from sklearn.utils.multiclass import unique_labels

# Path ke preprocessing class
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'Eksperimen_SML_Fakhrizal')
    ))
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

# Konfigurasi MLflow
mlflow.set_tracking_uri("https://dagshub.com/MFakhrizalNA/MSML_Fakhrizal.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "MFakhrizalNA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "82b116729c790ec9bd93d7e8f99d7d815b531339"

mlflow.set_experiment("Titanic Survival Prediction 1")
dagshub.init(repo_owner='MFakhrizalNA', repo_name='MSML_Fakhrizal', mlflow=True)

# Load data
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Eksperimen_SML_Fakhrizal', 'preprocessing', 'cleaned_data.csv'))
data = pd.read_csv(csv_path)
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing dan pipeline
preprocessor = SklearnPreprocessor(
    num_columns=['Age', 'Fare'],
    ordinal_columns=['Pclass'],
    nominal_columns=['Sex', 'Embarked'],
    degree=2
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Mulai manual MLflow run
with mlflow.start_run(run_name="ManualLog - RandomForestClassifier"):
    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()

    # Prediksi dan evaluasi
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    training_time = end - start

    support = y_test.value_counts().to_dict()

    # Manual logging semua metrik
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("ROC_AUC", roc_auc)
    mlflow.log_metric("Log_Loss", logloss)  # metrik tambahan 1
    mlflow.log_metric("Training_Time", training_time)  # metrik tambahan 2
    mlflow.log_metric("Support_Positive", support.get(1, 0))
    mlflow.log_metric("Support_Negative", support.get(0, 0))

    # Logging model dan artefak
    mlflow.sklearn.log_model(pipeline, "model")
    mlflow.log_artifact(csv_path, artifact_path="dataset")

    # Tampilkan hasil
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Training Time: {training_time:.2f} detik")
    print(f"Support (1): {support.get(1, 0)}, Support (0): {support.get(0, 0)}")

# Simpan model ke lokal
output_path = "models/randomforest_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Model disimpan di: {output_path}")