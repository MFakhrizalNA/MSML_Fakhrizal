import os, sys, time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Tambahkan path ke preprocessing
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Eksperimen_SML_Fakhrizal'))
)
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

# Konfigurasi MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/MFakhrizalNA/MSML_Fakhrizal.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "MFakhrizalNA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "82b116729c790ec9bd93d7e8f99d7d815b531339"

mlflow.set_experiment("Titanic Survival Manual Logging")
dagshub.init(repo_owner='MFakhrizalNA', repo_name='MSML_Fakhrizal', mlflow=True)

# Load data
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Eksperimen_SML_Fakhrizal', 'preprocessing', 'cleaned_data.csv'))
data = pd.read_csv(csv_path)
X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = SklearnPreprocessor(
    num_columns=['Age', 'Fare'],
    ordinal_columns=['Pclass'],
    nominal_columns=['Sex', 'Embarked'],
    degree=2
)

# Models + Hyperparameter Grid
models = {
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]}
    ),
    "XGBClassifier": (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]}
    ),
    "Ridge": (
        Ridge(),
        {"model__alpha": [0.1, 1.0, 10.0]}
    ),
    "LinearRegression": (
        LinearRegression(),
        {}
    ),
    "RandomForestRegressor": (
        RandomForestRegressor(random_state=42),
        {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]}
    ),
    "XGBRegressor": (
        XGBRegressor(),
        {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]}
    )
}

for model_name, (model, param_grid) in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    is_classifier = "Classifier" in str(type(model))
    scoring_method = "accuracy" if is_classifier else "neg_mean_squared_error"

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring=scoring_method)

    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        start = time.time()
        grid_search.fit(X_train, y_train)
        end = time.time()
        training_time = end - start

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("training_time", training_time)

        if is_classifier:
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)
        else:
            y_test_float = y_test.astype(float)
            y_pred_float = y_pred.astype(float)

            mae = mean_absolute_error(y_test_float, y_pred_float)
            mse = mean_squared_error(y_test_float, y_pred_float)
            rmse = np.sqrt(mse)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

        # Logging model dan dataset
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact(csv_path, artifact_path="dataset")

        print(f"✅ Model {model_name} selesai dan dilog.")