import sys, os, json
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, render_template, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Tambahkan path ke modul preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Workflow-CI', 'MLProject')))
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

app = Flask(__name__)

# ✅ Inisialisasi preprocessor dengan kolom sesuai training pipeline
preprocessor = SklearnPreprocessor(
    num_columns=['Age', 'Fare'],
    ordinal_columns=['Pclass'],
    nominal_columns=['Sex', 'Embarked']
)

# ✅ Endpoint MLflow model API yang sudah diserve
MODEL_API_URL = 'http://127.0.0.1:5005/invocations'

@app.route('/metrics')
def metrics_route():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }
        input_df = pd.DataFrame([data])
        input_df_processed = preprocessor.transform(input_df)

        # Format input sesuai format JSON yang diminta MLflow
        json_data = {
            "columns": input_df_processed.columns.tolist(),
            "data": input_df_processed.values.tolist()
        }

        response = requests.post(MODEL_API_URL, json=json_data)
        response.raise_for_status()  # Akan raise error kalau response 4xx/5xx

        prediction = response.json()[0]
        status = "✅ Survived" if prediction == 1 else "❌ Not Survived"

        return render_template('index.html', prediction_text=f"Prediction: {status}")
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat memproses input: {e}", 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('file')
    if not file:
        return "❌ Tidak ada file yang diupload.", 400

    try:
        df = pd.read_csv(file)
        df_processed = preprocessor.transform(df)

        json_data = {
            "columns": df_processed.columns.tolist(),
            "data": df_processed.values.tolist()
        }

        response = requests.post(MODEL_API_URL, json=json_data)
        response.raise_for_status()

        predictions = response.json()
        df['Prediction'] = predictions
        df['Prediction_Label'] = df['Prediction'].apply(lambda x: 'Survived' if x == 1 else 'Not Survived')

        result_table = df.to_html(classes='table table-bordered', index=False)
        return render_template('index.html', tables=result_table)
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
