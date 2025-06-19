import sys, os, time, psutil, logging, threading, json, requests
import pandas as pd
import numpy as np

from flask import Flask, request, render_template
from prometheus_client import Counter, Gauge, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# --- Logging setup ---
logging.basicConfig(level=logging.DEBUG)

# --- Flask setup ---
app = Flask(__name__)

# --- Konfigurasi Endpoint API Model ---
MODEL_API_URL = "http://127.0.0.1:5005/invocations"  # Ganti jika endpoint berbeda

# --- Prometheus Metrics ---
REQUESTS_TOTAL = Counter('requests_total', 'Total permintaan masuk', ['endpoint', 'method'])
REQUESTS_FAILED = Counter('requests_failed_total', 'Total permintaan gagal', ['endpoint'])

PREDICTION_TOTAL = Counter('prediction_total', 'Total prediksi berhasil')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Total prediksi error')
PREDICTION_PER_CLASS = Counter('prediction_per_class', 'Jumlah prediksi per kelas', ['class_'])

INFERENCE_LATENCY = Summary('inference_latency_seconds', 'Waktu inferensi model')

INPUT_DATA_SIZE = Gauge('input_data_size_bytes', 'Ukuran file input CSV')
CSV_ROW_COUNT = Gauge('csv_row_count', 'Jumlah baris CSV')
MODEL_LOAD_TIMESTAMP = Gauge('model_load_timestamp_seconds', 'Waktu model dimuat')

CPU_USAGE = Gauge('cpu_percent_usage', 'Penggunaan CPU (%)')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Penggunaan memori (byte)')
FLASK_UPTIME = Gauge('flask_uptime_seconds', 'Lama aplikasi berjalan (detik)')
ACTIVE_THREADS = Gauge('active_threads_total', 'Jumlah thread aktif')

start_time = time.time()
MODEL_LOAD_TIMESTAMP.set_to_current_time()

# --- Monitor sistem setiap 5 detik ---
def monitor_resources():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        ACTIVE_THREADS.set(threading.active_count())
        FLASK_UPTIME.set(time.time() - start_time)
        time.sleep(5)

threading.Thread(target=monitor_resources, daemon=True).start()

# --- Tambahkan /metrics endpoint Prometheus ---
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@app.before_request
def before_request():
    REQUESTS_TOTAL.labels(request.endpoint, request.method).inc()

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
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }
        input_df = pd.DataFrame([data])

        payload = {
            "dataframe_split": {
                "columns": input_df.columns.tolist(),
                "data": input_df.values.tolist()
            }
        }

        start = time.time()
        response = requests.post(MODEL_API_URL, json=payload)
        duration = time.time() - start

        if response.status_code == 200:
            prediction = response.json()[0]
            result = "Survived" if prediction == 1 else "Not Survived"
            INFERENCE_LATENCY.observe(duration)
            PREDICTION_TOTAL.inc()
            PREDICTION_PER_CLASS.labels(class_=str(prediction)).inc()
            logging.debug(f"Prediksi sukses: {result}. Waktu: {duration:.4f}s")
            return render_template('index.html', prediction_text=f"Prediction: {result}")
        else:
            raise ValueError(f"Model API error: {response.text}")

    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict').inc()
        logging.error(f"Error prediksi: {e}")
        return render_template('index.html', prediction_text=f"⚠️ Error during prediction: {e}")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('file')
    if not file:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict_csv').inc()
        return "❌ Tidak ada file yang diupload.", 400

    try:
        content = file.read()
        INPUT_DATA_SIZE.set(len(content))
        file.seek(0)

        df = pd.read_csv(file)
        CSV_ROW_COUNT.set(len(df))

        payload = {
            "dataframe_split": {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }
        }

        start = time.time()
        response = requests.post(MODEL_API_URL, json=payload)
        duration = time.time() - start

        if response.status_code == 200:
            predictions = response.json()
            df['Predicted'] = predictions

            INFERENCE_LATENCY.observe(duration)
            PREDICTION_TOTAL.inc(len(predictions))

            for val in np.unique(predictions):
                PREDICTION_PER_CLASS.labels(class_=str(val)).inc()

            result_table = df.to_html(classes='table table-bordered', index=False)
            return render_template('index.html', tables=result_table)

        else:
            raise ValueError(f"Model API error: {response.text}")

    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict_csv').inc()
        logging.error(f"Error saat prediksi CSV: {e}")
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
