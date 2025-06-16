import sys, os, time, psutil, logging, threading
import pandas as pd
import joblib
import numpy as np

from flask import Flask, request, render_template
from prometheus_client import Counter, Gauge, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Optional jika kamu memang butuh preprocessor manual (tidak dipakai di sini)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Workflow-CI', 'MLProject')))
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

# --- Logging setup ---
logging.basicConfig(level=logging.DEBUG)

# --- Flask setup ---
app = Flask(__name__)

# --- Load model ---
model_path = r"D:\Submission\Membangun_sistem_machine_learning\Workflow-CI\models\Forest_model.pkl"
model_path = os.path.abspath(model_path)

try:
    model = joblib.load(model_path)
    logging.info(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None  # agar tidak crash di awal, tapi error tetap ditangani

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

# --- Logging setiap request ---
@app.before_request
def before_request():
    REQUESTS_TOTAL.labels(request.endpoint, request.method).inc()

# --- Halaman utama ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Prediksi via form ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="❌ Model tidak tersedia.")

    try:
        data = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }
        input_df = pd.DataFrame([data])

        start = time.time()
        prediction = model.predict(input_df)[0]
        duration = time.time() - start

        INFERENCE_LATENCY.observe(duration)
        PREDICTION_TOTAL.inc()
        PREDICTION_PER_CLASS.labels(class_=str(prediction)).inc()

        result = "Survived" if prediction == 1 else "Not Survived"
        logging.debug(f"Prediksi sukses: {result}. Waktu: {duration:.4f}s")
        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict').inc()
        logging.error(f"Error prediksi: {e}")
        return render_template('index.html', prediction_text=f"⚠️ Error during prediction: {e}")

# --- Prediksi via upload file CSV ---
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if model is None:
        return "❌ Model tidak tersedia.", 500

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

        start = time.time()
        pred = model.predict(df)
        duration = time.time() - start

        INFERENCE_LATENCY.observe(duration)
        PREDICTION_TOTAL.inc(len(df))

        for val in np.unique(pred):
            PREDICTION_PER_CLASS.labels(class_=str(val)).inc()

        df['Predicted'] = pred
        result_table = df.to_html(classes='table table-bordered', index=False)
        return render_template('index.html', tables=result_table)

    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict_csv').inc()
        logging.error(f"Error saat prediksi CSV: {e}")
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

# --- Jalankan aplikasi ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)