from flask import Flask, request, render_template
import pandas as pd
import joblib, sys, os

# Tambahkan path folder preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Fakhrizal import SklearnPreprocessor

app = Flask(__name__)

# Load model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models', 'Forest_model.pkl'))
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    
    file = request.files.get('file')
    if not file:
        return "❌ Tidak ada file yang diupload.", 400

    try:
        # Baca dan pra-proses
        df = pd.read_csv(file)
        predictions = model.predict(df)

        # Tambahkan kolom hasil prediksi
        df['Survived_Prediction'] = predictions

        # Tampilkan hasil
        result_table = df.to_html(classes='table table-bordered', index=False)
        return render_template('index.html', tables=result_table)
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)