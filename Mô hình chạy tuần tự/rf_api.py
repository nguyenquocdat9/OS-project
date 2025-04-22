from flask import Flask, request, jsonify
import joblib
import time
import os
import psutil

app = Flask(__name__)
model = joblib.load('random_forest.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    pid = os.getpid()
    core_id = psutil.Process(pid).cpu_num()
    print(f"RandomForest model running on PID {pid} | Core #{core_id}")
    time.sleep(10)  # giữ tiến trình sống đủ lâu để kiểm tra

    start_time = time.time()
    data = request.json

    input_data = [[
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]]

    prediction = int(model.predict(input_data)[0])
    processing_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "prediction": prediction,
        "processing_time": processing_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
