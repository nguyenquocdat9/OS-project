from flask import Flask, render_template, request, jsonify
import requests
import time

app = Flask(__name__, template_folder='templates')

# Cấu hình các EC2 model
EC2_CONFIG = {
    "rf": {"ip": "10.128.0.11", "port": 5001, "name": "RandomForest"},
    "xgb": {"ip": "10.128.0.12", "port": 5002, "name": "XGBoost"},
    "svm": {"ip": "10.128.0.15", "port": 5003, "name": "SVM"}
}

@app.route('/')
def home():
    return render_template('index.html')

def call_model_api(model_name, data):
    """Gọi API đến EC2 chứa model và đo thời gian"""
    start_time = time.time()
    try:
        url = f"http://{EC2_CONFIG[model_name]['ip']}:{EC2_CONFIG[model_name]['port']}/predict"
        response = requests.post(url, json=data, timeout=3)
        processing_time = round((time.time() - start_time) * 1000, 2)  # ms
        return {
            "prediction": response.json().get('prediction', 0),
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Lỗi gọi {model_name.upper()} API:", str(e))
        return {"prediction": 0, "processing_time": 0}

@app.route('/ensemble_predict', methods=['POST'])
def ensemble_predict():
    start_time = time.time()
    try:
        data = request.get_json()

        # Gọi song song 3 model
        rf_result = call_model_api("rf", data)
        xgb_result = call_model_api("xgb", data)
        svm_result = call_model_api("svm", data)

        # Kết quả tổng hợp
        final_pred = 1 if (rf_result['prediction'] + xgb_result['prediction'] + svm_result['prediction']) >= 2 else 0

        return jsonify({
            "status": "success",
            "predictions": {
                "random_forest": rf_result['prediction'],
                "xgb": xgb_result['prediction'],
                "svm": svm_result['prediction'],
                "final": final_pred
            },
            "processing_time": {
                "random_forest": rf_result['processing_time'],
                "xgb": xgb_result['processing_time'],
                "svm": svm_result['processing_time'],
                "total": round((time.time() - start_time) * 1000, 2)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
