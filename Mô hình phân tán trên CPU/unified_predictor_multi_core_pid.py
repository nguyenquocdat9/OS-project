from flask import Flask, request, jsonify, send_file
import joblib
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import psutil

app = Flask(__name__)

# Load models
rf_model = joblib.load("random_forest.joblib")
xgb_model = joblib.load("xgboost.joblib")
svm_model = joblib.load("svm.joblib")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route("/")
def index():
    return send_file("index_unified_final.html")

def predict_model(model, X, name):
    pid = os.getpid()
    core_id = psutil.Process(pid).cpu_num()
    print(f" {name} model running on PID {pid} | Core #{core_id}")
    time.sleep(40)  # giữ tiến trình đủ lâu để dễ kiểm tra thủ công
    start = time.time()
    pred = int(model.predict(X)[0])
    duration = round((time.time() - start) * 1000, 2)
    return pred, duration

@app.route("/ensemble_predict", methods=["POST"])
def ensemble_predict():
    start_time = time.time()
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(predict_model, rf_model, input_df, "RandomForest"),
                executor.submit(predict_model, xgb_model, input_df, "XGBoost"),
                executor.submit(predict_model, svm_model, input_df, "SVM")
            ]
            results = [f.result() for f in futures]

        rf_pred, rf_time = results[0]
        xgb_pred, xgb_time = results[1]
        svm_pred, svm_time = results[2]

        final_pred = 1 if (rf_pred + xgb_pred + svm_pred) >= 2 else 0

        return jsonify({
            "status": "success",
            "predictions": {
                "random_forest": rf_pred,
                "xgb": xgb_pred,
                "svm": svm_pred,
                "final": final_pred
            },
            "processing_time": {
                "random_forest": rf_time,
                "xgb": xgb_time,
                "svm": svm_time,
                "total": round((time.time() - start_time) * 1000, 2)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
