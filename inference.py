
import json
import joblib
import pandas as pd
import os

MODEL_FILE = 'revenue_optimization_model.pkl'

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, MODEL_FILE))

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        df = pd.DataFrame(json.loads(request_body))
        return df[["subscribers", "avg_watch_hours", "monthly_fee", "marketing_spend"]]
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    return str(prediction)
