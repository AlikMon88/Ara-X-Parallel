from fastapi import FastAPI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_TRAIN_FILE_PATH = BASE_DIR / "model" / "sample_train_2.py"
STREAM_FILE_PATH = BASE_DIR / "main_stream.py"
TRAIN_LOG_FILE_PATH = BASE_DIR / "model" / "logs" / "training_logs.json"

api_app = FastAPI()

## GET-REST-call
@api_app.get('/training_logs')
def get_train_logs_api(train_path=TRAIN_LOG_FILE_PATH):
    import json
    with open(train_path, 'r') as f:
        train_logs_json = json.load(f)
    f.close()
    return train_logs_json


get_train_logs_api()