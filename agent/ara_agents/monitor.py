### Continuous Ara - ML - Train - Monitoring agent // runs on ara cloud
import ara_sdk as ara
# from . import ara_prompts as pmp  
import json
import subprocess
import time
from pprint import pprint
import re
import os
from pathlib import Path
from fastapi import FastAPI
import threading
import uvicorn

_cwd_ = os.getcwd()
BASE_DIR = Path(__file__).resolve().parents[2]
api_app = FastAPI()

MODEL_TRAIN_FILE_PATH = BASE_DIR / "model" / "sample_train_2.py"
STREAM_FILE_PATH = BASE_DIR / "main_stream.py"
TRAIN_LOG_FILE_PATH = BASE_DIR / "model" / "logs" / "training_logs.json"
DECISION_LOG_SAVE_PATH = BASE_DIR / "agent" / "ara_agents" / "logs" / "decision_logs.json"
LOCAL_IP = "192.168.1.100"

def get_ara_prompt_e1():
    system_instructions = """
    ROLE:
    You are an automated ML training monitoring agent.

    OBJECTIVE:
    Continuously analyze the latest machine learning training logs
    and determine whether a debugging workflow should be triggered.

    INSTRUCTIONS:
    1. Read the most recent training logs using the available tools.
    2. Identify abnormal patterns in training behavior.
    3. Decide whether a debugging trigger is required.
    4. Give proper and brief reasoning behind the trigger decision

    TRIGGER CONDITIONS (examples):
    - Loss becomes NaN
    - Loss increases continuously
    - Accuracy drops significantly
    - Training diverges
    - Unexpected runtime errors appear

    OUTPUT FORMAT:
    Return ONLY valid JSON in the following schema:

    {"utc_timestamp": "<ISO-8601 UTC time>", "epoch": "<int>", "train_loss": <float>, "is_trigger": true or false, "trigger_reason": <str>}

    RULES:
    - Do not include explanations outside JSON.
    - Always include UTC timestamp.
    - Always include current training epoch, read it from the training logs
    - Always include current training loss, read it from the training logs
    - Always return valid JSON.
    """
    
    return system_instructions

def get_ara_prompt_e2():
    
    system_instructions = """
    ROLE:
    You are an automated ML training monitoring agent.

    OBJECTIVE:
    Continuously analyze historical training logs and determine whether
    a debugging workflow should be triggered based on current behavior
    and recent training trends.

    --------------------------------------------------

    DATA SOURCE:

    You MUST call read_train_logs().

    The tool returns a JSON object in this structure:

    {
    "ml_model_training_logs": [
            {
            "epoch": <int>,
            "train_loss": <float>,
            "val_loss": <float>,
            "val_accuracy": <float>,
            "avg_grad_norm": <float>
            }, ...
            ]
    }

    IMPORTANT:

    • The "epochs" list contains historical training records.
    • The LAST element in "epochs" is the CURRENT epoch.
    • Previous elements represent historical training behavior.

    Use:

    Current epoch → epochs[-1]

    Previous epoch → epochs[:-2] (if available)

    --------------------------------------------------

    ANALYSIS LOGIC:

    Always perform the following steps:

    1. Read the full epochs history.
    2. Identify the CURRENT epoch using epochs[-1].
    3. Use earlier epochs to analyze trends.
    4. Determine whether training behavior is abnormal.
    5. Decide whether to trigger debugging.

    --------------------------------------------------

    TREND-BASED TRIGGER CONDITIONS:

    Trigger debugging if ANY condition occurs:

    CRITICAL CONDITIONS:

    • train_loss is NaN or null
    • val_loss is NaN or null
    • avg_grad_norm is extremely large
    • training diverges suddenly

    TREND CONDITIONS:

    • train_loss increases for multiple consecutive epochs
    • val_loss increases across recent epochs
    • val_accuracy drops significantly
    • gradients grow rapidly across epochs
    • loss stops improving unexpectedly

    SAFE CONDITIONS:

    If training is stable or improving:
    is_trigger = false

    --------------------------------------------------

    OUTPUT FORMAT:

    Return ONLY valid JSON.

    No markdown.
    No explanations.
    No extra text.

    Schema:

    {
    "utc_timestamp": "<ISO-8601 UTC time>",
    "epoch": <int>,
    "train_loss": <float>,
    "is_trigger": true or false,
    "trigger_reason": "<brief reasoning>"
    }

    --------------------------------------------------

    MANDATORY RULES:

    • Always call read_train_logs()
    • Always identify current epoch using epochs[-1]
    • Always extract:

    epoch → epochs[-1].epoch
    train_loss → epochs[-1].train_loss

    • Use historical epochs ONLY for trend detection
    • Never invent values
    • Never assume missing values
    • Never default epoch to 0 unless present in logs
    • Always use utc_now() to obtain current time
    • Output ONLY JSON

    Avoid long explanations.
    """

    return system_instructions


def run_adjoin_code_parallel(train_file_path=MODEL_TRAIN_FILE_PATH, stream_file_path=STREAM_FILE_PATH, is_train=True):
    if is_train:
        print('Training the model ...')
        subprocess.run(['python', train_file_path], check=True)
    else:
        print('loaded trained logs')
        
    subprocess.run(['streamlit', 'run', stream_file_path])

@ara.tool
def utc_now():
    from datetime import datetime, timezone
    """provides the UTC time"""
    return {"utc_time": datetime.now(timezone.utc).isoformat()}

## GET-REST-call
@api_app.get('/training_logs')
def get_train_logs_api(train_path=TRAIN_LOG_FILE_PATH):
    import json
    with open(train_path, 'r') as f:
        train_logs_json = json.load(f)
    f.close()
    return train_logs_json

## serve
def api_serve(port=8000):
    print(f'serving-train-log-API at {LOCAL_IP}:{port}')
    def api_run():
        uvicorn.run(api_app, host=LOCAL_IP, port=port)
    # start API in background thread
    threading.Thread(target=api_run, daemon=True).start()

@ara.tool
def read_train_logs():
    """reads the ml-model training-logs"""
    import json
    from urllib.request import urlopen
    
    print('realtime-train-logs-read')
    
    LOCAL_IP = "192.168.1.100"
    api_path = f"http://{LOCAL_IP}:8000/training_logs"
    try:
        with urlopen(api_path) as response:
            train_logs_json = json.loads(response.read().decode())    
        train_logs_json = train_logs_json['epochs']
        
        return {'ml_model_training_logs': train_logs_json}
    
    except Exception as e:
        print("ACTUAL ERROR:", repr(e))
        raise e                        

ara.Job(
    "ara-monitor-agent", ## triggers on train-errors
    system_instructions=(get_ara_prompt_e2())
)

def clean_ara_stdout(stdout):
    match = re.search(r'\{.*\}', stdout, re.DOTALL)
    if match:
        json_text = match.group()
        decision = json.loads(json_text)
        return decision
    else:
        print("No JSON found in output")
        return None

## do a seperate one-off ara-cloud call cause logs streams don't work | This needs to be crontabbed/cycled locally
def run_ara_monitor_subprocess():
    output_decision = subprocess.run(['ara', 'run', 'agent/ara_agents/monitor.py'], 
                                     capture_output=True, 
                                     text=True,
                                     cwd=BASE_DIR) ## terminal/working directory-mismatch
    output_decision = clean_ara_stdout(output_decision.stdout)
    print(' --- output-decision --- ')
    pprint(output_decision)
    return {'ara_monitor_decision': output_decision}

def run_ara_cloud_register(is_auth=False):
    if is_auth:
        subprocess.run(['ara', 'auth', 'login', '--reauth'], 
                        cwd=BASE_DIR) ## terminal/working directory-mismatch
        print('ara-auth-login')
    subprocess.run(['ara', 'run', 'agent/ara_agents/monitor.py', '--cron', '"*/5 * * * *"'], 
                   capture_output=True, 
                   text=True,
                   cwd=BASE_DIR) ## terminal/working directory-mismatch
    print('registered-ara-cloud')

def run_deregister_cloud():
    subprocess.run(['ara', 'deploy', 'agent/ara_agents/monitor.py', '--activate', 'false'],
                   cwd=BASE_DIR)
    print('deregistered-ara-cloud')

## cyclic-run // trigger with patience (p)
def save_ara_logs(log_save_path=DECISION_LOG_SAVE_PATH):
    """returns decision logs"""
    
    ## runs N-cycles locally
    for i in range(10):
        
        out_stream = run_ara_monitor_subprocess()
        
        with open(log_save_path, 'w') as f:    
            json.dump(out_stream, f)
        f.close()
        
        print()
        print(f'saved-ara-out-decision-stream # {i+1}')
        print()
        
        out_stream_decision = clean_ara_stdout(out_stream["ara_monitor_decision"]["result"]["output_text"])
        
        ## min-epoch-execution-patience
        if out_stream_decision['epoch'] > 3: ## epoch-threshold (tune)            
            ## parallel-execution-trigger
            if out_stream_decision["is_trigger"]:
                print('<Parallel> Triggered & Running ...')
            run_adjoin_code_parallel(is_train=False)
        
        # run_deregister_cloud() ## cause the cloud registered state is fixed in the runtime
        time.sleep(5*60) ## force 5mins retrieval-wait

if __name__ == '__main__':
    print('__running__ara/monitor___')
    
    ## serve-local-api-endpoint
    api_serve() ## local:8000 port
    
    # ## ara-cloud-register
    run_ara_cloud_register(is_auth=True)

    # ## local-cyclic-monitoring
    save_ara_logs()
    
    run_deregister_cloud()
 