### Continuous Ara - ML - Train - Monitoring agent // runs on ara cloud
import ara_sdk as ara
import json
import subprocess
import time
from pprint import pprint
import re
import os
from pathlib import Path
import pickle as pkl
# from ara_prompts import *

_cwd_ = os.getcwd()
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_TRAIN_FILE_PATH = BASE_DIR / "model" / "sample_train_2.py"
STREAM_FILE_PATH = BASE_DIR / "main_stream.py"
TRAIN_LOG_FILE_PATH = BASE_DIR / "model" / "logs" / "training_logs.json"
DECISION_LOG_SAVE_PATH = BASE_DIR / "agent" / "ara_agents" / "logs" / "decision_logs.json"
PRIOR_CONTEXT_PATH = BASE_DIR / "stream" / "stream_output" / "final_response.txt" 
LOCAL_IP = "192.168.1.100"
PUBLIC_IP = "2409:4060:2e14:51b4:6187:b0a4:c632:18dd"
NGROK_TUNNEL = "https://eggshell-wrecking-jingle.ngrok-free.dev" ## tunneling to localhost:8000

def get_ara_prompt_e3():
    
    system_instructions = """
    
    ROLE:
    ML Training Monitor Agent.

    TOOLS:
    You MUST call these tools every run:

    1. read_train_logs()
    Returns:
    {
        "ml_model_training_logs": [
        {
            "epoch": int,
            "train_loss": float,
            "val_loss": float,
            "val_accuracy": float,
            "avg_grad_norm": float
        }
        ]
    }

    2. retrieve_prior_run_context()
    Returns previous run info (optional).

    3. utc_now()
    Returns current UTC time.

    --------------------------------------------------

    STEP ORDER (MANDATORY):

    1. Call read_train_logs()
    2. Call utc_now()
    3. Read logs list
    4. Set:

    current_epoch = logs[-1]
    previous_epochs = logs[:-1]

    5. Extract:

    epoch = current_epoch.epoch
    train_loss = current_epoch.train_loss
    val_loss = current_epoch.val_loss
    val_accuracy = current_epoch.val_accuracy
    grad_norm = current_epoch.avg_grad_norm

    --------------------------------------------------

    TRIGGER RULES:

    Set is_trigger = true if ANY condition happens.

    CRITICAL:

    - train_loss is NaN
    - val_loss is NaN
    - grad_norm is extremely large
    - loss jumps suddenly

    TREND:

    - val_loss increasing for 3+ epochs
    - train_loss increasing for 3+ epochs
    - val_accuracy drops significantly
    - loss stopped improving

    Otherwise:

    is_trigger = false

    --------------------------------------------------

    OUTPUT FORMAT:

    Return ONLY JSON.

    {
    "utc_timestamp": "<from utc_now()>",
    "epoch": <epoch>,
    "train_loss": <train_loss>,
    "is_trigger": true or false,
    "trigger_reason": "<short reason>"
    }

    RULES:

    - Always use logs[-1] as current epoch
    - Never guess values
    - Never skip tool calls
    - Never output text outside JSON
    """

    return system_instructions


def llm_caching_prompt():
    system_instruction = """
    ROLE:
    ML training monitor with memory.

    TOOLS:
    Call:
    - read_previous_state_llm_summary()  -> previous state
    - read_train_logs()   -> current epoch log
    - utc_now()           -> current time

    --------------------------------

    INPUT:

    previous_state = read_previous_state_llm_summary()

    current_epoch = read_train_logs()

    Use:

    epoch
    train_loss
    val_loss
    val_accuracy
    avg_grad_norm

    --------------------------------

    TASK:

    1. Compare current val_loss with previous state.
    2. Decide if training is worsening.
    3. Update internal state.
    4. Trigger if performance worsens repeatedly.

    --------------------------------

    STATE SHOULD TRACK:

    - best_val_loss
    - last_val_loss
    - worsening_streak

    If no previous state exists:
    Initialize using current val_loss.

    --------------------------------

    TRIGGER LOGIC:

    Trigger if:

    - val_loss increases for multiple epochs
    OR
    - val_loss is much worse than best_val_loss
    OR
    - accuracy drops repeatedly

    Otherwise:
    Do not trigger.

    --------------------------------

    OUTPUT:

    Return ONLY JSON:

    {
    "utc_timestamp": "<utc_now()>",
    "epoch": epoch,
    "train_loss": train_loss,
    "is_trigger": true or false,
    "trigger_reason": "<short reason>",
    "updated_state_summary": "<descriptive summary until the current training state in words | 'str' type>"
    }

    RULES:

    - Always read previous_state
    - Always use latest epoch log information
    - Make decision logically
    - Output JSON only
    """

    return system_instruction

@ara.tool
def utc_now():
    from datetime import datetime, timezone
    """provides the current UTC time"""
    return {"utc_time": datetime.now(timezone.utc).isoformat()}


@ara.tool
def read_train_logs():
    """returns the ml-model training-logs"""
    import json
    from urllib.request import urlopen
    
    print('realtime-train-logs-read')
    
    NGROK_TUNNEL = "https://eggshell-wrecking-jingle.ngrok-free.dev"
    api_path = f"{NGROK_TUNNEL}/training_logs"
    try:
        with urlopen(api_path) as response:
            train_logs_json = json.loads(response.read().decode())    
        train_logs_epochs = train_logs_json['epochs']
        train_logs_epochs = train_logs_epochs[-1] ## state-machine       
        return {'ml_model_training_logs': train_logs_epochs}
    
    except Exception as e:
        print("ACTUAL ERROR:", repr(e))
        raise e                        

@ara.tool
def read_previous_state_llm_summary():
    """returns previous state LLM summary/cache of the ML model training"""
    import json
    from urllib.request import urlopen
    
    NGROK_TUNNEL = "https://eggshell-wrecking-jingle.ngrok-free.dev"
    api_path = f"{NGROK_TUNNEL}/llm_cache"
    try:
        with urlopen(api_path) as response:
            logs_llm = json.loads(response.read().decode())    
        ## previous-state-cache
        return logs_llm
    
    except Exception as e:
        print("ACTUAL ERROR:", repr(e))
        raise e                        

@ara.tool
def retrieve_prior_run_context():
    """returns the prior agent run context"""
    import json
    from urllib.request import urlopen
    
    print('realtime-train-logs-read')
    
    NGROK_TUNNEL = "https://eggshell-wrecking-jingle.ngrok-free.dev"
    api_path = f"{NGROK_TUNNEL}/prior_context"
    try:
        with urlopen(api_path) as response:
            train_logs_context = json.loads(response.read().decode())    
        return train_logs_context
    
    except Exception as e:
        print("ACTUAL ERROR:", repr(e))
        raise e     

ara.Job(
    "ara-monitor-agent", ## triggers on train-errors
    system_instructions=(llm_caching_prompt())
)


if __name__ == '__main__':
    
    print('__running__ara/monitor___')
    
    from fastapi import FastAPI
    import threading
    import uvicorn
    
    api_app = FastAPI()
    
    ## GET-REST-call
    @api_app.get('/training_logs')
    def get_train_logs_api(train_path=TRAIN_LOG_FILE_PATH):
        import json
        with open(train_path, 'r') as f:
            train_logs_json = json.load(f)
        f.close()
            
        return train_logs_json
    
    @api_app.get('/prior_context')
    def get_prior_context_api(prior_context_path=PRIOR_CONTEXT_PATH):
        import json
        if os.path.exists(prior_context_path):
            with open(prior_context_path, 'rb') as f:
                prior_context = pkl.load(f)
            f.close()
        else:
            prior_context = ''
        return {'prior_context' : prior_context}
    
    @api_app.get('/llm_cache')
    def get_prior_context_api(decision_path=DECISION_LOG_SAVE_PATH):
        import json
        if os.path.exists(decision_path):
            with open(decision_path, 'r') as f:
                logs = json.load(f)['ara_monitor_decision']['result']['output_text']
                logs = json.loads(logs)
            f.close()
            llm_summ = logs['updated_state_summary']
        else:
            llm_summ = 'State-Just-Initialized'
        return {'previous_state_llm_summary_cache' : llm_summ}
        
    ## serve
    def api_serve(port=8000):
        print(f'serving-train-log-API at 0.0.0.0:{port}')
        def api_run():
            uvicorn.run(api_app, host="0.0.0.0", port=port)
        # start API in background thread
        threading.Thread(target=api_run, daemon=True).start()
        
    def run_adjoin_code_parallel(train_file_path=MODEL_TRAIN_FILE_PATH, stream_file_path=STREAM_FILE_PATH, is_train=True):
        if is_train:
            print('Training the model ...')
            subprocess.run(['python', train_file_path], check=True)
        else:
            print('loaded trained logs')
        
        subprocess.run(['streamlit', 'run', stream_file_path])
        
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
    def save_ara_logs(log_save_path=DECISION_LOG_SAVE_PATH, call_patience_threshold=3):
        """returns decision logs"""
        
        epoch_threshold = call_patience_threshold
        call_patience = 0
        
        ## runs N-cycles locally
        for i in range(20):
            
            out_stream = run_ara_monitor_subprocess()
            
            with open(log_save_path, 'w') as f:    
                json.dump(out_stream, f)
            f.close()
            
            print()
            print(f'saved-ara-out-decision-stream # {i+1}')
            print()
            
            out_stream_decision = clean_ara_stdout(out_stream["ara_monitor_decision"]["result"]["output_text"])
            
            ## min-epoch-execution-patience
            if (out_stream_decision['epoch'] > epoch_threshold) and (call_patience > call_patience_threshold): ## epoch-threshold (tune)            
                ## parallel-execution-trigger
                if out_stream_decision["is_trigger"]:
                    print('<Parallel> Triggered & Running ...')
                run_adjoin_code_parallel(is_train=False)
                call_patience = 0
            
            # run_deregister_cloud() ## cause the cloud registered state is fixed in the runtime
            call_patience += 1
            time.sleep(10) ## force 5mins retrieval-wait
    
    ## serve-local-api-endpoint
    api_serve() ## local:8000 port
    
    # ## ara-cloud-register
    run_ara_cloud_register(is_auth=True)

    # ## local-cyclic-monitoring
    save_ara_logs()
    
    run_deregister_cloud()

    
