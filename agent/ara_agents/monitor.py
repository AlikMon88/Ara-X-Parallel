### Continuous Ara - ML - Train - Monitoring agent // runs on ara cloud
import ara_sdk as ara
# from . import ara_prompts as pmp  
import json
import subprocess
import time
from pprint import pprint
import re
import os

_cwd__ = os.getcwd()

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

def run_adjoin_code_parallel(train_file_path='model/sample_train_2.py', stream_file_path='main_stream.py', is_train=True):
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

@ara.tool
def read_train_logs(train_path='model/logs/training_logs.json'):
    import json
    """reads the ml-model training-logs"""
    print('realtime-train-logs')
    with open(train_path, 'r') as f:
        train_logs_json = json.load(f)
    f.close()
    return {'ml_model_training_logs': train_logs_json}

ara.Job(
    "ara-monitor-agent", ## triggers on train-errors
    system_instructions=(get_ara_prompt_e1())
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
                                     cwd=r"C:\Users\Alik\Desktop\M_1_year\Liquid-Net\AraXParallel-SDG") ## terminal/working directory-mismatch
    output_decision = clean_ara_stdout(output_decision.stdout)
    print(' --- output-decision --- ')
    pprint(output_decision)
    return {'ara_monitor_decision': output_decision}

def run_ara_cloud_register(is_auth=False):
    if is_auth:
        subprocess.run(['ara', 'auth', 'login'], 
                        cwd=r"C:\Users\Alik\Desktop\M_1_year\Liquid-Net\AraXParallel-SDG") ## terminal/working directory-mismatch
        print('ara-auth-login')
    subprocess.run(['ara', 'run', 'agent/ara_agents/monitor.py', '--cron', '"*/5 * * * *"'], 
                   capture_output=True, 
                   text=True,
                   cwd=r"C:\Users\Alik\Desktop\M_1_year\Liquid-Net\AraXParallel-SDG") ## terminal/working directory-mismatch
    print('registered-ara-cloud')

def run_deregister_cloud():
    subprocess.run(['ara', 'deploy', 'agent/ara_agents/monitor.py', '--activate', 'false'],
                   cwd=r"C:\Users\Alik\Desktop\M_1_year\Liquid-Net\AraXParallel-SDG")
    print('deregistered-ara-cloud')

## cyclic-run // trigger with patience (p)
def save_ara_logs(log_save_path='agent/ara_agents/logs/decision_logs.json'):
    """returns decision logs"""
    
    ## runs N-cycles locally
    for i in range(10):
        
        run_deregister_cloud()
        run_ara_cloud_register(is_auth=False)
        
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
    
    # ## ara-cloud-register
    run_ara_cloud_register(is_auth=True)

    # ## local-cyclic-monitoring
    save_ara_logs()
    
 