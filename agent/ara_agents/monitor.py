### Continuous Ara - ML - Train - Monitoring agent // runs on ara cloud
import ara_sdk as ara
# from . import ara_prompts as pmp  
import json
import subprocess
import time

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

    TRIGGER CONDITIONS (examples):
    - Loss becomes NaN
    - Loss increases continuously
    - Accuracy drops significantly
    - Training diverges
    - Unexpected runtime errors appear

    OUTPUT FORMAT:
    Return ONLY valid JSON in the following schema:

    {"utc_timestamp": "<ISO-8601 UTC time>", "is_trigger": true or false}

    RULES:
    - Do not include explanations outside JSON.
    - Always include UTC timestamp.
    - Always return valid JSON.
    """
    
    return system_instructions

@ara.tool
def utc_now() -> dict:
    from datetime import datetime, timezone
    return {"utc_time": datetime.now(timezone.utc).isoformat()}

@ara.tool
def read_train_logs(train_path='model/logs/training_logs.json'):
    """reads the ml-model training-logs"""
    print('realtime-train-logs')
    train_logs_json = json.load(train_path)
    return {'ml_model_training_logs': train_logs_json}

ara.Automation(
    "ara-monitor-agent", ## triggers on train-errors
    system_instructions=get_ara_prompt_e1(),
    tools=[utc_now, read_train_logs]
)

## do a seperate one-off ara-cloud call cause logs streams don't work | This needs to be crontabbed/cycled locally
def run_ara_monitor_subprocess():
    output_decision = subprocess.run(['ara', 'run', 'agent/ara_agents/monitor.py'], capture_output=True, text=True)
    output_decision = json.loads(output_decision.stdout)
    return {'ara_monitor_decision': output_decision}

def run_ara_cloud_register():
    subprocess.run(['ara', 'auth', 'login'])
    print('ara-auth-login')
    subprocess.run(['ara', 'run', 'agent/ara_agents/monitor.py', '--cron', "*/5 * * * *"], capture_output=True, text=True)
    print('registered-ara-cloud')

## cyclic-run
def save_ara_logs(log_save_path='logs'):
    """returns decision logs"""
    ## runs N-cycles locally
    for i in range(100):
        out_stream = run_ara_monitor_subprocess()    
        json.dump(out_stream, log_save_path)
        print(f'saved-ara-out-decision-stream#{i+1}')
        time.sleep(30) ## 30-seconds

if __name__ == '__main__':
    print('__running__ara/monitor___')
    ## ara-cloud-register
    run_ara_cloud_register()
    ## local-cyclic-monitoring
    save_ara_logs()
    