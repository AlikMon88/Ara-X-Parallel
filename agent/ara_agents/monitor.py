### Continuous Ara - ML - Train - Monitoring agent // runs on ara cloud
import ara_sdk as ara
from ara_prompts import *

@ara.tool
def utc_now() -> dict:
    from datetime import datetime, timezone
    return {"utc_time": datetime.now(timezone.utc).isoformat()}

@ara.tool
def read_train_logs():
    """reads the ML-model"""
    print('realtime-train-logs')
    pass

def run_ara_monitor_agent():
    ara.Automation(
        "ara-monitor-agent", ## triggers on train-errors
        # system_instructions=set(get_prompt_e1)
        system_instructions=(
            "Read the current realtime ML-model training log and flag/trigger (in True/False) current include UTC time. in a json format of {utc_timestamp: , is_trigger: True/False}"
        ),
        # system_instructions=(
        #     "Reply with one short hello message and include UTC time. "
        # ),
        # tools=[utc_now],
        tools=[utc_now, read_train_logs]
    )

def save_ara_logs():
    """returns decision logs"""
    pass

if __name__ == '__main__':
    run_ara_monitor_agent()