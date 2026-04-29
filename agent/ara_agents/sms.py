from twilio.rest import Client
from dotenv import load_dotenv
import os
import pickle as pkl
from pathlib import Path
from pprint import pprint
 
load_dotenv()

_BASE_DIR_ = Path(__file__).parents[2]
STORE_FINAL_RESPONSE = _BASE_DIR_ / "stream" / "stream_output" / "final_response.txt"

def wrap_outer_prompt():
    with open(STORE_FINAL_RESPONSE, 'rb') as f:
        final_response = pkl.load(f)
    f.close()
    response = f"""
    Create a simple and Minimalistic Webpage to show the MODEL TRAINING DIAGNOSTIC AND ROOT CAUSE ANALYSIS using the provided information \n\n
    {final_response} \n\n
    Always Return the hosted webpage link
    """
    return response

def call_free_twilio(body_pass):
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH")
    twilio_num = os.getenv("TWILIO_NUM")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body_pass,
        from_=f"+{twilio_num}",   # Your Twilio number
        to="+14155838940"         # Ara number
    )

    return message

if __name__ == '__main__':
    
    response = wrap_outer_prompt()
    print(response)
    
    # message = call_free_twilio(body_pass=response) 
    # print("Message SID:", message.sid)
    
    