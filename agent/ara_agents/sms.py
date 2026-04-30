from twilio.rest import Client
from dotenv import load_dotenv
import os
import pickle as pkl
from pathlib import Path
from pprint import pprint
from langchain_openai import ChatOpenAI
from functools import wraps

load_dotenv()

_BASE_DIR_ = Path(__file__).parents[2]
STORE_FINAL_RESPONSE = _BASE_DIR_ / "stream" / "stream_output" / "final_response.txt"

def load_llm():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY as an environment variable.")
        exit(1)
    print("Initializing LLM model ...")    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # gpt-4o-mini is fast and cheap
        
    return llm

def wrap_outer_prompt(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(STORE_FINAL_RESPONSE, 'rb') as f:
            final_response = pkl.load(f)

        final_response = func(final_response, *args, **kwargs)

        response = f"""
        Create a simple and Minimalistic Webpage to show the MODEL TRAINING DIAGNOSTIC AND ROOT CAUSE ANALYSIS using the provided information

        {final_response}

        Always Return the hosted webpage link
        """
        return response

    return wrapper

@wrap_outer_prompt
def get_condense_mssg(body_pass):
    llm = load_llm()
    message = f"You have this Final-Report \n\n\n {body_pass}. Extremely Condense it within 100 words (stay in MARKDOWN format) and keep all the relevant points intact. JUST RETURN DON'T SAY ANYTHING"
    body_pass = llm.invoke(message).content
    return body_pass

@wrap_outer_prompt
def get_raw_mssg(body_pass):
    return body_pass
    
    
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

def WA_Sandbox_run(body_pass): 
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH")
    cm_num = os.getenv("TWILIO_CM_NUM")
    wa_num = os.getenv("TWILIO_WA_NUM")
            
    client = Client(account_sid, auth_token)
    
    message = client.messages.create(
        body=body_pass,
        from_=f"whatsapp:+{wa_num}",   # Twilio sandbox number
        to=f"whatsapp:+{cm_num}"     # your own number
    )
    
    return message

def send_WA_message():
    response = get_condense_mssg()
    message = WA_Sandbox_run(body_pass=response)
    print("Message SID:", message.sid)
 
    
if __name__ == '__main__':
    send_WA_message()
    