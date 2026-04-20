import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *

from pprint import pprint 
from dotenv import load_dotenv
import sys

import streamlit as st
from stream.template import *

load_dotenv()

if __name__ == '__main__':
            
    
    ## Decoder LLM (TinyLLAMA/GPT-4o-mini)
    llm = load_llm(model_name='openai')   
    
    ### Streamlit / FrontEnd
    stream_frontend_parallel(load_llm=llm)
    # stream_frontend(load_vector_embed=vector_embed_cache, load_llm=llm)
    
    