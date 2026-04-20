from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

import streamlit as st

def route_query(query: str, llm) -> str:
    ## Categorizes the user's query
    
    system_prompt = """You are an intelligent routing agent for an ML system.
    Categorize the user's issue into exactly ONE of these categories:
    - [COMPUTE] (e.g., OOM, CUDA errors, hardware)
    - [DATA] (e.g., NaN loss, data drift, distribution shift)
    -[CODE] (e.g., syntax errors, tensor shape mismatches)
    
    Respond with ONLY the category tag (e.g., [DATA])."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    ## LCEL Chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()

## Agent-Inspection-RUN-tools
@tool
def query_grafana_metrics(model_id: str) -> str:
    """Use this to check live server metrics (CPU, RAM, VRAM, Latency) for a deployed model."""
    # Simulating an API call to a monitoring dashboard
    return f"Live Metrics for {model_id}: VRAM Usage 99% (15.8GB/16GB), Latency 2500ms, CPU Load 40%."

@tool
def run_shap_explainer(model_id: str) -> str:
    """Use this to analyze data drift or feature importance shifts for a model."""
    # Simulating a heavy ML inference job
    return f"SHAP Analysis for {model_id}: Feature 'user_age' drifted significantly. Null values detected in 15% of recent inference payloads."

def call_tools():    
    # Create a dictionary to easily map function names to actual functions
    TOOL_MAP = {
        "query_grafana_metrics": query_grafana_metrics,
        "run_shap_explainer": run_shap_explainer
    }
    
    return TOOL_MAP

def agent_tool_call(agent_decision):
    tool_results_list = []
    TOOL_MAP = call_tools()
    
    for tool in agent_decision.tool_calls:
        tool_name = tool['name']
        tool_args = tool['args']
        
        st.write(f"**Executing Tool:** `{tool_name}` with args: `{tool_args}`...")
        
        actual_tool = TOOL_MAP[tool_name]
        result = actual_tool.invoke(tool_args)
        tool_results_list.append(result)
        
        st.success(f'Tool-Output: {result}')
    
    live_tool_results = '\n'.join(tool_results_list)
    return live_tool_results