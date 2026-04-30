import streamlit as st
from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *
from agent.logic_backend import *
from langchain_core.messages import HumanMessage, AIMessage
import sys
from . import prompts as pmp    
from pathlib import Path
import time
import signal
from agent.ara_agents.sms import *

_BASE_DIR_ = Path(__file__).parents[0]
STORE_FINAL_RESPONSE = _BASE_DIR_ / 'stream_output' / 'final_response.txt'
            
### Parallel-Stream-FrontEnd
def render_agent_stream(agent, messages):
    """
    Stream LangGraph agent execution
    with clean structured UI.
    """
    
    print('access-render-stream')

    final_report = None

    reasoning_blocks = []

    with st.status("Running <Parallel> ...", expanded=True) as status:

        for event in agent.stream({"messages": messages}):
            current_point = list(event.keys())[-1]
            print('current_point: ', current_point)
            
            event = event[current_point]
            
            if "messages" not in event:
                continue

            msg = event["messages"][-1] 
            
            print(msg)
            
            # TOOL CALL
            if hasattr(msg, "tool_calls") and msg.tool_calls:

                # tool_name = msg.tool_calls[0]["name"]
                for tool in msg.tool_calls: 
                    st.markdown(f""" Called : `{tool['name']}`""")
                
            # TOOL OUTPUT
            elif msg.type == "tool":
                st.markdown(f"""
                    ##### Tool: `{msg.name}`
                    """)

                with st.expander("Tool Output", expanded=False):

                    st.code(
                        msg.content[:2000],
                        language="text"
                    )
                    
                st.success("Tool Completed")

            # LLM REASONING
            elif msg.type == "ai":

                reasoning_blocks.append(msg.content)

                final_report = msg.content

        status.update(
            label="Debugging Complete",
            state="complete",
            expanded=False
        )
        
    print('RB-len: ', len(reasoning_blocks))

    # Show reasoning cleanly
    if reasoning_blocks:

        with st.expander(
            "View LLM Reasoning",
            expanded=False
        ):

            for r in reasoning_blocks:
            
                if len(reasoning_blocks) < 2:
                    st.markdown("No Explicit Reasoning.")
                    break
                
                st.markdown(f"""
                <div style="
                padding:12px;
                border-radius:8px;
                background-color:#0E1117;
                border:1px solid #333;
                margin-bottom:10px;
                ">

                {r}

                </div>
                """, unsafe_allow_html=True)

    return final_report

def stream_frontend_parallel(load_llm):
    st.set_page_config(page_title="<Parallel>", page_icon="⚙️")

    st.title(" << Parallel | Auto. ML SDG Assist. >> ")
    # st.markdown("Enter your ML pipeline issue below. The agent will route your query to the correct specialized database ([COMPUTE] / [DATA] / [CODE]) and retrieve historical fixes.")

    with st.spinner("Initializing LLM Model ..."):
        llm = load_llm.copy()

    if 'agent' not in st.session_state:
        ## RAG-agent-call / creates OOS states
        tools_pack = [read_training_logs, main_run_shap_analysis, search_db_files, main_search_framework_docs, evaluate_model_per_class, model_arch_info, data_distribution_understand]
        st.session_state.agent = get_debugging_agent(llm, tools_pack=tools_pack)
        st.session_state.chat_history = []
    
    if 'agent_call' not in st.session_state:
        st.session_state.agent_call = True   
        
        intitial_instruction = pmp.get_human_instruction_e4()
        
        with st.status("Parallel: AI Agent analyzing the recent training run ... ", expanded=True) as status:
            
            final_report = render_agent_stream(st.session_state.agent, [HumanMessage(content=intitial_instruction)])
            
            if final_report is None:
                print('No Final-Report Generated.')
                sys.exit()
            
            st.session_state.chat_history.append(AIMessage(content=final_report))
            with open(STORE_FINAL_RESPONSE, 'wb') as f:
                pkl.dump(final_report, f)
            f.close()
            print('saved-final-response')
            print('send-WA-condensed-report')
            send_WA_message()
            status.update(label="Diagnosis Complete!", state="complete", expanded=False)
                
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
    
    if 'agent_call' in st.session_state:    

        # HUMAN-IN-THE-LOOP CHAT
        if user_input := st.chat_input("Ask follow-up debugging questions..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            with st.spinner("Agent thinking..."):
                response = st.session_state.agent.invoke({"messages": st.session_state.chat_history})
                agent_reply = response["messages"][-1]
                st.chat_message("assistant").markdown(agent_reply.content)
                st.session_state.chat_history.append(agent_reply)

    if final_report:
        print('waiting-to-terminate-sesssion ....')
        time.sleep(60)
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    llm = load_llm(model_name='openai')
    
        
    