import streamlit as st
from data.load import *
from rag.embed import *
from rag.retrieve import *
from agent.logic import *
from agent.logic_backend import *
from langchain_core.messages import HumanMessage, AIMessage
import sys

from . import prompts as pmp    

def stream_frontend(load_vector_embed, load_llm):
    st.set_page_config(page_title="ML-SYS-DBG", page_icon="⚙️")

    st.title(" < Autonomous ML Semantic-Debugging Assist. > ")
    st.markdown("Enter your ML pipeline issue below. The agent will route your query to the correct specialized database ([COMPUTE] / [DATA] / [CODE]) and retrieve historical fixes.")

    with st.spinner("Initializing AI Models and Vector Databases..."):
        vector_stores = load_vector_embed.copy()
        llm = load_llm.copy()
        llm_with_tools = llm.bind_tools([query_grafana_metrics, run_shap_explainer])

    # UI Form
    user_query = st.text_area("Describe your ML Issue:", placeholder="e.g., My training job keeps crashing with CUDA out of memory on step 500.")

    if st.button("Diagnose Issue"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            # Routing (Displaying the thought process)
            with st.status("Agent thinking...", expanded=True) as status:
                live_tool_results = 'Agent: No live tools data required.'
                
                st.write('Evaluating if live diagonistic tool are needed ...')
                agent_decision = llm_with_tools.invoke(user_query)
                
                if agent_decision.tool_calls:
                    live_tool_results = agent_tool_call(agent_decision)                    
                else:
                    st.write(live_tool_results)
                    
                st.write("1. Analyzing user intent...")
                category = route_query(user_query, llm)
                
                st.write(f"2. Routing to the **{category}** vector database...")
                selected_vs = vector_stores[category]
                  
                st.write("3. Retrieving historical context & generating diagnosis...")
                ## better to use a vector_embedd cache
                rag_chain = build_rag_chain(selected_vs, llm)
                response = rag_chain.invoke({"query": user_query, "live_metrics": live_tool_results})
                status.update(label="Diagnosis Complete!", state="complete", expanded=False)
            
            # Final Output
            st.subheader(" --- Diagnostic Report --- ")
            st.success(response)
            
            # Show a UI pill so recruiters see the exact tag it used
            st.caption(f"Routed via: `{category}` Database")
            
            
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
    st.set_page_config(page_title="Parallel | ML-SYS-DBG", page_icon="⚙️")

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
            
            ### Expand and discretize the tooling pipelining || show them in st.success green
            # response = st.session_state.agent.invoke({"messages": [HumanMessage(content=intitial_instruction)]})
            # final_report = response["messages"][-1].content
            
            ## UI-improve
            final_report = render_agent_stream(st.session_state.agent, [HumanMessage(content=intitial_instruction)])
            
            if final_report is None:
                print('No Final-Report Generated.')
                sys.exit()
            
            st.session_state.chat_history.append(AIMessage(content=final_report))
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



if __name__ == '__main__':
    llm = load_llm(model_name='openai')
    
    