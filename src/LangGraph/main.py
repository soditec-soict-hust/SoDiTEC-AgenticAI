import streamlit as st
from src.LangGraph.ui.streamlitui.loadui import LoadStreamlitUI
from src.LangGraph.LLMs.groqllm import GroqLLM
from src.LangGraph.LLMs.openaillm import OpenAILLM
from src.LangGraph.graph.graph_builder import GraphBuilder, RAGGraphBuilder
from src.LangGraph.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """Load the LangGraph Agentic AI app with Streamlit UI."""
    UI = LoadStreamlitUI()
    user_input = UI.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load user inputs. Please check your configuration.")
        return
    
    if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.time_frame
    else:
        user_message = st.chat_input("Enter your message: ")

    if user_message:
        try:
            # Configure the LLM based on user input
            sellected_llm = user_input.get("selected_llm")
            
            ## Configure The LLM's
            if sellected_llm == "Groq":
                obj_llm_config=GroqLLM(user_controls_input=user_input)
            elif sellected_llm == "OpenAI":
                obj_llm_config=OpenAILLM(user_controls_input=user_input)
            else:
                st.error(f"Error: Unsupported LLM selected: {sellected_llm}")
                return

            model=obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: LLM model could not be initialized")
                return
            
            # Initialize and set up the graph based on use case
            usecase=user_input.get("selected_usecase")

            if not usecase:
                st.error("Error: No use case selected.")
                return
            
            ## Graph Builder
            if usecase == "Chatbot with RAG":
                vectorstore_path = user_input.get("vectorstore_path")
                embedding_model = user_input.get("embedding_model")
                if not embedding_model:
                    st.error("Error: Embedding model path")
                    return
                if not vectorstore_path:
                    st.error("Error: Vectorstore path")
                    return
                
                graph_builder = RAGGraphBuilder(model, vectorstore_path, embedding_model)
            else:
                graph_builder = GraphBuilder(model)

            try:
                graph = graph_builder.setup_graph(usecase)
                print(user_message)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph set up failed- {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph set up failed- {e}")
            return

    