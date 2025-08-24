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
                # Ưu tiên sử dụng vectorstore_path từ session state nếu có
                vectorstore_path = st.session_state.get("vectorstore_path") or user_input.get("vectorstore_path")
                embedding_model = user_input.get("embedding_model")
                if not embedding_model:
                    st.error("Error: Embedding model path")
                    return
                if not vectorstore_path:
                    st.error("Error: Vectorstore path - Please upload and process a file first")
                    return
                
                # Kiểm tra nếu vectorstore path đã thay đổi, cần tạo graph builder mới
                current_path = st.session_state.get("current_vectorstore_path")
                if current_path != vectorstore_path or "rag_graph_builder" not in st.session_state:
                    # Tạo graph builder mới khi path thay đổi hoặc chưa có
                    graph_builder = RAGGraphBuilder(model, vectorstore_path, embedding_model)
                    st.session_state["rag_graph_builder"] = graph_builder
                    st.session_state["current_vectorstore_path"] = vectorstore_path
                    # Xóa compiled graph cũ để tạo mới
                    if "compiled_rag_graph" in st.session_state:
                        del st.session_state["compiled_rag_graph"]
                else:
                    # Sử dụng lại graph builder hiện tại
                    graph_builder = st.session_state["rag_graph_builder"]
            else:
                graph_builder = GraphBuilder(model)

            try:
                # Sử dụng cached compiled graph nếu có (cho RAG case)
                if usecase == "Chatbot with RAG" and "compiled_rag_graph" in st.session_state:
                    graph = st.session_state["compiled_rag_graph"]
                else:
                    graph = graph_builder.setup_graph(usecase)
                    # Cache compiled graph cho RAG case
                    if usecase == "Chatbot with RAG":
                        st.session_state["compiled_rag_graph"] = graph
                
                print(user_message)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph set up failed- {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph set up failed- {e}")
            return

    