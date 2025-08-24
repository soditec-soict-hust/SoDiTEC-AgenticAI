import streamlit as st
import os

from src.LangGraph.ui.uiconfigfile import Config
from src.LangGraph.vectorstore.file_ingestion import ingest_uploaded_file

from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}
    
    def load_streamlit_ui(self):
        st.set_page_config(page_title="🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())
        st.session_state.timeframe = ''
        st.session_state.IsFetchButtonClicked = False


        with st.sidebar:
            # Get options from config
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            # User choose Groq LLM 
            if self.user_controls["selected_llm"] == "Groq":
                groq_model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", groq_model_options)
                self.user_controls["API_KEY"] = st.session_state["API_KEY"] = st.text_input("API Key", type="password")
                # Validate API key
                if not self.user_controls["API_KEY"]:
                    st.warning("Please enter your API key to proceed.")
            elif self.user_controls["selected_llm"] == "OpenAI":
                openai_model_options = self.config.get_openai_model_options()
                self.user_controls["selected_openai_model"] = st.selectbox("Select Model", openai_model_options)
                self.user_controls["API_KEY"] = st.session_state["API_KEY"] = st.text_input("API Key", type="password")
                # Validate API key
                if not self.user_controls["API_KEY"]:
                    st.warning("Please enter your API key to proceed.")
            # Usecase selection
            self.user_controls["selected_usecase"] = st.selectbox("Select Usecases", usecase_options)

            if self.user_controls["selected_usecase"] == "Chatbot With WebTool" or self.user_controls["selected_usecase"] == "AI News" or self.user_controls["selected_usecase"] == "Chatbot with RAG":
                os.environ["TAVILY_API_KEY"] = self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"] = st.text_input("TAVILY API Key", type="password")
                # Validate TAVILY API key
                if not self.user_controls["TAVILY_API_KEY"]:
                    st.warning("Please enter your TAVILY API key to proceed.")

            if self.user_controls["selected_usecase"] == "AI News":
                st.subheader("AI News Explorer ")

                with st.sidebar:
                    time_frame = st.selectbox(
                        "Select Time Frame", 
                        ["Daily", "Weekly", "Monthly", "Yearly"],
                        index=0
                    )
                if st.button("Fetch AI News", use_container_width=True):
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.time_frame = time_frame
            
            if self.user_controls["selected_usecase"] == "Chatbot with RAG":
                # Sellect embedding model
                embedding_model_options = self.config.get_embedding_model_options()
                self.user_controls["selected_embedding_model"] = st.selectbox("Select Embedding Model", embedding_model_options)

                # Khởi tạo embedding model ngay khi chọn usecase RAG
                self.user_controls["embedding_model"] = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )

                # Đặt vectorstore_path mặc định
                self.user_controls["vectorstore_path"] = "src/LangGraph/vectorstore/store"

                # upload file for RAG
                st.subheader("📂 Upload Knowledge Files for RAG")

                uploaded_file = st.file_uploader(
                    "Upload a file (PDF, DOCX, or TXT)",
                    type=["pdf", "docx", "txt"],
                    help="Upload a document to serve as knowledge base for RAG")
                
                # Process uploaded file
                if uploaded_file is not None:
                    self.user_controls["uploaded_file"] = uploaded_file
                    file_details = {
                        "filename": uploaded_file.name,
                        "filetype": uploaded_file.type,
                        "filesize": uploaded_file.size
                    }
                    st.success(f"Uploaded {file_details['filename']} successfully!")

                    vectorstore_path = "src/LangGraph/vectorstore/store"
                    if st.button("Process File", use_container_width=True):
                        with st.spinner("🔄 Processing file and uploading to vectorstore..."):
                            result = ingest_uploaded_file(uploaded_file, self.user_controls["selected_embedding_model"], vectorstore_path)
                            if result:
                                st.success("File processed and uploaded to vectorstore successfully!")
                            else:
                                st.error("Failed to process and upload file to vectorstore.")
                        self.user_controls["vectorstore_path"] = vectorstore_path

                        # Khởi tạo object embedding model và lưu vào self.user_controls
                        self.user_controls["embedding_model"] = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                            encode_kwargs={"normalize_embeddings": True}  # chuẩn hóa để cosine similarity hoạt động tốt hơn
                        )

                else:
                    self.user_controls["uploaded_file"] = None
                    self.user_controls["vectorstore_path"] = None
                    # Giữ nguyên vectorstore_path nếu đã upload và process file trước đó 
        return self.user_controls