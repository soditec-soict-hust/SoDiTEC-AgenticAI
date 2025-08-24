import streamlit as st
import os

from src.LangGraph.ui.uiconfigfile import Config
from src.LangGraph.vectorstore.file_ingestion import ingest_uploaded_file
from src.LangGraph.vectorstore.vectorstore_manager import VectorStoreManager

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

                # Khởi tạo manager
                manager = VectorStoreManager()
                
                # Hiển thị lịch sử file đã upload
                st.subheader("📋 Previously Uploaded Files")
                active_files = manager.get_active_files()
                show_upload = True  # Mặc định hiển thị upload
                
                if active_files:
                    # Tạo options cho selectbox
                    file_options = ["Upload new file..."] + [f"{f['filename']} (ID: {f['id']})" for f in active_files]
                    selected_option = st.selectbox("Choose file or upload new:", file_options)
                    
                    if selected_option != "Upload new file...":
                        # Người dùng chọn file cũ
                        selected_id = int(selected_option.split("(ID: ")[1].split(")")[0])
                        selected_file = next(f for f in active_files if f['id'] == selected_id)
                        
                        self.user_controls["vectorstore_path"] = selected_file['vectorstore_path']
                        st.session_state["vectorstore_path"] = selected_file['vectorstore_path']
                        
                        st.success(f"✅ Using existing file: {selected_file['filename']}")
                        st.info(f"📂 Vectorstore path: {selected_file['vectorstore_path']}")
                        st.info(f"📅 Uploaded: {selected_file['upload_time']}")
                        
                        # Nút xóa file
                        if st.button(f"🗑️ Delete '{selected_file['filename']}'", key=f"delete_{selected_id}"):
                            if manager.delete_file_record(selected_file['vectorstore_path']):
                                st.success("File deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete file!")
                        
                        show_upload = False
                    else:
                        show_upload = True
                else:
                    st.info("No files uploaded yet. Please upload a file below.")
                    show_upload = True

                # Đặt vectorstore_path mặc định nếu chưa có
                if "vectorstore_path" not in self.user_controls:
                    self.user_controls["vectorstore_path"] = "src/LangGraph/vectorstore/store"

                # upload file for RAG (chỉ hiển thị khi cần)
                if show_upload:
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

                        # Tùy chọn thay thế hay giữ lại file cũ
                        replace_existing = st.checkbox("Replace existing vectorstore (recommended)", value=True, 
                                                     help="Uncheck to keep old data and add new data alongside")

                        vectorstore_path = "src/LangGraph/vectorstore/store"
                        
                        if st.button("Process File", use_container_width=True):
                            with st.spinner("🔄 Processing file and uploading to vectorstore..."):
                                result = ingest_uploaded_file(uploaded_file, self.user_controls["selected_embedding_model"], 
                                                            vectorstore_path, replace_existing)
                                if result and result != False:
                                    # Cập nhật vectorstore_path với path thực tế được sử dụng
                                    self.user_controls["vectorstore_path"] = result
                                    st.session_state["vectorstore_path"] = result  # Lưu vào session state
                                    
                                    # Xóa các cached objects khi có file mới
                                    if "rag_graph_builder" in st.session_state:
                                        del st.session_state["rag_graph_builder"]
                                    if "compiled_rag_graph" in st.session_state:
                                        del st.session_state["compiled_rag_graph"]
                                    if "current_vectorstore_path" in st.session_state:
                                        del st.session_state["current_vectorstore_path"]
                                        
                                    st.success(f"File processed and uploaded to vectorstore successfully! Store path: {result}")
                                    if replace_existing:
                                        st.info("🔄 Old data has been replaced with new file content.")
                                    else:
                                        st.info("➕ New content added alongside existing data.")
                                    st.info("Graph will be rebuilt with new vectorstore on next message.")
                                    st.rerun()  # Refresh để hiển thị file mới trong lịch sử
                                else:
                                    st.error("Failed to process and upload file to vectorstore.")

                            # Khởi tạo object embedding model và lưu vào self.user_controls
                            self.user_controls["embedding_model"] = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                encode_kwargs={"normalize_embeddings": True}  # chuẩn hóa để cosine similarity hoạt động tốt hơn
                            )

                else:
                    self.user_controls["uploaded_file"] = None
                    # Giữ nguyên vectorstore_path từ session state nếu đã có
                    if "vectorstore_path" in st.session_state:
                        self.user_controls["vectorstore_path"] = st.session_state["vectorstore_path"]
                    else:
                        self.user_controls["vectorstore_path"] = None 
        return self.user_controls