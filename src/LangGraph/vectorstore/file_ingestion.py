from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import tempfile
import uuid
from src.LangGraph.vectorstore.embedding_model.all_MiniLM_model import AllMiniLMModel
from src.LangGraph.vectorstore.vectorstore_manager import VectorStoreManager

def ingest_uploaded_file(uploaded_file, embedding_model: str, vectorstore_path: str, replace_existing: bool = True):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Sellect loader
        if suffix == ".txt":
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            raise ValueError("Unsupported file type")

        # Load and split documents
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(chunks)
        # Load embedding model
        embedding_model = AllMiniLMModel(embedding_model).get_embedding_model()
        
        # Khởi tạo manager
        manager = VectorStoreManager()

        # Xử lý vectorstore path
        if replace_existing:
            # Xóa vectorstore cũ nếu tồn tại
            import shutil
            import time
            if os.path.exists(vectorstore_path):
                try:
                    shutil.rmtree(vectorstore_path)
                    time.sleep(1)  # Đợi 1 giây để đảm bảo file được giải phóng
                    print(f"Deleted existing vectorstore at {vectorstore_path}")
                except PermissionError:
                    print(f"Warning: Cannot delete existing vectorstore at {vectorstore_path}. Using a new directory.")
                    vectorstore_path = f"{vectorstore_path}_{uuid.uuid4().hex[:8]}"
        else:
            # Tạo path mới với UUID để không ghi đè
            if os.path.exists(vectorstore_path):
                vectorstore_path = f"{vectorstore_path}_{uuid.uuid4().hex[:8]}"

        # Tạo vectorstore mới
        vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=vectorstore_path)

        if len(chunks):
            # Lưu thông tin vào manager
            manager.add_file_record(uploaded_file.name, vectorstore_path, "sentence-transformers/all-MiniLM-L6-v2")
            print(f"Successfully processed {len(chunks)} chunks from {uploaded_file.name} and stored in vectorstore.")
            return vectorstore_path  # Return actual path used
        else:
            return False

    finally:
        # Remove temporary file if it exists
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def get_retriever(vectorstore_path: str, embedding_model: str):
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 5})
