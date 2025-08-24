from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import tempfile
from src.LangGraph.vectorstore.embedding_model.all_MiniLM_model import AllMiniLMModel

def ingest_uploaded_file(uploaded_file, embedding_model: str, vectorstore_path: str):
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

        # Xóa vectorstore cũ nếu tồn tại (với xử lý lỗi permission)
        import shutil
        import time
        if os.path.exists(vectorstore_path):
            try:
                shutil.rmtree(vectorstore_path)
                time.sleep(1)  # Đợi 1 giây để đảm bảo file được giải phóng
            except PermissionError:
                print(f"Warning: Cannot delete existing vectorstore at {vectorstore_path}. Using a new directory.")
                import uuid
                vectorstore_path = f"{vectorstore_path}_{uuid.uuid4().hex[:8]}"

        # Tạo vectorstore mới
        vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=vectorstore_path)

        if len(chunks):
            print(f"Successfully processed {len(chunks)} chunks from {uploaded_file.name} and stored in vectorstore.")
            return True
        else:
            return False

    finally:
        # Remove temporary file if it exists
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def get_retriever(vectorstore_path: str, embedding_model: str):
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 5})
