from langchain_huggingface import HuggingFaceEmbeddings
import torch

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # chuẩn hóa để cosine similarity hoạt động tốt hơn
)