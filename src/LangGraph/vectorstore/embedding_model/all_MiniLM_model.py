from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class AllMiniLMModel:
    def __init__(self, model_name: str):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # Normalize for better cosine similarity
        )

    def get_embedding_model(self):
        return self.embedding_model