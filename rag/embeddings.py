from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=2)
def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
