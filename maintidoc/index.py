"""Embeddings open source (BGE) + stockage dans PGVector via LlamaIndex."""
import os
from urllib.parse import urlparse

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maintidoc:maintidoc@localhost:5432/maintidoc")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # open source, 384 dimensions, tourne sur CPU
EMBED_DIM = 384
TABLE_NAME = "maintidoc_chunks"  # LlamaIndex crée la table "data_maintidoc_chunks"


def get_embed_model() -> BaseEmbedding:
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def get_vector_store() -> PGVectorStore:
    u = urlparse(DATABASE_URL)
    return PGVectorStore.from_params(
        host=u.hostname, port=u.port or 5432, user=u.username, password=u.password,
        database=u.path.lstrip("/"), table_name=TABLE_NAME, embed_dim=EMBED_DIM,
    )


def build_index(nodes, embed_model: BaseEmbedding | None = None) -> VectorStoreIndex:
    storage = StorageContext.from_defaults(vector_store=get_vector_store())
    return VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model or get_embed_model())
