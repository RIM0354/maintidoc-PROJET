"""Traitement des images : embeddings CLIP + descriptions textuelles indexables."""
import os
from functools import lru_cache
from pathlib import Path

import ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from PIL import Image
from sentence_transformers import SentenceTransformer

from maintidoc.index import get_embed_model, get_vector_store

CLIP_MODEL = "clip-ViT-B-32"  # open source, image et texte dans le même espace (512 dim)
CLIP_DIM = 512
CLIP_TABLE = "maintidoc_images"  # table "data_maintidoc_images"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")  # modèle vision servi par Ollama
DESCRIBE_PROMPT = (
    "Describe this technical figure or photo in 2-4 sentences: the components, "
    "labels, arrows, indicator lights or values that are visible."
)


@lru_cache(maxsize=1)
def _clip() -> SentenceTransformer:
    return SentenceTransformer(CLIP_MODEL)


def clip_embed_image(path: str) -> list[float]:
    return _clip().encode(Image.open(path).convert("RGB"), normalize_embeddings=True).tolist()


def clip_embed_text(text: str) -> list[float]:
    return _clip().encode(text, normalize_embeddings=True).tolist()


def describe_image(path: str) -> str:
    resp = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": DESCRIBE_PROMPT, "images": [path]}],
    )
    return resp["message"]["content"].strip()


def index_images(image_nodes: list[TextNode], embed_model=None) -> None:
    """Pour chaque image : vecteur CLIP (table images) + description (table principale)."""
    clip_store = get_vector_store(CLIP_TABLE, CLIP_DIM)
    clip_nodes, desc_nodes = [], []
    for node in image_nodes:
        path = node.metadata["image_path"]
        clip_nodes.append(TextNode(
            text=node.text, metadata=node.metadata, embedding=clip_embed_image(path),
        ))
        description = describe_image(path)
        desc_nodes.append(TextNode(
            text=f"{node.text}\n{description}".strip(),
            metadata={**node.metadata, "type": "image_description", "description": description},
        ))
    clip_store.add(clip_nodes)
    VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=embed_model or get_embed_model()) \
        .insert_nodes(desc_nodes)


def search_images(query: str, top_k: int = 3) -> list[tuple[float, dict]]:
    """Recherche d'images par texte (ou de texte vers image) via CLIP."""
    result = get_vector_store(CLIP_TABLE, CLIP_DIM).query(
        VectorStoreQuery(query_embedding=clip_embed_text(query), similarity_top_k=top_k)
    )
    return list(zip(result.similarities or [], [n.metadata for n in result.nodes]))
