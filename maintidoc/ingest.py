"""Chaîne complète d'ingestion d'un PDF : extraction -> index texte -> images."""
from collections import Counter
from pathlib import Path

from maintidoc.extract import extract_nodes
from maintidoc.index import build_index, get_embed_model


def ingest_pdf(pdf_path: Path, with_images: bool = True) -> dict:
    nodes = extract_nodes(pdf_path)
    embed_model = get_embed_model()
    build_index(nodes, embed_model)
    if with_images:
        from maintidoc.images import index_images  # import tardif : CLIP est lourd
        index_images([n for n in nodes if n.metadata["type"] == "image"], embed_model)
    return dict(Counter(n.metadata["type"] for n in nodes))
