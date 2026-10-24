"""Extraction du PDF en nœuds LlamaIndex : texte, tableaux et images.

PyMuPDF lit le PDF ; LlamaIndex fournit les objets (TextNode) et le découpage
en passages (SentenceSplitter) utilisés ensuite pour l'indexation.
"""
import re
from pathlib import Path

import pymupdf
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

IMAGES_DIR = Path("data/images")
CAPTION_RE = r"^{kind}\s+(\d+)\s*:.*"


def _captions(page_text: str, kind: str) -> list[str]:
    return [m.group(0).strip() for m in re.finditer(CAPTION_RE.format(kind=kind), page_text, re.M)]


def extract_nodes(pdf_path: Path, images_dir: Path = IMAGES_DIR) -> list[TextNode]:
    doc = pymupdf.open(pdf_path)
    source = pdf_path.name
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    images_dir.mkdir(parents=True, exist_ok=True)
    nodes: list[TextNode] = []

    for page in doc:
        n = page.number + 1
        page_text = page.get_text()
        base = {"source": source, "page": n}

        # 1) Texte : découpé en passages de ~512 tokens
        for chunk in splitter.split_text(page_text):
            nodes.append(TextNode(text=chunk, metadata={**base, "type": "text"}))

        # 2) Tableaux : convertis en Markdown, un nœud par tableau
        captions = _captions(page_text, "Table")
        for i, table in enumerate(page.find_tables().tables):
            caption = captions[i] if i < len(captions) else ""
            nodes.append(TextNode(
                text=f"{caption}\n{table.to_markdown()}".strip(),
                metadata={**base, "type": "table", "caption": caption},
            ))

        # 3) Images : sauvegardées sur disque, indexées via leur légende
        captions = _captions(page_text, "Figure")
        for i, info in enumerate(page.get_images(full=True)):
            pix = pymupdf.Pixmap(doc, info[0])
            if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            path = images_dir / f"page{n:02d}_img{i + 1}.png"
            pix.save(path)
            caption = captions[i] if i < len(captions) else ""
            nodes.append(TextNode(
                text=caption or f"Image de la page {n}",
                metadata={**base, "type": "image", "image_path": str(path), "caption": caption},
            ))
    return nodes
