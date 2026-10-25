"""Étape 3 : CLIP + descriptions pour les images du PDF (à lancer après index_pdf.py)."""
import sys
from pathlib import Path

# Python ne met que scripts/ dans le chemin d'import : on ajoute la racine du dépôt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintidoc.extract import extract_nodes
from maintidoc.images import describe_image, index_images, search_images

pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "data/attention_is_all_you_need.pdf")
image_nodes = [n for n in extract_nodes(pdf) if n.metadata["type"] == "image"]
index_images(image_nodes)
print(f"{len(image_nodes)} images indexées (CLIP + descriptions).")

for q in ["architecture diagram of the Transformer", "multi-head attention"]:
    print(f"\nQ: {q}")
    for score, meta in search_images(q):
        print(f"  score={score:.3f} page {meta['page']} {meta['image_path']} {meta['caption'][:60]!r}")
