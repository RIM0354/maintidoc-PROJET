"""Étape 2 : extrait le PDF, calcule les embeddings et les stocke dans PGVector."""
import sys
from pathlib import Path

# Python ne met que scripts/ dans le chemin d'import : on ajoute la racine du dépôt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import Counter

from maintidoc.extract import extract_nodes
from maintidoc.index import build_index, get_embed_model

pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "data/attention_is_all_you_need.pdf")
nodes = extract_nodes(pdf)
print("Nœuds extraits :", dict(Counter(n.metadata["type"] for n in nodes)))

embed_model = get_embed_model()
index = build_index(nodes, embed_model)
print(f"{len(nodes)} nœuds indexés dans PGVector.")

# Test de recherche par similarité
for q in ["How many attention heads does the base model use?", "BLEU score on English-to-German"]:
    print(f"\nQ: {q}")
    for r in index.as_retriever(similarity_top_k=3, embed_model=embed_model).retrieve(q):
        m = r.node.metadata
        print(f"  score={r.score:.3f} [{m['type']}] page {m['page']} : {r.node.text[:90]!r}")
