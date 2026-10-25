"""Étape 4 : pose une question au pipeline RAG depuis le terminal."""
import sys
from pathlib import Path

# Python ne met que scripts/ dans le chemin d'import : on ajoute la racine du dépôt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintidoc.rag import RagPipeline

question = " ".join(sys.argv[1:]) or "How many attention heads does the base Transformer use?"
result = RagPipeline().answer(question)
print(result.text, "\n\nSources :")
for s in result.sources:
    print(f"  [{s.ref}] page {s.page} ({s.type}) score={s.score:.3f}")
