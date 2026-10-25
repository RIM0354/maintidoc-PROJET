"""Étape 6 : question posée à l'agent LangChain (il choisit ses outils)."""
import sys
from pathlib import Path

# Python ne met que scripts/ dans le chemin d'import : on ajoute la racine du dépôt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintidoc.agent import ask, build_agent

print(ask(build_agent(), " ".join(sys.argv[1:]) or "Which figure shows the Transformer architecture?"))
