"""Étape 6 : question posée à l'agent LangChain (il choisit ses outils)."""
import sys

from maintidoc.agent import ask, build_agent

print(ask(build_agent(), " ".join(sys.argv[1:]) or "Which figure shows the Transformer architecture?"))
