"""Orchestration agentique : un agent LangChain choisit ses outils de recherche."""
import os

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.rag import LLM_MODEL, RagPipeline

SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant pour techniciens de maintenance.
Tu disposes d'outils pour interroger l'index du document :
- search_documents : passages de texte et tableaux (à utiliser pour toute question factuelle) ;
- search_images : figures et photos (à utiliser quand la question porte sur un schéma ou une image).
Utilise les outils avant de répondre, reformule ta recherche si le résultat est insuffisant.
Réponds uniquement à partir des résultats, en citant la page, par exemple (p. 4).
Si l'information est absente, dis-le clairement."""


def build_tools(pipeline: RagPipeline) -> list:
    @tool
    def search_documents(query: str) -> str:
        """Cherche dans les textes et tableaux du document les passages liés à la requête."""
        sources = pipeline.retrieve(query)
        return "\n\n".join(f"(p. {s.page}, {s.type}) {s.text}" for s in sources) or "Aucun résultat."

    @tool
    def search_images(query: str) -> str:
        """Cherche dans les images et figures du document (recherche CLIP par description)."""
        from maintidoc.images import search_images as clip_search  # import tardif : CLIP est lourd

        hits = clip_search(query)
        return "\n".join(
            f"(p. {m['page']}) {m['caption'] or 'image'} -> {m['image_path']}" for _, m in hits
        ) or "Aucune image."

    return [search_documents, search_images]


def build_agent(pipeline: RagPipeline | None = None, model: BaseChatModel | None = None):
    pipeline = pipeline or RagPipeline()
    model = model or ChatOllama(model=LLM_MODEL, temperature=0.1)
    # InMemorySaver = mémoire de conversation : le contexte est conservé par thread_id
    return create_agent(model, build_tools(pipeline), system_prompt=SYSTEM_PROMPT,
                        checkpointer=InMemorySaver())


def ask(agent, question: str, thread_id: str = "default") -> str:
    out = agent.invoke({"messages": [{"role": "user", "content": question}]},
                       {"configurable": {"thread_id": thread_id}})
    return out["messages"][-1].content
