"""Pipeline RAG : récupération des passages pertinents + réponse de Qwen (via Ollama)."""
import os
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

from maintidoc.index import get_embed_model, get_vector_store

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

PROMPT = """Tu es MaintiDoc, un assistant pour techniciens de maintenance.
Réponds à la question en utilisant UNIQUEMENT le contexte ci-dessous.
Cite chaque information avec sa référence, par exemple [1] ou [2].
Si le contexte ne contient pas la réponse, dis-le clairement au lieu d'inventer.

Contexte :
{context}

Question : {question}
Réponse :"""


@dataclass
class Source:
    ref: int
    type: str
    page: int
    text: str
    score: float | None
    image_path: str | None = None


@dataclass
class Answer:
    text: str
    sources: list[Source]


class RagPipeline:
    def __init__(self, llm: LLM | None = None, embed_model: BaseEmbedding | None = None, top_k: int = 5):
        self.embed_model = embed_model or get_embed_model()
        self.llm = llm or Ollama(model=LLM_MODEL, request_timeout=300, temperature=0.1)
        index = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=self.embed_model)
        self.retriever = index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, question: str) -> list[Source]:
        return [
            Source(i, r.node.metadata["type"], r.node.metadata["page"], r.node.text, r.score,
                   r.node.metadata.get("image_path"))
            for i, r in enumerate(self.retriever.retrieve(question), start=1)
        ]

    def answer(self, question: str) -> Answer:
        sources = self.retrieve(question)
        context = "\n\n".join(f"[{s.ref}] (page {s.page}, {s.type})\n{s.text}" for s in sources)
        reply = self.llm.complete(PROMPT.format(context=context, question=question))
        return Answer(reply.text.strip(), sources)
