"""Étape 5 : interface Streamlit de MaintiDoc."""
from pathlib import Path

import streamlit as st

from maintidoc.ingest import ingest_pdf
from maintidoc.rag import RagPipeline

UPLOAD_DIR = Path("data/uploads")

st.set_page_config(page_title="MaintiDoc", page_icon="🔧")
st.title("🔧 MaintiDoc")
st.caption("Posez une question sur vos documents techniques : les réponses citent leurs sources.")


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


with st.sidebar:
    st.header("1. Charger un PDF")
    uploaded = st.file_uploader("Document PDF", type="pdf")
    with_images = st.checkbox("Traiter les images (CLIP + descriptions, plus lent)", value=True)
    if uploaded and st.button("Indexer le document"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / uploaded.name
        path.write_bytes(uploaded.getvalue())
        with st.spinner("Extraction et indexation en cours..."):
            counts = ingest_pdf(path, with_images)
        st.success(f"Document indexé : {counts}")

st.header("2. Poser une question")
question = st.text_input("Votre question", placeholder="Que signifie le voyant orange ?")
if st.button("Envoyer") and question.strip():
    with st.spinner("Recherche et génération de la réponse..."):
        result = get_pipeline().answer(question)
    st.subheader("Réponse")
    st.write(result.text)
    st.subheader("Sources")
    for s in result.sources:
        with st.expander(f"[{s.ref}] page {s.page} · {s.type} · score {s.score:.2f}"):
            if s.image_path and Path(s.image_path).exists():
                st.image(s.image_path)
            st.text(s.text)
