# MaintiDoc

Assistant destiné aux techniciens de maintenance. Il exploite des manuels, procédures, schémas et photographies pour répondre à des questions techniques avec des références vérifiables, peut consulter l'historique des interventions et préparer un ticket de maintenance (la création effective nécessite une validation humaine).

## Plan par étapes (une branche GitHub par étape)

| Étape | Contenu | Branche |
|---|---|---|
| 1 | Sélection du document | `etape-1-selection-document` |
| 2 | Extraction et indexation (LlamaIndex, embeddings open source, PGVector) | `etape-2-extraction-indexation` |
| 3 | Traitement des images (CLIP, descriptions textuelles) | `etape-3-images-clip` |
| 4 | Question-réponse RAG (Qwen via Ollama) | `etape-4-rag-qwen` |
| 5 | Interface Streamlit | `etape-5-streamlit` |
| 6 | Orchestration agentique (LangChain) | à venir |

## Étape 1 : document de test

`data/attention_is_all_you_need.pdf` (11 pages) sert de banc d'essai. Il contient du texte, des figures (architecture du Transformer, attention multi-têtes) et des tableaux (complexité, scores BLEU, variantes du modèle). Il sera remplacé plus tard par un vrai manuel de maintenance sans changer le pipeline.

```bash
pip install -r requirements.txt
python scripts/inspect_pdf.py
```

Résultat sur ce PDF : ~5000 mots, 3 images matricielles (pages 3 et 4), 3 tableaux détectés automatiquement (pages 6, 8, 9). L'article en compte 4 : l'un est mal détecté, ce qui illustre pourquoi l'étape 2 utilisera un parseur plus robuste.

## Étape 2 : extraction et indexation

```bash
docker compose up -d                 # PostgreSQL + extension pgvector
pip install -r requirements.txt
python scripts/index_pdf.py          # extrait, vectorise (BGE) et stocke dans PGVector
```

- `maintidoc/extract.py` : texte (passages de ~512 tokens), tableaux (en Markdown) et images (PNG dans `data/images/`) deviennent des nœuds LlamaIndex avec leurs métadonnées (source, page, type, légende).
- `maintidoc/index.py` : embeddings `BAAI/bge-small-en-v1.5` et stockage via `PGVectorStore`.

## Étape 3 : images (CLIP + descriptions)

```bash
ollama pull qwen2.5vl:3b             # modèle vision pour décrire les images
python scripts/index_images.py       # après scripts/index_pdf.py
```

- `maintidoc/images.py` : vecteurs CLIP (`clip-ViT-B-32`) dans la table `maintidoc_images` ; descriptions générées par le modèle vision d'Ollama, indexées avec BGE dans la table principale (type `image_description`).
- `search_images("texte")` retrouve une image à partir d'une requête texte.

## Étape 4 : question-réponse (RAG)

```bash
ollama pull qwen2.5:3b
python scripts/ask.py "How many attention heads does the base Transformer use?"
```

`maintidoc/rag.py` : `RagPipeline` récupère les 5 passages les plus proches dans PGVector, puis demande à Qwen de répondre uniquement à partir de ce contexte, en citant `[1]`, `[2]`… (page et type de chaque source). Le modèle se change avec la variable `LLM_MODEL`.

## Étape 5 : interface Streamlit

```bash
streamlit run app.py
```

Barre latérale : chargement du PDF et indexation (`maintidoc/ingest.py`). Page principale : saisie de la question, réponse et sources dépliables (avec l'image quand la source en est une).
