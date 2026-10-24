"""Étape 1 : vérifie que le PDF choisi contient bien du texte, des images et des tableaux."""
import sys
from pathlib import Path

import pymupdf

PDF_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/attention_is_all_you_need.pdf")


def main() -> None:
    doc = pymupdf.open(PDF_PATH)
    n_words = n_images = n_tables = 0
    print(f"Document : {PDF_PATH.name} ({doc.page_count} pages)\n")
    print(f"{'page':>4} {'mots':>6} {'images':>7} {'tableaux':>9}")
    for page in doc:
        words = len(page.get_text("words"))
        images = len(page.get_images(full=True))
        tables = len(page.find_tables().tables)
        n_words, n_images, n_tables = n_words + words, n_images + images, n_tables + tables
        print(f"{page.number + 1:>4} {words:>6} {images:>7} {tables:>9}")
    print(f"\nTotal : {n_words} mots, {n_images} images, {n_tables} tableaux détectés")


if __name__ == "__main__":
    main()
