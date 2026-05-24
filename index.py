# -*- coding: utf-8 -*-
"""
index.py -- Индексация текстов экспонатов в ChromaDB.

Запуск: python index.py

Что делает:
1. Читает все .txt файлы из папки content/
2. Получает эмбеддинги через Ollama (nomic-embed-text)
3. Сохраняет в ChromaDB (локальная папка chroma_db/)
"""

import httpx
import chromadb
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CONTENT_DIR = Path("content")
CHROMA_PATH = "./chroma_db"


def get_embedding(text: str) -> list:
    """Получить эмбеддинг текста через Ollama."""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chunk_text(text: str, max_words: int = 70) -> list[str]:
    """Разбить текст на абзацы, объединяя короткие."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())
        if current_words + words > max_words and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_words = words
        else:
            current_parts.append(para)
            current_words += words

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks if chunks else [text]


def main():
    print("=" * 50)
    print("  Muzejnyj II-assistent -- Indeksaciya")
    print("=" * 50)

    # Проверяем наличие папки content/
    if not CONTENT_DIR.exists():
        print(f"\n[ERR] Papka '{CONTENT_DIR}/' ne najdena.")
        print("   Sozdajte papku content/ i polozhite v nee .txt fajly.")
        return

    txt_files = sorted(CONTENT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"\n[ERR] V papke '{CONTENT_DIR}/' net .txt fajlov.")
        return

    print(f"\nНайдено файлов: {len(txt_files)}")

    # Проверяем Ollama
    print(f"\nProverka Ollama ({OLLAMA_URL})...")
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"  [OK] Ollama rabotaet. Modeli: {', '.join(models) if models else 'net'}")
        if EMBED_MODEL not in " ".join(models):
            print(f"\n  [WARN] Model '{EMBED_MODEL}' ne najdena!")
            print(f"     Zapustite: ollama pull {EMBED_MODEL}")
            return
    except Exception as e:
        print(f"  [ERR] Ollama nedostupna: {e}")
        print("     Ubedites chto Ollama zapushchena: ollama serve")
        return

    # Инициализация ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Пересоздаём коллекцию (чистая индексация)
    try:
        client.delete_collection("museum_exhibits")
        print("\nStaryj indeks udalyon.")
    except Exception:
        pass

    collection = client.create_collection(
        "museum_exhibits",
        metadata={"hnsw:space": "cosine"},
    )

    print("\nИндексирую файлы...\n")
    total_chunks = 0

    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8").strip()
        chunks = chunk_text(text)

        print(f"  [FILE] {filepath.name} -> {len(chunks)} fragment(ov)")

        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                doc_id = f"{filepath.stem}_{i}"
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": filepath.name,
                            "exhibit": filepath.stem.replace("_", " ").title(),
                            "chunk": i,
                        }
                    ],
                )
                total_chunks += 1
            except Exception as e:
                print(f"    [WARN] Oshibka fragment {i}: {e}")

    print(f"\n{'=' * 50}")
    print(f"[DONE] Proindeksirovano: {len(txt_files)} fajlov, {total_chunks} fragmentov.")
    print(f"   Baza sohranena v: {CHROMA_PATH}/")
    print(f"\nTeper zapustite server:")
    print(f"   uvicorn main:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()
