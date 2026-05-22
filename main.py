"""
main.py — FastAPI-сервер музейного ИИ-ассистента (RAG).

Запуск: uvicorn main:app --reload
Открыть: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import chromadb
import re
from pathlib import Path

# ─── Настройки ────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434"
CHAT_MODEL  = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
CONTENT_DIR = Path("content")
TOP_K       = 5          # документов из векторного поиска
MAX_CONTEXT = 8          # максимум документов в промпте

# ─── Приложение ───────────────────────────────────────────────────
app = FastAPI(title="Музейный ИИ-ассистент")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ChromaDB ─────────────────────────────────────────────────────
collection    = None
content_cache = {}   # filename -> text (для ключевого поиска)

def load_db():
    global collection, content_cache
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("museum_exhibits")
        print(f"[OK] ChromaDB loaded: {collection.count()} fragments")
    except Exception as e:
        print(f"[WARN] ChromaDB not ready: {e}")
        collection = None

    # Кэшируем тексты экспонатов для ключевого поиска
    if CONTENT_DIR.exists():
        for f in CONTENT_DIR.glob("*.txt"):
            content_cache[f.name] = f.read_text(encoding="utf-8")
        print(f"[OK] Content cache: {len(content_cache)} files")

load_db()


# ─── Утилиты ──────────────────────────────────────────────────────
def keyword_search(query: str, exclude_sources: set) -> list:
    """
    Ключевой поиск по текстам экспонатов.
    Возвращает (doc, meta) для файлов, в которых встречаются слова запроса.
    """
    # Слова длиннее 3 букв из запроса
    words = [w.lower() for w in re.findall(r"[а-яёa-z]+", query.lower()) if len(w) > 3]
    if not words:
        return []

    hits = []
    for fname, text in content_cache.items():
        if fname in exclude_sources:
            continue
        text_lower = text.lower()
        # Считаем сколько ключевых слов встречается в тексте
        score = sum(1 for w in words if w in text_lower)
        if score > 0:
            hits.append((score, fname, text))

    # Сортируем по убыванию совпадений
    hits.sort(key=lambda x: -x[0])
    result = []
    for score, fname, text in hits[:3]:
        result.append((text, {
            "source": fname,
            "exhibit": Path(fname).stem.replace("_", " ").title(),
            "chunk": 0,
        }))
    return result


# ─── Схемы ────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []


# ─── Роуты ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    html_path = Path("index.html")
    if html_path.exists():
        return FileResponse("index.html", media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html не найден")


@app.get("/health")
async def health():
    db_ok     = collection is not None
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp      = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "status":   "ok" if (db_ok and ollama_ok) else "partial",
        "database": "ok" if db_ok else "не проиндексирована (python index.py)",
        "ollama":   "ok" if ollama_ok else "недоступна (ollama serve)",
        "chunks":   collection.count() if db_ok else 0,
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Гибридный RAG-эндпоинт (векторный + ключевой поиск) с историей диалога.
    """
    if collection is None:
        raise HTTPException(status_code=503,
            detail="База знаний не проиндексирована. Запустите: python index.py")

    query = request.query.strip()

    # ── 1. Поисковый запрос: для коротких уточнений добавляем контекст из истории
    search_query = query
    if request.history:
        no_info = ["нет информации", "не содержит", "не могу ответить", "извините", "ошибкой", "не ясен"]
        last_bot = ""
        for m in reversed(request.history):
            if m.role == "assistant" and not any(p in m.content.lower() for p in no_info):
                last_bot = m.content
                break
        if last_bot and len(query.split()) <= 8:
            search_query = last_bot[:300] + " " + query

    # ── 2. Векторный поиск (TOP_K документов)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            embed_resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": search_query},
            )
            embed_resp.raise_for_status()
            query_embedding = embed_resp.json()["embedding"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama недоступна: {e}")

    n_results = min(TOP_K, collection.count())
    results   = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    vec_docs   = results["documents"][0]
    vec_metas  = results["metadatas"][0]
    vec_dists  = results["distances"][0]

    # ── 3. Ключевой поиск (дополнение к векторному)
    vec_sources  = {m["source"] for m in vec_metas}
    keyword_hits = keyword_search(query, vec_sources)

    # ── 4. Объединяем: сначала векторные, потом ключевые (без дублей)
    all_docs   = list(zip(vec_docs, vec_metas))
    all_docs  += keyword_hits
    all_docs   = all_docs[:MAX_CONTEXT]

    # Логируем
    print(f"\n[QUERY] {query!r}")
    for i, (doc, meta) in enumerate(all_docs):
        tag = "(vector)" if meta["source"] in vec_sources else "(keyword)"
        dist = vec_dists[i] if i < len(vec_dists) else 0.0
        print(f"  {i+1}. {dist:.3f} {tag} {meta['source']}")

    # ── 5. Формируем контекст
    context_parts = []
    all_sources   = []
    for doc, meta in all_docs:
        context_parts.append(f"[Источник: {meta['exhibit']}]\n{doc}")
        if meta["source"] not in all_sources:
            all_sources.append(meta["source"])

    context = "\n\n---\n\n".join(context_parts)

    # ── 6. Системный промпт (очень строгий)
    system_prompt = (
        "Ты — строгий музейный гид. Твоя единственная цель — искать ответы В ПРЕДОСТАВЛЕННОМ ТЕКСТЕ.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. Отвечай ТОЛЬКО используя текст фрагментов. Игнорируй свои внутренние знания.\n"
        "2. Если в тексте фрагментов НЕТ точного ответа на вопрос (даже если ты знаешь ответ сам) — ты ОБЯЗАН ответить ровно так: "
        "«К сожалению, в базе знаний музея нет информации по этому вопросу.»\n"
        "3. Запрещено рассуждать или додумывать. Если информации нет — применяй правило 2.\n"
        "4. Отвечай кратко, 1-4 предложения.\n\n"
        f"ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ МУЗЕЯ:\n{context}"
    )

    # ── 7. Собираем messages с историей
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (request.history or [])[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": query})

    # ── 8. Генерация
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            chat_resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.05},
                },
            )
            chat_resp.raise_for_status()
            answer = chat_resp.json()["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ошибка генерации: {e}")

    # ── 9. Источники — только при реальном ответе
    no_info = any(p in answer.lower() for p in [
        "нет информации", "не содержит", "не могу ответить",
        "специализируюсь только", "не связан с музеем"
    ])

    final_sources = []
    if not no_info and all_docs:
        # Для демо-версии с 3B моделью просто возвращаем самый релевантный документ (топ-1).
        # Это гарантирует, что мы не получим спам из 5-7 файлов из-за пересечения частых слов
        # типа "искусство", "произведение", "находится", "история".
        final_sources.append(all_docs[0][1]["source"])

    return {
        "answer": answer,
        "sources": final_sources,
    }
