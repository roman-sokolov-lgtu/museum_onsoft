"""
main.py — FastAPI-сервер музейного ИИ-ассистента (RAG).

Запуск: uvicorn main:app --reload
Открыть: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import chromadb
from pathlib import Path

# ─── Настройки ────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:3b"          # Легковесная модель для CPU
EMBED_MODEL = "nomic-embed-text"    # Модель эмбеддингов
CHROMA_PATH = "./chroma_db"         # Папка с базой ChromaDB
DISTANCE_THRESHOLD = 0.85           # Порог релевантности (cosine, 0–2)
TOP_K = 3                           # Сколько фрагментов извлекать

# ─── Приложение ───────────────────────────────────────────────────
app = FastAPI(title="Музейный ИИ-ассистент")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ChromaDB ─────────────────────────────────────────────────────
collection = None

def load_db():
    global collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("museum_exhibits")
        print(f"✅ ChromaDB загружена: {collection.count()} фрагментов")
    except Exception as e:
        print(f"⚠️  ChromaDB не готова: {e}")
        print("   Запустите индексацию: python index.py")
        collection = None

load_db()


# ─── Схемы ────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str


# ─── Роуты ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Отдаём HTML-страницу чата."""
    html_path = Path("index.html")
    if html_path.exists():
        return FileResponse("index.html", media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html не найден")


@app.get("/health")
async def health():
    """Проверка состояния сервисов."""
    db_ok = collection is not None
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok" if (db_ok and ollama_ok) else "partial",
        "database": "ok" if db_ok else "не проиндексирована (python index.py)",
        "ollama": "ok" if ollama_ok else "недоступна (ollama serve)",
        "chunks": collection.count() if db_ok else 0,
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Основной RAG-эндпоинт.
    1. Получает эмбеддинг вопроса
    2. Ищет релевантные фрагменты в ChromaDB
    3. Формирует промпт с контекстом
    4. Получает ответ от Ollama
    5. Возвращает ответ + список источников
    """

    # Проверяем готовность базы
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="База знаний не проиндексирована. Запустите: python index.py",
        )

    # 1. Эмбеддинг вопроса
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            embed_resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": request.query},
            )
            embed_resp.raise_for_status()
            query_embedding = embed_resp.json()["embedding"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama недоступна: {e}")

    # 2. Поиск в ChromaDB
    n_results = min(TOP_K, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Фильтрация по порогу релевантности
    relevant = [
        (doc, meta, dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if dist < DISTANCE_THRESHOLD
    ]

    if not relevant:
        return {
            "answer": (
                "К сожалению, в базе знаний музея нет информации по вашему вопросу. "
                "Попробуйте спросить об одном из экспонатов нашей коллекции."
            ),
            "sources": [],
        }

    # 4. Формируем контекст
    context_parts = []
    sources = []
    for doc, meta, dist in relevant:
        context_parts.append(f"[Источник: {meta['source']}]\n{doc}")
        if meta["source"] not in sources:
            sources.append(meta["source"])

    context = "\n\n---\n\n".join(context_parts)

    # 5. Промпт
    system_prompt = (
        "Ты — виртуальный гид музея. Отвечай СТРОГО на основе предоставленного контекста.\n"
        "ПРАВИЛА:\n"
        "- Если ответа нет в контексте — скажи: «К сожалению, в базе знаний музея нет информации по этому вопросу.»\n"
        "- Не придумывай факты. Не используй знания вне контекста.\n"
        "- Отвечай на русском языке. Ответ: 2–4 предложения.\n"
        "- Будь точен и информативен.\n\n"
        f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ МУЗЕЯ:\n{context}"
    )

    # 6. Генерация ответа
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            chat_resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            chat_resp.raise_for_status()
            answer = chat_resp.json()["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ошибка генерации: {e}")

    return {"answer": answer, "sources": sources}
