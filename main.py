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
TOP_K       = 3          # документов из векторного поиска
MAX_CONTEXT = 5          # максимум документов в промпте

# ─── Приложение ───────────────────────────────────────────────────
app = FastAPI(title="Музейный ИИ-ассистент")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ChromaDB ─────────────────────────────────────────────────────
collection  = None
chunk_cache = []   # список (doc, meta) для ключевого поиска

def load_db():
    global collection, chunk_cache
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("museum_exhibits")
        print(f"[OK] ChromaDB loaded: {collection.count()} fragments")
        
        db_data = collection.get()
        chunk_cache = list(zip(db_data["documents"], db_data["metadatas"]))
        print(f"[OK] Chunk cache: {len(chunk_cache)} chunks")
    except Exception as e:
        print(f"[WARN] ChromaDB not ready: {e}")
        collection = None
        chunk_cache = []

load_db()


# ─── Утилиты ──────────────────────────────────────────────────────
def keyword_search(query: str, exclude_chunks: set) -> list:
    """
    Ключевой поиск по фрагментам экспонатов.
    Возвращает (doc, meta) для фрагментов, в которых встречаются слова запроса.
    """
    # Слова длиннее 3 букв из запроса
    words = [w.lower() for w in re.findall(r"[а-яёa-z]+", query.lower()) if len(w) > 3]
    if not words:
        return []

    hits = []
    for doc, meta in chunk_cache:
        chunk_id = f"{meta['source']}_{meta['chunk']}"
        if chunk_id in exclude_chunks:
            continue
        text_lower = doc.lower()
        score = sum(text_lower.count(w) for w in words)
        if score > 0:
            hits.append((score, doc, meta))

    # Сортируем по убыванию совпадений
    hits.sort(key=lambda x: -x[0])
    result = []
    for score, doc, meta in hits[:3]:
        result.append((doc, meta))
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
    if not query:
        return {"answer": "Пустой запрос", "sources": []}

    # ── 1. Проверка на запрос каталога (Intent Routing)
    catalog_triggers = [
        "каталог", "весь список", "полный список", "какие экспонаты", 
        "какие картины", "какие в музее", "все картины", "все экспонаты"
    ]
    is_catalog_query = any(t in query.lower() for t in catalog_triggers)
    
    if is_catalog_query:
        exhibits_dict = {}
        for doc, meta in chunk_cache:
            if str(meta.get("chunk", "")) == "0":
                title_match = re.search(r"Название:\s*(.+)", doc)
                author_match = re.search(r"Автор:\s*(.+)", doc)
                if title_match:
                    title = title_match.group(1).strip()
                    if author_match:
                        author = author_match.group(1).strip()
                        exhibits_dict[meta["source"]] = f"«{title}» ({author})"
                    else:
                        exhibits_dict[meta["source"]] = f"«{title}»"
                else:
                    exhibits_dict[meta["source"]] = meta["exhibit"]
        exhibits_list = sorted(list(exhibits_dict.values()))
        exhibits_str = "\n".join(f"• {title}" for title in exhibits_list)
        return {
            "answer": f"В нашей коллекции представлены следующие произведения (картины, скульптуры и предметы):\n\n{exhibits_str}",
            "sources": ["Каталог музея"]
        }

    # ── 2. Формируем расширенный запрос с учетом контекста
    search_query = query
    if request.history:
        last_user = ""
        for msg in reversed(request.history):
            if msg.role == "user":
                last_user = msg.content
                break
                
        if last_user:
            search_query = f"{last_user} {query}"

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
    vec_chunks   = {f"{m['source']}_{m['chunk']}" for m in vec_metas}
    keyword_hits = keyword_search(search_query, vec_chunks)

    # ── 4. Объединяем: сначала векторные, потом ключевые (без дублей)
    all_docs   = list(zip(vec_docs, vec_metas))
    
    for doc, meta in keyword_hits:
        if not any(m["source"] == meta["source"] and m["chunk"] == meta["chunk"] for _, m in all_docs):
            all_docs.append((doc, meta))

    # Добавляем синтетический документ со списком всех экспонатов, 
    # чтобы LLM могла отвечать на общие вопросы о составе музея, не нарушая Правило 2.
    exhibits_dict = {}
    for doc, meta in chunk_cache:
        if str(meta.get("chunk", "")) == "0":
            title_match = re.search(r"Название:\s*(.+)", doc)
            author_match = re.search(r"Автор:\s*(.+)", doc)
            if title_match:
                title = title_match.group(1).strip()
                if author_match:
                    author = author_match.group(1).strip()
                    exhibits_dict[meta["source"]] = f"«{title}» ({author})"
                else:
                    exhibits_dict[meta["source"]] = f"«{title}»"
            else:
                exhibits_dict[meta["source"]] = meta["exhibit"]
                
    exhibits_list = sorted(list(exhibits_dict.values()))
    exhibits_str = "\n".join(f"- {title}" for title in exhibits_list)
    
    all_docs.append((
        f"Полный каталог музея (картины, скульптуры, предметы):\n{exhibits_str}", 
        {"source": "katalog_muzeya.txt", "exhibit": "Музей", "chunk": "0"}
    ))

    # ── 5. Формируем контекст
    print(f"\n[QUERY] {query!r}")
    for i, (doc, meta) in enumerate(all_docs):
        tag = "(vector)" if i < len(vec_docs) else "(keyword)"
        dist = vec_dists[i] if i < len(vec_dists) else 0.0
        print(f"  {i+1}. {dist:.3f} {tag} {meta['source']}_{meta['chunk']}")

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
        "4. Отвечай кратко, 1-4 предложения. (Исключение: если просят перечислить экспонаты/картины/скульптуры, смело выводи полный список).\n"
        "5. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование иероглифов и китайского языка категорически запрещено.\n\n"
        f"ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ МУЗЕЯ:\n{context}"
    )

    # ── 7. Собираем messages с историей
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (request.history or [])[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": query})

    # ── 8. Генерация
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Ошибка генерации: {e}")

    # ── 9. Источники — только при реальном ответе
    no_info = any(p in answer.lower() for p in [
        "нет информации", "не содержит", "не могу ответить",
        "специализируюсь только", "не связан с музеем"
    ])

    final_sources = []
    if not no_info and all_docs:
        # Умный фильтр: проверяем наличие хотя бы 1 общего корня слова (первые 4 буквы)
        ans_roots = set(w[:4] for w in re.findall(r"[а-яё]+", answer.lower()) if len(w) > 3)
        
        doc_intersections = []
        for doc, meta in all_docs:
            doc_roots = set(w[:4] for w in re.findall(r"[а-яё]+", doc.lower()) if len(w) > 3)
            intersect_count = len(ans_roots.intersection(doc_roots))
            doc_intersections.append((intersect_count, meta["source"]))
            
        # Сортируем документы по убыванию количества общих корней
        doc_intersections.sort(key=lambda x: x[0], reverse=True)
        
        # Для коротких ответов достаточно 1 общего корня
        best_count, best_source = doc_intersections[0]
        if best_count >= 1:
            final_sources.append(best_source)

    # ── 10. Пост-фильтрация галлюцинаций
    # Если нейросеть дала ответ, но ни один файл не прошел фильтр (нет общих слов), 
    # значит она выдумала ответ из головы (как в случае с Бэтменом или Windows).
    # Или если в ответе проскочил китайский иероглиф.
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', answer))
    
    if (not no_info and not final_sources) or has_chinese:
        answer = "К сожалению, в базе знаний музея нет информации по этому вопросу."
        final_sources = []

    return {
        "answer": answer,
        "sources": final_sources,
    }
