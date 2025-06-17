import os
import json
import faiss
import re
import time
import uuid
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import threading
from threading import Lock
import uvicorn
import hashlib
import spacy

CHUNKS_FILE = "chunks.jsonl"
PROCESSED_URLS_FILE = "processed_urls.txt"
PROCESSED_PDFS_FILE = "processed_pdfs.txt"
CHAT_HISTORY_FILE = "chat_history.jsonl"
PDF_DIR = "pdfs"
URLS_FILE = "urls.txt"
INDEX_FILE = "faiss_index.bin"
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REFRESH_INTERVAL = 3600  # 1 hour
FAILED_URLS_FILE = "failed_urls.txt"
DUPLICATE_URLS_FILE = "duplicate_urls.txt"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# --- Init ---
model = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
chunk_texts = []
chunk_sources = []
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
data_lock = Lock()
nlp = spacy.load("en_core_web_sm")  # For sentence splitting and NER

# For content-based deduplication (optional)
seen_content = {}

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stores selected model per session_id
session_model_context = {}
# Create a new session for each app run
ACTIVE_SESSION_ID = str(uuid.uuid4())
print(f"[+] New session started: {ACTIVE_SESSION_ID}")

# --- Utilities ---
def sanitize_string(s):
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)

def clean_response(response):
    return re.sub(r'\s+', ' ', response.replace("\n", " ").replace("\t", " ")).strip()

def smart_chunk_text(text, chunk_size=500, overlap=0):
    """Sentence-level chunking using spaCy"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < chunk_size:
            current_chunk += " " + sent if current_chunk else sent
        else:
            chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def save_chunks(chunks, source):
    with open(CHUNKS_FILE, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({"chunk": chunk, "source": source}) + "\n")

def embed_and_index_chunks(chunks, source):
    global chunk_texts, chunk_sources
    embeddings = model.encode(["passage: " + c for c in chunks])
    with data_lock:
        index.add(np.array(embeddings))
        chunk_texts.extend(chunks)
        chunk_sources.extend([source] * len(chunks))
    save_chunks(chunks, source)
    save_faiss_index()

def save_faiss_index():
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index with {index.ntotal} embeddings")

def load_faiss_index():
    if os.path.exists(INDEX_FILE):
        try:
            loaded_index = faiss.read_index(INDEX_FILE)
            print(f"Loaded FAISS index with {loaded_index.ntotal} embeddings")
            return loaded_index
        except Exception as e:
            print(f"[!] Failed to load FAISS index: {e}")
            return None
    return faiss.IndexFlatL2(embedding_dim)

def fetch_clean_text_from_url(url, max_retries=3, backoff_factor=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/113.0.0.0 Safari/537.36"
    }
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, headers=headers, timeout=30)
            if res.status_code == 200:
                soup = BeautifulSoup(res.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                clean_text = re.sub(r'\s+', ' ', text).strip()
                if len(clean_text) < 50:  # content too small, likely useless
                    print(f"[!] Skipping URL {url}: extracted text too small")
                    append_processed(FAILED_URLS_FILE, url + " (too small or empty)")
                    return ""
                return clean_text
            else:
                print(f"[!] Non-200 status code {res.status_code} for {url} (attempt {attempt})")
        except Exception as e:
            print(f"[!] Attempt {attempt} failed for {url}: {e}")
        if attempt < max_retries:
            time.sleep(backoff_factor * (2 ** (attempt - 1)))
    append_processed(FAILED_URLS_FILE, url + " (unreachable)")
    return ""

def extract_links_from_page(url):
    try:
        res = requests.get(url, timeout=30)
        soup = BeautifulSoup(res.content, "html.parser")
        return [tag['href'] for tag in soup.find_all("a", href=True) if tag['href'].startswith("http")]
    except Exception as e:
        print(f"[!] Failed to extract links from {url}: {e}")
        with open(FAILED_URLS_FILE, "a") as f:
            f.write(url + "\n")
        return []

def load_processed(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []

def append_processed(file_path, item):
    with open(file_path, "a") as f:
        f.write(item + "\n")

def append_duplicate(url, reason=""):
    with open(DUPLICATE_URLS_FILE, "a") as f:
        f.write(url + (f" ({reason})" if reason else "") + "\n")

def background_refresh():
    while True:
        time.sleep(REFRESH_INTERVAL)
        print("[*] Background refresh started")
        process_urls()
        process_pdfs()
        save_faiss_index()

def check_duplicates(file_path):
    if not os.path.exists(file_path):
        print(f"[!] {file_path} not found. Skipping duplicate check.")
        return
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    unique = set(lines)
    print(f"Total lines: {len(lines)}")
    print(f"Unique lines: {len(unique)}")
    duplicates = [url for url in lines if lines.count(url) > 1]
    duplicates = list(set(duplicates))
    print(f"Duplicate URLs: {duplicates}")
    for url in duplicates:
        append_duplicate(url, "historical duplicate")

def load_conversation_history(session_id, max_history=3):
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in reversed(f.readlines()):
                entry = json.loads(line)
                if entry['session_id'] == session_id:
                    history.insert(0, entry)
                    if len(history) >= max_history:
                        break
    return history

def save_chat_history(session_id, question, response, clarification=False, original_question=None):
    entry = {
        "session_id": session_id,
        "question": question,
        "response": response,
        "clarification": clarification,
        "original_question": original_question,
        "timestamp": time.time()
    }
    with open(CHAT_HISTORY_FILE, "a", encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")

def is_complex_question(question):
    return ("difference between" in question.lower() or
            "compare" in question.lower() or
            "versus" in question.lower() or
            "vs" in question.lower() or
            "explain how" in question.lower())

def extract_model_names(text):
    # Improved regex to capture model patterns like DLOS8N, RS485-LN, LHT65, etc.
    return set(re.findall(r'\b[A-Z0-9]+(?:-[A-Z0-9]+)*\b', text))

def get_device_and_models(question, context_chunks):
    device_mentions = set()
    model_mentions = set()
    model_mentions.update(extract_model_names(question))
    for chunk in context_chunks:
        model_mentions.update(extract_model_names(chunk["chunk"]))
    device = None
    if model_mentions:
        prefixes = [m.split("-")[0] for m in model_mentions]
        if len(set(prefixes)) == 1:
            device = prefixes[0]
    return device, model_mentions

def generate_disambiguation_prompt(model_mentions, question):
    model_list = sorted(model_mentions)
    return f"""Your question mentions a device type with multiple models: {model_list}
Please specify which model you meant so I can provide a precise answer to your question: "{question}"

Which model are you referring to? (Please type the model name)
"""

def is_model_name(text, model_mentions):
    return text.strip().upper() in {m.upper() for m in model_mentions}

def load_chunks():
    global chunk_texts, chunk_sources
    chunk_texts.clear()
    chunk_sources.clear()
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunk_texts.append(data['chunk'])
                chunk_sources.append(data['source'])
        print(f"[!] Loaded {len(chunk_texts)} chunks from {CHUNKS_FILE}")

def compute_quality_metrics(question, context_chunks, answer):
    # Semantic similarity: max cosine similarity between question and any top chunk
    question_embedding = model.encode(["query: " + question])
    top_chunks = [c["chunk"] for c in context_chunks[:5]]

    if not top_chunks:
        return {
            "semantic_similarity": 0.0,
            "comprehensiveness": 0.0,
            "competence": 0.0,
            "accuracy": 0.0
        }

    chunk_embeddings = model.encode(["passage: " + c for c in top_chunks])
    sim_scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
    semantic_similarity = float(np.max(sim_scores))

    # Comprehensiveness: % of question entities captured in answer
    question_entities = get_entities(question)
    answer_entities = get_entities(answer)
    overlap = len(question_entities & answer_entities)
    comprehensiveness = min(1.0, overlap / max(1, len(question_entities)))

    # Competence: does the answer reflect the content chunks?
    chunk_keywords = " ".join(top_chunks).lower()
    answer_lower = answer.lower()
    competence = float(any(word in chunk_keywords for word in answer_lower.split()))

    # Accuracy: average of all three
    accuracy = round((semantic_similarity + comprehensiveness + competence) / 3, 3)

    return {
        "semantic_similarity": round(semantic_similarity, 3),
        "comprehensiveness": round(comprehensiveness, 3),
        "competence": round(competence, 3),
        "accuracy": accuracy
    }

def reindex_all_chunks():
    global index, chunk_texts, chunk_sources
    if not chunk_texts:
        return
    index = faiss.IndexFlatL2(embedding_dim)
    for i in range(0, len(chunk_texts), 1000):
        batch = chunk_texts[i:i+1000]
        embeddings = model.encode(["passage: " + b for b in batch])
        index.add(np.array(embeddings))
    print(f"Reindexed {len(chunk_texts)} chunks into FAISS index")
    save_faiss_index()

def process_urls():
    seen_urls = set()
    duplicate_urls = set()

    if not os.path.exists(URLS_FILE):
        print("[!] urls.txt not found")
        return

    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"[*] Processing {len(urls)} URLs (including potential duplicates)")

    processed_urls = set(load_processed(PROCESSED_URLS_FILE))
    failed_urls = set(load_processed(FAILED_URLS_FILE))

    for url in urls:
        print(f"[+] Processing URL: {url}")

        if url in seen_urls:
            print(f"Duplicate detected in current batch: {url}")
            duplicate_urls.add(url)
            append_duplicate(url, "duplicate in current batch")
            append_processed(FAILED_URLS_FILE, url + " (duplicate)")
            continue

        seen_urls.add(url)

        if url in processed_urls or url in failed_urls:
            continue

        text = fetch_clean_text_from_url(url)
        if text:
            content_id = hashlib.sha256(text.encode()).hexdigest()
            if content_id in seen_content:
                append_duplicate(url, f"content duplicate of {seen_content[content_id]}")
                append_processed(FAILED_URLS_FILE, url + " (content duplicate)")
                continue
            seen_content[content_id] = url

            chunks = smart_chunk_text(text)
            embed_and_index_chunks(chunks, url)
            append_processed(PROCESSED_URLS_FILE, url)
        else:
            append_processed(FAILED_URLS_FILE, url + " (no content)")

    print("[*] Checking historical duplicates...")
    if os.path.exists(PROCESSED_URLS_FILE):
        with open(PROCESSED_URLS_FILE, 'r', encoding='utf-8') as f:
            all_urls = [line.strip() for line in f]
    else:
        all_urls = []

    historical_seen = set()
    with open(PROCESSED_URLS_FILE, "w") as f:
        for url in all_urls:
            if url not in historical_seen:
                f.write(url + "\n")
                historical_seen.add(url)
            else:
                duplicate_urls.add(url)
        append_duplicate(url, "historical duplicate")

    print(f"[*] Found {len(duplicate_urls)} duplicates")

def process_pdfs():
    processed = load_processed(PROCESSED_PDFS_FILE)
    if not os.path.exists(PDF_DIR):
        print("[!] PDF directory not found.")
        return
    for fname in os.listdir(PDF_DIR):
        if not fname.endswith(".pdf") or fname in processed:
            continue
        path = os.path.join(PDF_DIR, fname)
        try:
            reader = PdfReader(path)
            heading = ""
            for page in reader.pages:
                lines = (page.extract_text() or "").splitlines()
                heading = next((line for line in lines if len(line.strip()) > 5), "")
                if heading:
                    break
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            text = re.sub(r'\s+', ' ', text)
            print(f"[+] Processing PDF: {fname}")
            print(f"Extracted text length: {len(text)}")
            chunks = [f"{heading} — {c}" for c in smart_chunk_text(text)]
            print(f"Generated chunks: {len(chunks)}")
            embed_and_index_chunks(chunks, fname)
            append_processed(PROCESSED_PDFS_FILE, fname)
        except Exception as e:
            print(f"[!] Failed to process {fname}: {e}")

def get_entities(text):
    """Extract named entities from text using spaCy"""
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

def prioritize_entity_overlap(chunks, query_entities):
    """Sort chunks by number of overlapping named entities with query"""
    scored = []
    for chunk in chunks:
        chunk_entities = get_entities(chunk["chunk"])
        overlap = len(query_entities & chunk_entities)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: -x[0])
    return [chunk for (overlap, chunk) in scored]

def rewrite_ambiguous_query(question, context_chunks):
    """Rule-based query rewriting for ambiguous queries"""
    if "difference between" in question.lower():
        return question.replace("difference between", "compare")
    return question

def retrieve_relevant_chunks(question, session_id, top_k=15):
    """Retrieve and filter chunks based on model and entity overlap"""
    query_embedding = model.encode(["query: " + question])
    distances, indices = index.search(np.array(query_embedding), top_k * 2)
    results = []

    # Get current model context
    model_filter = session_model_context.get(session_id)
    # Only apply model filter if:
    # 1. Question explicitly mentions a model, OR
    # 2. User is asking for model-specific docs
    use_model_filter = model_filter and (
        bool(extract_model_names(question)) or
        any(keyword in question.lower() for keyword in ["datasheet", "manual", "specs"])
    )

    print("\n[DEBUG] Retrieved chunks for:", question)
    for i in indices[0]:
        if 0 <= i < len(chunk_texts):
            chunk = chunk_texts[i]
            source = chunk_sources[i]
            if use_model_filter and model_filter.lower() not in chunk.lower():
                continue
            if len(chunk) > 50:
                results.append({"chunk": chunk, "source": source})

    # Prioritize chunks with entity overlap
    query_entities = get_entities(question)
    results = prioritize_entity_overlap(results, query_entities)

    return results[:top_k]

def build_prompt(question, context_chunks, conversation_history):
    # Format conversation history
    history = "\n".join([
        f"User: {entry['question']}\nAssistant: {entry['response']}"
        for entry in conversation_history
    ])
    # Format knowledge base context
    context = "\n\n".join([f"Source: {c['source']}\nContent: {c['chunk']}" for c in context_chunks])
    # Build prompt
    return f"""Consider the following conversation history:
{history}

Relevant knowledge base context:
{context}

Current question: {question}

Rules:
- If the answer is explicitly stated in the context, respond with it directly.
- Do not mention the source.
- If the answer is not explicitly stated but can be inferred(mention it only once) from patterns, comparisons, or multiple data points, then reason through the context and generate a logically sound answer.
- For comparison or difference questions, analyze the attributes of each item separately, then explain how they differ.
- Respond even if no sentence in the data directly answers the question, as long as the information is sufficient to deduce a meaningful answer.
- Never fabricate information not present in or inferable from the context.
- Keep the tone factual and analytical.... Answer:
"""

def needs_clarification(results):
    model_names = set()
    for chunk in results:
        model_names.update(re.findall(r"\b[A-Z0-9]+-[A-Z0-9]+\b", chunk["chunk"]))
    return len(model_names) > 1

def generate_llm_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant. Follow the user's prompt rules carefully."},
            {"role": "user", "content": prompt}],
            model=LLM_MODEL,
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM generation error: {e}")
        if "Request too large" in str(e) or "rate_limit_exceeded" in str(e):
            return "Sorry, your request is too large or I've hit a rate limit. Please try again later or ask a more specific question."
        else:
            return "I couldn't generate a response at this time."

class QueryPayload(BaseModel):
    question: str
    session_id: str = "default"

class FeedbackPayload(BaseModel):
    session_id: str
    question: str
    response: str
    feedback: str

@app.post("/feedback/")
async def submit_feedback(payload: FeedbackPayload):
    save_chat_history(payload.session_id, payload.question, payload.response)
    return {"status": "success"}

@app.post("/ask/")
async def ask_question(payload: QueryPayload):
    session_id = payload.session_id or ACTIVE_SESSION_ID
    question = payload.question.strip()

    # Load chat history
    conversation_history = load_conversation_history(session_id)

    # Step 1: Handle user replying with a model after clarification prompt
    if conversation_history and conversation_history[-1].get("clarification"):
        model_mentions = extract_model_names(conversation_history[-1]["response"])
        if is_model_name(question, model_mentions):
            # user clarified model → continue with original question
            session_model_context[session_id] = question.strip().upper()
            original_question = conversation_history[-1]["original_question"]
            question = original_question + f" (clarified: {question})"
        elif extract_model_names(question):
            # user mentioned a different model → treat it as a new topic
            session_model_context[session_id] = list(extract_model_names(question))[0]

    # Step 2: Always extract model names from current question and update context if present
    current_models = extract_model_names(question)
    previous_model = session_model_context.get(session_id)
    if current_models:
        # Always use the first detected model in the question
        new_model = list(current_models)[0]
        if new_model != previous_model:
            print(f"Context switch detected: {previous_model} → {new_model}")
            session_model_context[session_id] = new_model

    # Step 3: If user doesn't mention model, keep the previous context (for multi-turn)
    # (No action needed here)

    # Step 4: Rewrite ambiguous queries
    top_k = 25 if is_complex_question(question) else 15
    results = retrieve_relevant_chunks(question, session_id, top_k=top_k)
    rewritten_question = rewrite_ambiguous_query(question, results)

    # Step 5: Trigger clarification only if multiple models & no memory yet
    if session_id not in session_model_context and needs_clarification(results):
        models = set(re.findall(r"\b[A-Z0-9]+-[A-Z0-9]+\b", " ".join([c["chunk"] for c in results])))
        save_chat_history(session_id, question, "", clarification=True, original_question=question)
        return {
            "response": f"Which model are you asking about? {', '.join(sorted(models))}",
            "session_id": session_id
        }

    # Step 6: Check if user is asking for a datasheet or manual for a model
    # and if a matching URL is found in the retrieved chunks
    model_name = session_model_context.get(session_id, None)
    if model_name and ("datasheet" in question.lower() or "manual" in question.lower()):
        for chunk in results:
            if model_name.lower() in chunk["source"].lower():
                response = f"You can find the {model_name} datasheet/manual here: {chunk['source']}"
                save_chat_history(session_id, question, response)
                return {
                    "response": response,
                    "session_id": session_id,
                    "metrics": {"accuracy": 1.0, "semantic_similarity": 1.0, "comprehensiveness": 1.0, "competence": 1.0}
                }

    # Step 7: Build prompt and generate answer
    prompt = build_prompt(rewritten_question, results, conversation_history)
    response = generate_llm_response(prompt)
    response = clean_response(response)

    # Step 8: Check if the chatbot cannot answer the question
    if (not response or len(response.strip()) < 10 or
        "I couldn't generate a response" in response or
        "Sorry" in response):
        response = "Sorry, I cannot answer that. Please contact support@gmail.com"

    # Step 9: Compute quality metrics and return
    metrics = compute_quality_metrics(question, results, response)
    save_chat_history(session_id, question, response)

    return {
        "response": response,
        "session_id": session_id,
        "metrics": metrics
    }

@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    history = load_conversation_history(session_id, max_history=100)
    return {"session_id": session_id, "history": history}

@app.get("/sessions/")
def list_sessions():
    session_ids = set()
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                session_ids.add(entry['session_id'])
    return {"sessions": sorted(session_ids)}

@app.post("/new_session/")
async def new_session():
    session_id = str(uuid.uuid4())
    # Save a "session started" entry to the chat history
    save_chat_history(session_id, "Session started", "New chat session created.")
    return {"session_id": session_id}

@app.on_event("startup")
def startup_event():
    global index
    index = load_faiss_index()
    load_chunks()
    if index.ntotal == 0:
        reindex_all_chunks()
    process_urls()
    process_pdfs()
    print("[*] Data loaded.")
    print(f"Chunk texts after load: {len(chunk_texts)}")
    print(f"Chunk sources after load: {len(chunk_sources)}")
    print(f"Index size after load: {index.ntotal if hasattr(index, 'ntotal') else 'N/A'}")
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
