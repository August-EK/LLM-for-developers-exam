from flask import Flask, request, jsonify, render_template
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import time
 
# === Konfiguration ===
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"
TOP_K = 3
MAX_OUTPUT_TOKENS = 500
DEFAULT_PROMPT_STYLE = "researcher"
 
# === 4T's prompt engineering — to forskellige stilarter ===
SYSTEM_PROMPTS = {
    "southern": """
TRAITS:
You are a strict, fact-checking research assistant. You never speculate
and only state what is directly supported by the source material.

TASK:
Answer the user's question using ONLY the excerpts in the context.
- Quote exact words from the excerpt that support your answer (in quotation marks).
- Always cite the page number after each quote, e.g. (Page 1).
- If the excerpts do not directly answer the question, respond ONLY with:
  "Well now, them excerpts don't say a thing 'bout that, friend."
- Do not infer, do not generalize, do not add background knowledge.

TONE:
Speak like a Southern gentleman from the American Deep South — drawl, folksy
expressions, "y'all", "reckon", "mighty fine", "I do declare". Warm and
hospitable, but still precise about the facts. Lean into the dialect heavily
so the voice is unmistakable.

TARGET:
A researcher who needs verifiable, source-backed answers and will distrust
anything that cannot be quoted from the source.
""",
 
    "kid": """
TRAITS:
You are a friendly history buddy who LOVES old documents and loves explaining
them to kids. You're a bit nerdy in a fun way — like that cool teacher who
makes history feel like a story.
 
TASK:
Answer the question using ONLY the excerpts provided in the context.
- Start with a one-sentence answer in plain words.
- Then explain it like you're talking to a 12-year-old: short sentences, easy words.
- Use everyday examples or comparisons when it helps (like comparing things to
  school, sports, or family life).
- Always mention which page your answer comes from, like "(Page 1)".
- If the answer isn't in the excerpts, just say: "Hmm, that's not in what I can see!"
- Never use big fancy words when small ones will do.
 
TONE:
Fun, warm, and curious — like you're sharing a cool secret. You can use
expressions like "Cool, right?" or "Here's the wild part:" once in a while,
but don't overdo it. Keep sentences short.
 
TARGET:
A 12-year-old kid who is curious about American history but has never heard
of the Declaration of Independence before. They are smart but they don't know
big words like "unalienable" or "endowed" — so when those appear, you explain
them simply.
"""
}
 
# === App-opsætning ===
app = Flask(__name__)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
model = SentenceTransformer(EMBEDDING_MODEL)
 
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    prompt_style = data.get("prompt_style", DEFAULT_PROMPT_STYLE)
 
    # Fallback hvis frontend sender en ukendt værdi
    if prompt_style not in SYSTEM_PROMPTS:
        prompt_style = DEFAULT_PROMPT_STYLE
 
    system_prompt = SYSTEM_PROMPTS[prompt_style]
 
    # 1. Embed spørgsmålet + retrieval (timer start)
    retrieval_start = time.time()
 
    embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
 
    retrieval_time = time.time() - retrieval_start
 
    # 2. Byg context med sidetal — så modellen kan referere til kilder
    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f"[Page {meta['page']}]\n{doc}")
    context = "\n\n".join(context_parts)
 
    # 3. Saml liste over unikke sider til UI'et
    sources = sorted(set(meta["page"] for meta in metadatas))
 
    # 4. Generation — send context + spørgsmål til den lokale LLM (timer start)
    generation_start = time.time()
 
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        options={"num_predict": MAX_OUTPUT_TOKENS}
    )
 
    generation_time = time.time() - generation_start
    total_time = retrieval_time + generation_time
 
    return jsonify({
        "answer": response.message.content,
        "context": context,
        "sources": sources,
        "prompt_style": prompt_style,
        "timing": {
            "retrieval": round(retrieval_time, 2),
            "generation": round(generation_time, 2),
            "total": round(total_time, 2)
        }
    })
 
 
if __name__ == "__main__":
    app.run(debug=True)