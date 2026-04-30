from flask import Flask, request, jsonify, render_template
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

app = Flask(__name__)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = """
You are a knowledgeable and neutral historian (Traits) specialized in American founding documents.
Your task is to answer questions strictly based on the provided excerpts from the US Declaration of Independence (Task).
Use a clear, academic and informative tone (Tone).
Your answers are aimed at curious students and researchers (Target).
If the answer is not found in the provided excerpts, say so clearly.
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        options={"num_predict": 200}
    )
    return jsonify({"answer": response.message.content, "context": context})

if __name__ == "__main__":
    app.run(debug=True)