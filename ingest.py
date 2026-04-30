from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

def ingest_document(pdf_path="document.pdf"):
    # Læs PDF
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks.append({"id": str(i), "text": text})

    # Gem i ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("documents")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for chunk in chunks:
        embedding = model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]]
        )
    print(f"Ingested {len(chunks)} pages into ChromaDB.")

if __name__ == "__main__":
    ingest_document()