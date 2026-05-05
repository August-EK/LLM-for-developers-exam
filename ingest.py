from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
 
# === Konfiguration ===
PDF_PATH = "document.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
 
# Chunking-parametre (tegn-baseret med overlap)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
 
 
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Deler en tekststreng op i overlappende chunks af fast tegn-størrelse.
    Overlap sikrer at sætninger der krydser chunk-grænser ikke mister kontekst.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # spring tomme chunks over
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
 
 
def ingest_document(pdf_path=PDF_PATH):
    # Læs PDF side for side
    reader = PdfReader(pdf_path)
    all_chunks = []
 
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
 
        # Del siden op i mindre chunks
        page_chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(page_chunks):
            all_chunks.append({
                "id": f"page{page_num}_chunk{chunk_idx}",
                "text": chunk,
                "metadata": {
                    "page": page_num + 1,  # 1-indekseret for menneske-læselighed
                    "chunk_index": chunk_idx
                }
            })
 
    # Initialiser ChromaDB og embedding-model
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)
 
    # Embed og gem hver chunk (upsert tillader sikre re-kørsler)
    for chunk in all_chunks:
        embedding = model.encode(chunk["text"]).tolist()
        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )
 
    print(f"Ingested {len(all_chunks)} chunks from {len(reader.pages)} pages.")
 
 
if __name__ == "__main__":
    ingest_document()