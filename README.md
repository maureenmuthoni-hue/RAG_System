# RAG_systems_with_ChromaDB_and_Groq
**How to build RAG systems**
The system ingests a PDF document, chunks and embeds its content, stores the vectors in a local ChromaDB instance, and retrieves semantically relevant passages for a given query with Groq available for the generation step.
The demo document used is the **Constitution of Kenya (2010)**.

## How It Works

```
PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Search → (LLM Generation)
```

1. **Embed** — Load `BAAI/bge-small-en-v1.5` via `sentence-transformers` to encode text into 384-dimensional vectors.
2. **Ingest** — Extract text from a PDF with `pypdf`, split it into 300-word chunks, and embed each chunk.
3. **Store** — Persist embeddings and document chunks in a local ChromaDB collection.
4. **Retrieve** — Embed a user query and retrieve the top-k most semantically similar chunks using cosine similarity.
5. **Generate** *(optional)* Pass retrieved context to a Groq-hosted LLM for a grounded answer.

---
## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (for the generation step)
### Installation

```bash
git clone https://github.com/your-username/rag-system.git
cd rag-system
pip install sentence_transformers groq numpy python-dotenv pypdf chromadb
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Notebook

1. Place your PDF in the project directory (or update the file path in the notebook).
2. Open `RAG_System.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom.

```bash
jupyter notebook RAG_System.ipynb
```

---

## Project Structure

```
rag-system/
├── RAG_System.ipynb        # Main notebook
├── chroma_db/              # Auto-generated ChromaDB persistent store
├── .env                    # API keys (not committed)
└── README.md
```

---

## Example Query

```python
query = "What does the Constitution say about freedom of expression?"
query_embedding = model.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)
```

The system returns the two most relevant chunks from the document based on semantic similarity.

---

## Key Concepts Demonstrated

- **Dense vector embeddings** and how sentence-transformers encode semantic meaning
- **Cosine similarity** for measuring embedding distance
- **Text chunking** strategies for long documents
- **Vector databases** (ChromaDB) for efficient similarity search
- **RAG architecture** as an alternative to fine-tuning for knowledge-grounded generation

---
