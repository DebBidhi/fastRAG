# 🧠 RAG with Qdrant, HuggingFace Embeddings, and LLaMA3

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline using:

- SQuAD dataset from HuggingFace 🤗
- Embedding with `nomic-embed-text-v1.5`
- Vector storage & retrieval using **Qdrant**
- Query answering using **LLaMA3 via Ollama**

---

## 🚀 Features

- ✅ Load and process SQuAD dataset
- ✅ Generate high-quality text embeddings
- ✅ Store and retrieve embeddings in Qdrant vector DB
- ✅ Fast semantic search using HNSW with optional binary quantization
- ✅ LLaMA3-powered generative answers from retrieved context

---

## 📦 Requirements

Install all required dependencies:

```bash
pip install -r requirments.txt
```

Ensure Qdrant is running locally at `http://localhost:6333`.
If you have docker you can follow this https://github.com/DebBidhi/Retrieval-Augmented-Generation-RAG-/blob/main/docker-compose.yml file and do
```
 docker-compose up
```
to run Qdrant locally at http://localhost:6333 

---

## 🧩 Project Structure

```text
.
├── embeddings_data.pkl        # Pre-generated embeddings and contexts
├── main_notebook.ipynb        # Main workflow (your full pipeline)
├── README.md                  # This file
├── hf_cache/                  # HuggingFace cache
```

---

## 🔁 Pipeline Steps

### 1. Load Dataset

```python
from datasets import load_dataset
dataset = load_dataset("squad")
texts = list(set([item["context"] for item in dataset["train"]]))
```

---

### 2. Generate Embeddings

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    cache_folder="./hf_cache"
)
```

Embed in batches using a custom class and save to `embeddings_data.pkl`.

---

### 3. Store in Qdrant

```python
from qdrant_client import QdrantClient, models

# Create collection and ingest data in batches
database = QdrantVDB("squad_collection")
database.define_client()
database.create_collection()
database.ingest_data(embeddata)
```

Supports fast indexing using HNSW and `on_disk=True` for memory efficiency.

---

### 4. Search via Retriever

```python
retriever = Retriever(database, embed_model)
results = retriever.search("Sample query")
```

This uses semantic similarity with DOT product and returns top contexts.

---

### 5. Answer Queries (RAG)

```python
from llama_index.llms.ollama import Ollama

rag = RAG(retriever)
answer = rag.query("Who uses VIP services at airports?")
```

Uses LLaMA3 from Ollama to generate the final answer using the retrieved context.

---

## 💡 Example Output

**Query:**
> The premium and VIP services in Airports are reserved for which type of passengers?

**Answer:**
> They are typically intended for First and Business class passengers, as well as members of airline's clubs.

---

## ⚠️ Notes

- Make sure to start **Ollama** and have the LLaMA3 model pulled (`ollama run llama3.2:1b`).
- Warning messages about `search` deprecation can be ignored but should eventually be replaced with `query_points`.

---
