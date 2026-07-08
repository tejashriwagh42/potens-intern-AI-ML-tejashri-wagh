# Multilingual Document Q&A with Citations and Contradiction Detection

A production-quality Retrieval-Augmented Generation (RAG) system with hybrid vector retrieval, Cross-Encoder reranking, multi-language support (Hindi, Marathi, Spanish, French, English), automated citation mapping, and contradiction analysis.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[File Uploads: PDF, DOCX, TXT, MD, CSV] --> B[Text Extractors: loader.py]
        B --> C[Paragraph-Preserving Chunker: chunker.py]
        C --> D[Embeddings Generator: all-MiniLM-L6-v2]
        D --> E[(ChromaDB Vector Store)]
    end

    subgraph Q&A Pipeline (POST /ask)
        F[User Question] --> G{Language Detector}
        G -->|Hindi / Marathi / Spanish / French| H[LLM Query Translator]
        G -->|English| I[English Query]
        H --> I
        I --> J[Vector Similarity Search: Top-10]
        J --> K[Cross-Encoder Reranking: ms-marco-MiniLM-L-6-v2]
        K --> L[LLM Context Synthesizer: Gemini/Groq]
        L --> M[JSON Answer + Citations]
        M --> N{Original Query Language == English?}
        N -->|No| O[LLM Response Translator]
        N -->|Yes| P[Final Response Payload]
        O --> P
    end

    subgraph Contradiction Detector (POST /contradict)
        Q[Document ID 1 + Document ID 2 + Topic] --> R[Retrieve Topic Chunks per Doc]
        R --> S[LLM Comparative Reasoning Engine]
        S --> T[Conflict Status: true/false/unknown + Explanation]
    end
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.11 or higher installed.
- API Key for **Gemini** (or **Groq** if configured).

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the `rag_project/` root directory (matching the structure of `.env.example`):
```env
# Choose "gemini" or "groq"
LLM_PROVIDER=gemini

# Provide your actual API keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# RAG Configurations
RERANKING_ENABLED=true
TOP_K=10
CHROMA_DB_PATH=chroma_db
DOCUMENTS_DIR=documents
```

### 3. Running the Server (FastAPI Backend)
Start the Uvicorn server:
```bash
uvicorn app:app --reload
```
The backend API is now running locally at `http://127.0.0.1:8000`. You can view the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.

### 4. Running the Streamlit Frontend UI
In a separate terminal, launch the Streamlit interface:
```bash
streamlit run ui/streamlit_app.py
```
This will open the user interface in your web browser, typically at `http://localhost:8501`.

---

## 📖 API Documentation

### 1. `POST /upload`
Ingests and indexes a single document into the vector database.
- **Parameters**: `file` (Multipart Form-Data)
- **Response Format**:
  ```json
  {
    "status": "success",
    "document_id": "doc_a1b2c3d4e5",
    "chunks_count": 8,
    "message": "File 'refund_policy.txt' ingested successfully."
  }
  ```

### 2. `POST /ask`
Submit a question and receive a citation-backed response in the matching language.
- **Body JSON**:
  ```json
  {
    "question": "What is the refund window?"
  }
  ```
- **Response Format**:
  ```json
  {
    "answer": "Refunds are available for physical items within 30 days of the purchase date.",
    "confidence_score": 0.94,
    "warning": null,
    "citations": [
      {
        "source_file": "refund_policy.txt",
        "page": 1,
        "chunk_id": "doc_a1b2_chunk_1",
        "snippet": "Refunds are available for physical items within 30 days..."
      }
    ]
  }
  ```

### 3. `POST /contradict`
Analyze two documents for opposing statements on a given topic.
- **Body JSON**:
  ```json
  {
    "doc1": "doc_id_1",
    "doc2": "doc_id_2",
    "topic": "refunds"
  }
  ```
- **Response Format**:
  ```json
  {
    "conflict": true,
    "reasoning": "Document 1 states refunds are allowed within 30 days while Document 2 states refunds are not available."
  }
  ```

### 4. `GET /documents`
List all ingested documents.
- **Response Format**:
  ```json
  [
    {
      "document_id": "doc_a1b2c3d4e5",
      "file_name": "refund_policy.txt",
      "file_type": "txt",
      "chunks_count": 4
    }
  ]
  ```

### 5. `GET /health`
Get database connectivity status and active LLM configuration.
- **Response Format**:
  ```json
  {
    "status": "healthy",
    "db_connected": true,
    "llm_provider": "gemini"
  }
  ```

---

## 🔍 Detailed Explanations

### Chunking Strategy
- **Chunk Size**: **800 characters** (~120-150 words). Chosen to match the average paragraph size of typical business/policy documents, keeping each chunk focused on a single logical claim or rule.
- **Chunk Overlap**: **150 characters** (~2-3 sentences). Ensures that context isn't lost if a sentence is split across boundaries, guaranteeing that core facts are preserved intact inside at least one chunk.
- **Section Heading Preservation**: When the chunker processes paragraphs, it actively scans for heading patterns (bold uppercase text, Markdown hashes, numbered sections) and stores the current section name. This section heading is then prepended directly to the text of the chunk (`[Section: heading]`), keeping the chunk contextually self-contained for vector lookup.
- **Advantages & Limitations**:
  - *Advantages*: Highly precise retrieval, maintains context of section headers, fits standard transformer context sizes effortlessly.
  - *Limitations*: Might break broad tables across chunks if they contain long rows.

### Retrieval Flow
1. **Embedding generation**: Text is embedded using the `all-MiniLM-L6-v2` model generating 384-dimensional dense vectors.
2. **Base Retrieval (similarity search)**: Retrieves the top $K=10$ candidate chunks using Cosine Distance metric.
3. **Cross-Encoder Reranking**: Re-orders the top 10 chunks using the `ms-marco-MiniLM-L-6-v2` cross-encoder. This calculates a query-document joint attention relevance score, sorting the most contextually relevant chunks to the top.

### Multilingual Workflow
1. The user inputs a query in their original language.
2. **Language Detection**: `langdetect` checks if the language is English, Hindi, Marathi, Spanish, or French.
3. **Query Translation**: If the query is not in English, the LLM translates the question to English. This is crucial because our embedding model is primarily English-trained.
4. **Retrieve & Answer**: Search and reranking run on English vectors, retrieving relevant English chunks. The LLM generates the answer in English, mapping out citations.
5. **Back-Translation**: The system translates the final answer and text snippets back into the user's input language.

### Confidence Scoring Logic
We employ a composite confidence formula combining relevance, reranker validation, and answer coverage:
$$Score = 0.5 \times S_{similarity} + 0.3 \times S_{rerank} + 0.2 \times S_{coverage}$$
- $S_{similarity}$: Average similarity score of the top-3 retrieved chunks.
- $S_{rerank}$: Sigmoid-normalized score of the top reranked chunk.
- $S_{coverage}$: Ratio of cited chunks inside the answer relative to total retrieved chunks.
- **Warning Trigger**: If the score is $< 0.50$, the API triggers a low confidence flag, appending a `"Low confidence answer. Please verify manually."` warning.

### Contradiction Detection Workflow
1. Topic query is embedded.
2. The vector database is queried twice: once filtering by `document_id == doc1`, and once by `document_id == doc2`.
3. The top retrieved chunks from both files are combined and fed to the LLM.
4. The LLM acts as an auditor, comparing the statements and returning structured JSON with a conflict status (`true`/`false`/`unknown`) and explanation.

---

## 📊 Evaluation Methodology

To evaluate the precision and correctness of the pipeline, we provide an automated evaluation script:
1. **Dataset**: Located in `evaluation/eval_dataset.json`, contains 10 QA pairs covering the 5 sample policy files.
2. **Metrics**:
   - **Retrieval Recall@10**: Whether the correct source file is present in the top-10 retrieved chunks.
   - **Precision@10**: The percentage of retrieved chunks belonging to the expected document.
   - **Answer Correctness**: An LLM-as-a-judge metric grading generated answers against ground-truths on a 1-to-5 scale.
3. **Execution**:
   ```bash
   python evaluation/evaluate.py
   ```
   This generates a comprehensive audit report stored in `evaluation/evaluation_report.json`.

---

## ⚠️ Limitations & Unfinished Parts
While the core pipeline is fully functional and stable, the following constraints exist in this 24-hour build:
1. **Passive Human-in-the-Loop Gate:** If the confidence score drops below `0.50`, the system generates a UI warning prompting the user to manually verify, but it does not actively block/queue the answer in a database for an administrator's explicit approval/rejection.
2. **Stateless Conversational Interface:** The Streamlit interface functions on a per-query stateless basis and does not maintain a multi-turn conversation memory history.
3. **Basic Sentence Splitting:** The chunking regex (`re.split`) splits sentences using punctuation spacing. This can occasionally split text prematurely on abbreviations (e.g., `U.S.`, `Corp.`).

## 🚀 Future Roadmap
If given more time, these would be the next features implemented:
1. **Active Human-in-the-Loop Queue:** Route any query that scores `< 0.50` confidence to an admin dashboard. The admin can edit and approve the answer, which gets saved to cache and logged as a new training/evaluation data point.
2. **Conversational Memory:** Store message history in Streamlit session state and pass past chat summaries to the retrieval query translator for contextual follow-up questions.
3. **Hybrid Sparse-Dense Retrieval:** Implement BM25 keyword search alongside dense vector embeddings, combining scores using Reciprocal Rank Fusion (RRF) to cover both exact keyword matches and semantic matches.
4. **NLTK/SpaCy Sentence Tokenization:** Replace standard regex splitting with a robust NLP sentence tokenizer to avoid splitting abbreviations.

---

## 🤖 AI Use Log
As requested in Rule #6, here is the log of all AI assistance used in this project:

| Tool | Message / Prompt Count | Primary Purpose / Usage |
|---|---|---|
| **Gemini 3.5 Flash** (via Antigravity assistant) | ~30 messages | Drafting the modular directory layout, troubleshooting PyTorch/sentence-transformers issues, handling Windows port conflict detection, writing the composite confidence score formula, and structuring Git commits. |

