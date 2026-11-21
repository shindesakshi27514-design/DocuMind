# DocuMind AI — Document Q&A Assistant

DocuMind AI is a Retrieval-Augmented Generation (RAG) application. Upload PDF documents, index their content with sentence embeddings, search relevant passages with FAISS, and ask questions using Google's Gemini API.

## Features

- Upload one or more PDF files
- Extract text page by page with source metadata
- Split text into overlapping chunks
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Store and search vectors with FAISS
- Generate context-grounded answers with Gemini
- Display source filenames and page numbers
- Streamlit chat interface

## Requirements

- Python 3.10 or newer
- Internet connection for the first embedding-model download and Gemini API
- Gemini API key: https://aistudio.google.com/app/apikey

## Run on Windows PowerShell

```powershell
cd Desktop
cd DocuMind-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Configure Gemini

When the app opens, paste your Gemini API key into the password field in the sidebar. You can create a key in Google AI Studio. Do not share your key or commit it to GitHub.

Alternatively, copy `.env.example` to `.env` and set `GEMINI_API_KEY=your_key`. For production, use your hosting provider's secret manager rather than committing `.env`.

## How the RAG pipeline works

1. **Ingestion:** `document_processor.py` extracts text from each PDF page.
2. **Chunking:** text is split into 900-character chunks with 150-character overlap.
3. **Embedding:** `vector_store.py` converts each chunk to a vector using Sentence Transformers.
4. **Indexing:** vectors are stored in a FAISS inner-product index (normalized vectors make this cosine-similarity search).
5. **Retrieval:** the user's question is embedded and the top four matching chunks are retrieved.
6. **Generation:** `rag_pipeline.py` sends the question and retrieved context to Gemini and instructs it to answer only from that context.

## Limitations

- This starter version supports text-based PDFs. Scanned/image-only PDFs need OCR and are not automatically readable.
- The vector index is held in Streamlit session memory and is rebuilt when documents are processed or the session is restarted.
- Gemini API usage may have quotas or charges depending on your account and model availability.
- Avoid uploading confidential documents unless you are authorized to use the selected AI service.

## Troubleshooting

- **No readable text:** the PDF may be scanned. Use a text-based PDF or add OCR support.
- **Gemini authentication/model error:** check the API key and confirm the selected model is available to your account.
- **First run is slow:** Sentence Transformers downloads its embedding model on first use.
- **FAISS install error:** use a supported 64-bit Python version (Python 3.10–3.12 is a good starting point).

## Suggested test

Upload a text-based PDF containing a topic you know. Ask:
- "Summarize this document."
- "What are the main points?"
- "Which page discusses [a specific topic]?"
- Ask a question whose answer is absent and verify the assistant says it cannot find the answer in the uploaded documents.

## Project structure

```text
DocuMind-AI/
├── app.py
├── config.py
├── document_processor.py
├── embeddings.py
├── vector_store.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── uploads/
```

`embeddings.py` is not needed as a separate module in this version; embedding creation is implemented in `vector_store.py`.
