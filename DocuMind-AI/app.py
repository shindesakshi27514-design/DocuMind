import streamlit as st
from dotenv import load_dotenv
from document_processor import extract_pdf_pages, split_pages
from vector_store import build_vector_store, search_vector_store
from rag_pipeline import answer_question

load_dotenv()

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    .hero { padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
            background: linear-gradient(120deg,#102a56,#176b87); margin-bottom: 1.2rem; }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: #dbeafe; margin: .35rem 0 0; }
    [data-testid="stChatMessage"] { background: white; border: 1px solid #e7edf5; border-radius: 14px; }
</style>
<div class="hero">
  <h1>📚 DocuMind AI</h1>
  <p>Upload your PDFs, ask questions, and get answers grounded in your documents.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API key",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="Get a key from https://aistudio.google.com/app/apikey"
    )
    st.session_state["api_key"] = api_key
    model_name = st.selectbox("Gemini model", ["gemini-3.8-flash", "gemini-2.0-flash"], index=0)
    st.caption("Your API key is used for answer generation. It is not saved to the project files.")
    if st.button("Clear chat and documents", use_container_width=True):
        for key in ("messages", "vector_store", "chunks", "document_names"):
            st.session_state.pop(key, None)
        st.rerun()

st.subheader("1. Upload documents")
uploads = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="For this version, PDF documents are supported."
)

if st.button("Process documents", type="primary", disabled=not uploads):
    with st.spinner("Reading PDFs, creating text chunks and building the vector index..."):
        try:
            pages = []
            names = []
            for uploaded_file in uploads:
                data = uploaded_file.getvalue()
                if not data:
                    continue
                names.append(uploaded_file.name)
                pages.extend(extract_pdf_pages(data, uploaded_file.name))
            chunks = split_pages(pages)
            if not chunks:
                st.error("No readable text was found. The PDF may be scanned or image-only.")
            else:
                store = build_vector_store(chunks)
                st.session_state["chunks"] = chunks
                st.session_state["vector_store"] = store
                st.session_state["document_names"] = names
                st.session_state["messages"] = []
                st.success(f"Processed {len(names)} PDF(s) into {len(chunks)} searchable chunks.")
        except Exception as exc:
            st.error(f"Could not process the documents: {exc}")

if st.session_state.get("vector_store") is not None:
    names = st.session_state.get("document_names", [])
    st.info("Loaded documents: " + ", ".join(names))
else:
    st.info("Upload PDF files and select **Process documents** to begin.")

st.divider()
st.subheader("2. Ask your documents")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['source']}**, page {source['page']}")

question = st.chat_input("Ask a question about your uploaded documents...")
if question:
    if st.session_state.get("vector_store") is None:
        st.warning("Please upload and process at least one PDF first.")
    elif not api_key.strip():
        st.warning("Enter your Gemini API key in the sidebar to generate an answer.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and preparing an answer..."):
                try:
                    retrieved = search_vector_store(
                        st.session_state["vector_store"], question, k=4
                    )
                    result = answer_question(
                        question, retrieved, api_key.strip(), model_name
                    )
                    st.markdown(result)
                    sources = []
                    seen = set()
                    for item in retrieved:
                        key = (item["source"], item["page"])
                        if key not in seen:
                            seen.add(key)
                            sources.append({"source": item["source"], "page": item["page"]})
                    if sources:
                        with st.expander("Sources used"):
                            for source in sources:
                                st.markdown(f"- **{source['source']}**, page {source['page']}")
                    st.session_state["messages"].append({
                        "role": "assistant", "content": result, "sources": sources
                    })
                except Exception as exc:
                    error_text = f"Sorry, something went wrong: {exc}"
                    st.error(error_text)
                    st.session_state["messages"].append({
                        "role": "assistant", "content": error_text
                    })

st.caption("DocuMind AI • RAG-powered document question answering")
