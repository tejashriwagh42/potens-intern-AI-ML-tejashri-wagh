import os
import requests
import streamlit as st

# Setup API URL
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="Multilingual Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    /* Font family overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient styling */
    .title-gradient {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards and Containers styling */
    .premium-card {
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.5rem;
    }
    
    .answer-box {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #6366f1;
        color: #f8fafc;
        font-size: 1.15rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Badges */
    .badge-green {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-amber {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-red {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .citation-container {
        background-color: #0f172a;
        border-radius: 8px;
        border: 1px solid #1e293b;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get list of indexed documents
def get_documents():
    try:
        res = requests.get(f"{API_URL}/documents")
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.sidebar.error(f"Could not connect to backend API: {e}")
    return []

# Helper function to check health status
def get_health():
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

# ================= SIDEBAR =================
st.sidebar.markdown("### 🤖 System Control")

# Backend health status
health = get_health()
if health:
    st.sidebar.success(f"Backend Connected ({health['llm_provider'].upper()})")
else:
    st.sidebar.error("Backend Disconnected")

# File uploader
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📂 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Support PDF, TXT, DOCX, MD, CSV",
    type=["pdf", "txt", "docx", "md", "csv"],
    help="Select a file to index into the Chroma Vector Database"
)

if uploaded_file is not None:
    # Check if we should ingest
    if st.sidebar.button("🚀 Ingest Document"):
        with st.spinner("Extracting & Indexing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.sidebar.success(data["message"])
                else:
                    st.sidebar.error(f"Error {res.status_code}: {res.json().get('detail')}")
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")

# Display Indexed Documents
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📚 Indexed Documents")
documents = get_documents()

if documents:
    for doc in documents:
        with st.sidebar.container():
            st.markdown(
                f"📄 **{doc['file_name']}**  \n"
                f"`ID: {doc['document_id']}` | `Chunks: {doc['chunks_count']}`",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
else:
    st.sidebar.info("No documents indexed yet. Please upload files to get started.")


# ================= MAIN AREA =================
st.markdown("<h1 class='title-gradient'>Multilingual Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Perform multilingual queries, trace citations with ease, and run contradiction detection.</p>", unsafe_allow_html=True)

# Create Main Tabs
tab_qa, tab_contradiction = st.tabs(["💬 Query & Chat", "🔍 Contradiction Check"])

# ================= TAB 1: Q&A =================
with tab_qa:
    st.markdown("### Ask any question about your documents")
    st.info("You can query in English, Hindi (हिंदी), Marathi (मराठी), Spanish (Español), or French (Français).")
    
    question = st.text_input(
        "Enter your question here:",
        placeholder="e.g., What is the refund policy for digital products?",
        key="qa_question_input"
    )
    
    if st.button("🔍 Ask Document", key="qa_ask_button"):
        if not question.strip():
            st.warning("Please enter a valid question.")
        elif not documents:
            st.error("No documents indexed. Please upload and ingest a document first.")
        else:
            with st.spinner("Analyzing context & generating response..."):
                try:
                    payload = {"question": question}
                    res = requests.post(f"{API_URL}/ask", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        confidence = data["confidence_score"]
                        warning = data.get("warning")
                        citations = data.get("citations", [])
                        
                        # Display Confidence score badge
                        if confidence >= 0.75:
                            badge_html = f"<span class='badge-green'>Confidence: {confidence:.2f} (High)</span>"
                        elif confidence >= 0.50:
                            badge_html = f"<span class='badge-amber'>Confidence: {confidence:.2f} (Medium)</span>"
                        else:
                            badge_html = f"<span class='badge-red'>Confidence: {confidence:.2f} (Low)</span>"
                            
                        # Show Warning alert if present
                        if warning:
                            st.warning(f"⚠️ {warning}")
                            
                        st.markdown(f"**Answer Status:** {badge_html}", unsafe_allow_html=True)
                        
                        # Display Answer in custom styled container
                        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                        
                        # Display Citations
                        st.markdown("#### 🔍 Citations & Sources")
                        if citations:
                            for idx, cit in enumerate(citations):
                                with st.expander(f"📍 Citation #{idx+1} — {cit['source_file']} (Page {cit['page']})"):
                                    st.markdown(f"**Source Document:** `{cit['source_file']}`")
                                    st.markdown(f"**Location:** Page `{cit['page']}` | Chunk ID: `{cit['chunk_id']}`")
                                    st.markdown(f"**Supporting text snippet:**")
                                    st.markdown(f"<div class='citation-container'><em>\"{cit['snippet']}\"</em></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("*No supporting citations were found in the database.*")
                    else:
                        st.error(f"Error {res.status_code}: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ================= TAB 2: CONTRADICTION =================
with tab_contradiction:
    st.markdown("### Contradiction Detection Engine")
    st.markdown("Select two ingested documents and supply a topic. The system will retrieve relevant chunks and analyze them for conflicting claims.")
    
    if len(documents) < 2:
        st.warning("Please upload and index at least **two** documents to use this feature.")
    else:
        # Document Selection dropdowns
        doc_options = {doc["file_name"]: doc["document_id"] for doc in documents}
        file_names = list(doc_options.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            doc1_name = st.selectbox("Select Document 1:", file_names, index=0, key="contra_doc1")
        with col2:
            doc2_name = st.selectbox("Select Document 2:", file_names, index=min(1, len(file_names)-1), key="contra_doc2")
            
        topic = st.text_input(
            "Topic/Claim to verify:",
            placeholder="e.g., refund timeframe, remote work policy",
            key="contra_topic_input"
        )
        
        if st.button("⚡ Compare Documents", key="contra_compare_button"):
            if not topic.strip():
                st.warning("Please enter a valid topic.")
            elif doc1_name == doc2_name:
                st.error("Please select two different documents to compare.")
            else:
                with st.spinner("Analyzing claims..."):
                    try:
                        payload = {
                            "doc1": doc_options[doc1_name],
                            "doc2": doc_options[doc2_name],
                            "topic": topic
                        }
                        res = requests.post(f"{API_URL}/contradict", json=payload)
                        
                        if res.status_code == 200:
                            data = res.json()
                            conflict = data["conflict"]
                            reasoning = data["reasoning"]
                            
                            # Beautified outputs depending on results
                            if conflict is True:
                                st.error("❌ **Contradiction Detected!**")
                                st.markdown(f"<div class='premium-card' style='border-color: #f87171;'>{reasoning}</div>", unsafe_allow_html=True)
                            elif conflict is False:
                                st.success("✅ **No Contradiction Found**")
                                st.markdown(f"<div class='premium-card' style='border-color: #34d399;'>{reasoning}</div>", unsafe_allow_html=True)
                            else:
                                st.warning("❓ **Result: Unknown / Insufficient Evidence**")
                                st.markdown(f"<div class='premium-card' style='border-color: #fbbf24;'>{reasoning}</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"Error {res.status_code}: {res.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
