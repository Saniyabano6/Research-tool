import os
import pickle
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # take environment variables from .env (especially OPENAI_API_KEY)

st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")
with open("faiss_store_openai.pkl", "rb") as f:
    vector_store = pickle.load(f)
# ---------- UI Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e293b 100%);
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
    }
    .stTextInput input {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
    }
    .stButton button {
        background-color: #38bdf8;
        color: #0f172a;
        font-weight: 600;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #0ea5e9;
        color: white;
    }

    /* Box / card layout */
    .box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5em;
        margin-bottom: 1em;
    }
    .box h2, .box h3 {
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.pkl"

main_placeholder = st.empty()

# LLM used for answering (needs a valid OPENAI_API_KEY in your .env)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

# Embeddings run locally via HuggingFace — no OpenAI cost/dependency for this part
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if process_url_clicked:
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        # load data
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # create embeddings and build FAISS index
        vectorindex = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(1)

        # save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex, f)

        main_placeholder.text("Processing complete. Ask a question below. ✅✅✅")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain.invoke({"question": query}, return_only_outputs=True)
        # result format --> {"answer": "", "sources": ""}

        answer_col, sources_col = st.columns([2, 1])

        with answer_col:
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.header("Answer")
            st.write(result["answer"])
            st.markdown('</div>', unsafe_allow_html=True)

        sources = result.get("sources", "")
        if sources:
            with sources_col:
                st.markdown('<div class="box">', unsafe_allow_html=True)
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    if source.strip():
                        st.write(source)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No processed data found. Please add URLs and click 'Process URLs' first.")
