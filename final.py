# 📦 Phase 1 Libraries
import os
import warnings
import logging
import streamlit as st

# 📦 Phase 2 Libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 📦 Phase 3 Libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# 🔇 Disable warnings/logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 🧠 Title & Session
st.set_page_config(page_title="Ask Your PDF 🔍", layout="wide")
st.title("📄🤖 Ask your PDF – Text RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 🧠 Load PDF
@st.cache_resource
def get_vectorstore(uploaded_file):
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loaders = [PyPDFLoader("temp.pdf")]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ).from_loaders(loaders)
        return index.vectorstore
    return None

# 📄 Upload PDF
uploaded_file = st.sidebar.file_uploader("📤 Upload your PDF", type="pdf")

# 🧠 Model setup
groq_chat = ChatGroq(
    groq_api_key="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",  # 🔑 Replace with your actual key
    model_name="llama3-8b-8192"
)

# 🧠 Build vectorstore from uploaded PDF
vectorstore = get_vectorstore(uploaded_file)

# 🧠 Chain Setup
if vectorstore:
    chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )
else:
    st.warning("Please upload a PDF to begin.")

# 🧠 Chat UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# 🧠 Input Area (Text Only)
text_input = st.chat_input("💬 Ask a question...")

# 🧠 On input
if text_input and vectorstore:
    st.chat_message("user").markdown(text_input)
    st.session_state.messages.append({"role": "user", "content": text_input})

    try:
        response = chain.invoke({"query": text_input})
        result = response["result"]
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
    except Exception as e:
        st.error(f"❌ Error: {e}")
