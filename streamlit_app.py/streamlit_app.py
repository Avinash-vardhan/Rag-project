import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory 
from sentence_transformers import SentenceTransformer, util
from bert_score import score
import torch



# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Company Policy RAG", layout="wide")
st.title("🏢 Company Policy AI Assistant")
st.write("Ask questions about company policies")

DATA_DIR = "data"
CHROMA_DIR = "chroma"

# -----------------------------
# Build RAG (cached)
# -----------------------------
@st.cache_resource
def load_rag():
    docs = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    llm = Ollama(model="llama3")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        chain_type="stuff"
    )

    return qa

qa = load_rag()
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def load_bertscore():
    return True  # just to cache loading

model = load_embedding_model()
load_bertscore()


# -----------------------------
# Chat UI
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask a question about company policy:")


if st.button("Ask") and question:

    response = qa.invoke({"query": question})
    answer = response["result"]

    # Save chat
    st.session_state.chat.append(("You", question))
    st.session_state.chat.append(("AI", answer))

    # -----------------------------
    # Add Ground Truth (Manual for now)
    # -----------------------------
    GROUND_TRUTH = {
    "What is the leave policy?": "Employees receive 20 paid leave days annually.",
    "What is work from home policy?": "Employees can work from home 2 days per week.",
    "What are office hours?": "Office hours are from 9 AM to 6 PM.",
    }
    ground_truth = None
    for key in GROUND_TRUTH:
        if key.lower() in question.lower():
            ground_truth = GROUND_TRUTH[key]
            break
    if ground_truth:

        # -----------------------------
        # Semantic Similarity
        # -----------------------------
        emb1 = model.encode(answer, convert_to_tensor=True)
        emb2 = model.encode(ground_truth, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()

        # -----------------------------
        # BERTScore
        # -----------------------------
        P, R, F1 = score([answer], 
                         [ground_truth], 
                         lang="en",
                         rescale_with_baseline=True)
        bert_f1 = F1.mean().item()
        
        overall_score = (similarity + bert_f1) / 2
        if overall_score > 0.85:
            performance = "🟢 Excellent"
        elif overall_score > 0.70:
            performance = "🟡 Good"
        else:
            performance = "🔴 Needs Improvement"

        # -----------------------------
        # Show Evaluation Panel
        # -----------------------------
        st.markdown("## 📊 Evaluation Metrics")
        st.markdown(f"##{performance}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Semantic Similarity", round(similarity, 3))
            st.progress(float(similarity))

        with col2:
            st.metric("BERTScore F1", round(bert_f1, 3))
            st.progress(float(bert_f1))
        st.markdown("---")
        st.metric("Overall Score",round(overall_score,3))
        st.progress(float(overall_score))

    else:
        st.info("No ground truth available for this question.")

for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
