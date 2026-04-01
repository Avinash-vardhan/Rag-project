# ===============================
# HUGGING FACE RAG EVALUATION
# ===============================

from sentence_transformers import SentenceTransformer, util
from bert_score import score
from streamlit_app import load_rag 
qa=load_rag()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example question
question = "What is the leave policy?"

# Your RAG generated answer (replace this with actual output)
rag_answer =qa.invoke(question)
print("RAG Answer:",rag_answer)
# Ground truth answer (correct answer from company policy)
ground_truth = "Employees receive 20 paid leave days annually."

# -----------------------------------
# 1️⃣ Sentence Transformer Similarity
# -----------------------------------

emb1 = model.encode(rag_answer, convert_to_tensor=True)
emb2 = model.encode(ground_truth, convert_to_tensor=True)

similarity = util.cos_sim(emb1, emb2)

print("Semantic Similarity Score:", similarity.item())

# -----------------------------------
# 2️⃣ BERTScore
# -----------------------------------

P, R, F1 = score([rag_answer], [ground_truth], lang="en")

print("BERTScore F1:", F1.mean().item())