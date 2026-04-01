from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

# Load embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load vector DB
db = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings
)

retriever = db.as_retriever()

# Load LLM
llm = Ollama(model="llama3")
ragas_llm = LangchainLLMWrapper(llm)

# Example question
question = "What is the leave policy?"

docs = retriever.get_relevant_documents(question)
context = " ".join([doc.page_content for doc in docs])

answer = llm.invoke(question)

data = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [[context]],
})

result = evaluate(
    data,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm
)

print(result)
