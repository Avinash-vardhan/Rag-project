import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

print("Loading documents...")

docs = []
for file in os.listdir("data"):
    path = f"data/{file}"
    if file.endswith(".pdf"):
        docs.extend(PyPDFLoader(path).load())
    elif file.endswith(".txt"):
        docs.extend(TextLoader(path).load())

print(f"Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")

print("Creating embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma")
db.persist()

print("Vector database created")

llm = Ollama(model="llama3")
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("\nRAG system is ready! Ask your questions.")
print("Type 'exit' to stop.\n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break
    print("\nAnswer:", qa.run(q), "\n")
