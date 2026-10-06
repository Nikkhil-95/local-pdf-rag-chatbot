import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Loading Document 

loader = PyPDFLoader("documents/The_Alchemist.pdf")

documents = loader.load()

# Splitting Document into Chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(documents)

# Creating Embeddings and storing them into vector_store

embedding_model = OllamaEmbeddings(
    model = "nomic-embed-text"
)

if not os.path.exists("chromo_db"):

    print("Creating vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chromo_db"
    )

else:

    print("Loading existing vector database...")

    vector_store = Chroma(
        persist_directory="chromo_db",
        embedding_function=embedding_model
    )


# Retriever

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 100,
        "fetch_k": 200
    }
)

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
                    You are a english literature teacher who job is to decipher the hidden meaning.
                    Use only the context Provided to answer the query.           
        """),
        ("human", """
                    context:{context}
                    question:{question}    
        """)
    ]
)

print("RAG System")

print("Press 0 to Exit")

while True:

    question = input(f"User: ")
    if question == "0":
        break

    retrieved_documents = retriever.invoke(question)

    print("Retrieved documents:", len(retrieved_documents))

    if not retrieved_documents:
        print("AI: I could not find the answer in the document")
        continue

    context = "\n\n".join(
        [docs.page_content for docs in retrieved_documents]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question" : question
    })

    response = llm.invoke(
        final_prompt
    )

    print(f"AI: {response.content}")

    print()