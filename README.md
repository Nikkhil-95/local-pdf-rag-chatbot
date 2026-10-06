# 📚 Local PDF RAG Chatbot with LangChain & Ollama

A local **Retrieval-Augmented Generation (RAG)** chatbot built with Python, LangChain, Ollama, ChromaDB, and a PDF document.

The application loads a PDF, splits it into smaller chunks, converts those chunks into vector embeddings, stores them in a persistent Chroma vector database, retrieves relevant content based on the user's question, and uses an Ollama LLM to generate an answer based only on the retrieved context.

---

## 🚀 Project Overview

This project demonstrates the basic architecture of a RAG application:

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Document
     ↓
RecursiveCharacterTextSplitter
     ↓
Text Chunks
     ↓
Ollama Embeddings
     ↓
Chroma Vector Database
     ↓
MMR Retriever
     ↓
Relevant Documents
     ↓
Prompt + Context
     ↓
Ollama LLM
     ↓
Answer
```

The entire application runs locally using Ollama.

---

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **LangChain Community**
* **LangChain Text Splitters**
* **LangChain Ollama**
* **Ollama**
* **ChromaDB**
* **PyPDF**
* **Llama 3.2**
* **nomic-embed-text**

---

## ✨ Features

* Load PDF documents using `PyPDFLoader`
* Split documents using `RecursiveCharacterTextSplitter`
* Generate local embeddings using `nomic-embed-text`
* Store embeddings persistently using ChromaDB
* Reuse the existing vector database on subsequent runs
* Retrieve relevant document chunks using MMR
* Use `Llama 3.2` through Ollama to generate responses
* Interactive command-line chatbot
* Runs completely locally without requiring a cloud LLM API

---

## 📁 Project Structure

```text
Gen-AI/
│
├── documents/
│   └── your_document.pdf
│
├── RAG_chat_bot.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

The `chromo_db/` directory is generated automatically when the application is run for the first time.

It should **not** be committed to GitHub.

---

## ⚙️ Prerequisites

Make sure you have:

1. Python installed
2. Ollama installed
3. Required Ollama models downloaded

### Ollama Models

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Pull the LLM:

```bash
ollama pull llama3.2
```

Verify installed models:

```bash
ollama list
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📋 requirements.txt

The project uses the following packages:

```text
langchain
langchain-community
langchain-text-splitters
langchain-ollama
langchain-chroma
pypdf
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Place your PDF inside the `documents` folder.

For example:

```text
documents/
└── your_document.pdf
```

Update the PDF path in `RAG_chat_bot.py`:

```python
loader = PyPDFLoader("documents/your_document.pdf")
```

Then run:

```bash
python RAG_chat_bot.py
```

The first time the application runs, it creates the Chroma vector database:

```text
Creating vector database...
```

On subsequent runs, the existing database is loaded:

```text
Loading existing vector database...
```

---

## 💬 Example

```text
RAG System
Press 0 to Exit

User: Who is Santiago?

AI: Santiago is a young shepherd who travels in search of his Personal Legend...
```

Enter:

```text
0
```

to exit the application.

---

## 🧠 How the RAG System Works

### 1. PDF Loading

`PyPDFLoader` extracts the text from the PDF.

```python
loader = PyPDFLoader("documents/The_Alchemist.pdf")
documents = loader.load()
```

---

### 2. Text Splitting

Large documents are divided into smaller chunks.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
```

The overlap helps preserve context between neighboring chunks.

---

### 3. Embeddings

The project uses Ollama's `nomic-embed-text` model to convert text into numerical vectors.

```python
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

---

### 4. Vector Database

The chunks and their embeddings are stored in Chroma.

On the first run:

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chromo_db"
)
```

On later runs, the existing database is loaded:

```python
vector_store = Chroma(
    persist_directory="chromo_db",
    embedding_function=embedding_model
)
```

This prevents the document from being embedded again every time the application starts.

---

### 5. Retrieval

The application uses an MMR retriever:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 100,
        "fetch_k": 200
    }
)
```

`fetch_k` determines the candidate documents considered by MMR, while `k` determines how many documents are ultimately returned.

---

### 6. Context Creation

The retrieved documents are combined into a single context:

```python
context = "\n\n".join(
    [docs.page_content for docs in retrieved_documents]
)
```

The context is then passed to the LLM along with the user's question.

---

### 7. LLM Response

The project uses Llama 3.2 through Ollama:

```python
llm = ChatOllama(model="llama3.2")
```

The LLM receives:

```text
Context
+
User Question
```

and generates the final response.

---

## 🔒 Local & Private

The project is designed to run locally.

The following components run on the local machine:

* PDF processing
* Embedding generation
* Vector storage
* Retrieval
* LLM inference

No external LLM API key is required.

---

## ⚠️ Notes

### Chroma Database

The `chromo_db/` directory is generated locally and should not be committed to GitHub.

If the database is deleted, the application will recreate it the next time it runs.

### PDF Copyright

The example project uses a PDF for development/testing. Do not upload copyrighted books or other documents to GitHub unless you have permission to redistribute them.

For the public repository, use a document that you are legally allowed to distribute or instruct users to place their own PDF in the `documents/` directory.

---

## 🔮 Future Improvements

Possible improvements include:

* Add a graphical user interface
* Add streaming responses
* Support multiple PDFs
* Add document upload functionality
* Add source/page citations to answers
* Improve retrieval for broad questions
* Add conversation memory
* Add hybrid search
* Add reranking
* Add configurable chunk size and overlap
* Add automated evaluation of RAG responses
* Add support for different embedding models
* Add support for multiple LLMs

---

## 🎯 Learning Goals

This project was created to understand the fundamentals of:

* Large Language Models
* Embeddings
* Vector databases
* Semantic search
* Document chunking
* Retrieval-Augmented Generation
* LangChain
* Local LLMs with Ollama

---

## 👨‍💻 Author

**Nikhil Kumar**

Built as a learning project while exploring Generative AI, LangChain, and Retrieval-Augmented Generation.
