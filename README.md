# IIT Bhubaneswar RAG Assistant

A standalone, terminal-based Retrieval-Augmented Generation (RAG) application that answers questions about IIT Bhubaneswar using institutional documents and faculty information.

The application combines semantic search with a locally running Large Language Model (LLM) to retrieve relevant information and generate context-grounded responses with source references.

## Features

- **Semantic Search:** Converts user queries into 384-dimensional embeddings using Sentence Transformers.
- **Vector Retrieval:** Uses PostgreSQL and pgvector to retrieve relevant document chunks through cosine-distance similarity search.
- **LLM-based Generation:** Integrates Ollama with Llama 3.2 (3B) to generate answers using retrieved context.
- **Source Attribution:** Displays relevant document titles and source URLs alongside generated answers.
- **Local Execution:** Runs entirely through the terminal without requiring a frontend or backend server.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | PostgreSQL + pgvector |
| LLM Runtime | Ollama |
| Language Model | Llama 3.2 (3B) |
| Database Driver | psycopg |

## Architecture

```text
             User Query
                  |
                  v
       Sentence Transformer
          (384-D Embedding)
                  |
                  v
       PostgreSQL + pgvector
          (Vector Search)
                  |
                  v
         Relevant Chunks
                  |
                  v
       Context + User Query
                  |
                  v
          Ollama / Llama 3.2
                  |
                  v
          Generated Answer
                  |
                  v
        Answer + Source URLs
```

## Project Structure

```text
iitbbs-rag-assistant/
│
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── answer.py
│
├── scripts/
│   ├── extract_pages.py
│   ├── chunk_documents.py
│   ├── embed_and_store.py
│   ├── test_vector_search.py
│   └── test_local_retriever.py
│
└── data/
    ├── raw/
    │   └── seed_urls.txt
    └── processed/
        ├── documents.jsonl
        ├── chunks.jsonl
        └── faculty.json
```

## Prerequisites

Install the following before running the application:

- Python 3.12
- Docker Desktop
- Ollama
- PostgreSQL with the pgvector extension

Pull the required language model:

```powershell
ollama pull llama3.2:3b
```

## Setup and Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/Saathwik2006/iitbbs-rag-assistant.git
cd iitbbs-rag-assistant
```

### 2. Create a Virtual Environment

```powershell
py -3.12 -m venv venv
```

### 3. Install Dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` and configure the PostgreSQL connection string and other required settings.

Example database URL:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
```

Ensure PostgreSQL is running and the database contains the required IIT Bhubaneswar documents and their embeddings.

### 5. Start the Application

Ensure Ollama and PostgreSQL are running, then execute:

```powershell
.\venv\Scripts\python.exe main.py
```

## Usage

Once the application starts, enter questions directly in the terminal.

Example:

```text
You: Who is Abhinav Arya?

Assistant: Abhinav Arya is an Assistant Professor at
the School of Electrical and Computer Sciences,
IIT Bhubaneswar.

Sources:
- Faculty record — Dr. Abhinav Arya
- Abhinav Arya – School of Electrical and Computer Sciences
```

Type `exit` to close the application.

## How RAG Works

1. **Query Embedding:** The user's question is converted into a numerical vector using the Sentence Transformer model.
2. **Similarity Search:** pgvector compares the query embedding with stored document embeddings and retrieves the most relevant chunks.
3. **Context Preparation:** Retrieved chunks are combined with the original question.
4. **Answer Generation:** Llama 3.2 generates a response using the retrieved information.
5. **Source Attribution:** The application displays the source documents used during retrieval.

## Future Improvements

- Improve retrieval relevance through reranking.
- Introduce relevance thresholds to filter unrelated chunks.
- Expand the institutional knowledge base.
- Evaluate retrieval quality and answer accuracy.

## Author

**Saathwik Aithal**  
B.Tech – Computer Science and Engineering  
Institute of Technology Bhubaneswar

[GitHub Profile](https://github.com/Saathwik2006)
