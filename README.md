# 🧠 Ask-My-Repo (Local Code RAG System)

A lightweight, fully local Retrieval-Augmented Generation (RAG) system built to search and understand codebases. Give it a GitHub URL, and it will clone, chunk, embed, and let you ask plain-English questions to find the exact lines of code you need.

## 🚀 Features

- **Automated Repo Cloning:** Downloads public GitHub repositories on the fly.
- **Context-Aware Chunking:** Slices code into 40-line chunks with a 5-line overlap so functions are never completely severed at the boundary.
- **Local Vector Database:** Uses `SentenceTransformers` and `ChromaDB` to embed code into vectors and store them persistently on your hard drive.
- **Semantic Search:** Ask questions in plain English and retrieve the exact file names, line numbers, and code snippets.

## 🛠️ Prerequisites

- Python 3.8+
- Install the required dependencies:
  ```bash
  pip install GitPython sentence-transformers chromadb
  ```
