# Enterprise-multisource-RAG-platform
A production-style Retrieval-Augmented Generation (RAG) chatbot built using LangChain that ingests data from multiple sources including websites, YouTube, files, images and URLs, stores embeddings in a vector database, and maintains chat history using MongoDB.

**🚀 Features**

**🔗 Multi-source data ingestion**

Websites & sitemap crawling

YouTube channel & video transcript ingestion

Files (PDF, CSV, DOCX, XLSX, PPTX, TXT, JSON)

Image & S3 URL-based document ingestion

🧠 RAG-based Chatbot

Context-aware responses using retrieved documents

Built using LangChain document loaders & chains

🗄️ Vector Database Integration

Stores embeddings for efficient semantic search

Supports incremental document updates

💬 Chat History Management

Stores chat sessions in MongoDB

Supports admin-based chat history retrieval

Generates chatbot usage statistics

⚡ Concurrent Web Crawling

Multi-threaded website crawling

Sitemap & internal link discovery

**🏗️ Architecture Overview**
User Query
   ↓
Retriever (Vector DB)
   ↓
Relevant Documents
   ↓
LLM (LangChain)
   ↓
Response
   ↓
MongoDB (Chat History)

**🔧 Tech Stack**

Python

LangChain

MongoDB

Vector Database (FAISS / Chroma / similar)

BeautifulSoup

YouTube Data API

ThreadPoolExecutor (Concurrency)


**📊 Use Cases**

Enterprise knowledge base chatbot

Website & documentation Q&A system

YouTube content summarization & querying

Internal AI assistant with persistent memory
