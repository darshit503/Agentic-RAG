# 🧠 Agentic RAG-Based Enterprise Knowledge Assistant

This repository contains the implementation of an **Agentic Retrieval-Augmented Generation (RAG)** pipeline built to automate secure knowledge ingestion, semantic retrieval, and contextual response generation from internal enterprise documents.

Developed in collaboration with **Worley Services India**, the system is designed to operate in secure backend environments and integrates modular LLM agents, semantic indexing, and intelligent categorization.

---

## 🚀 Project Overview

Modern enterprises often struggle with accessing scattered and unstructured documents across secure internal systems. This project solves that by introducing an end-to-end AI-powered enterprise assistant that:
- Ingests documents from dynamic, authenticated web portals
- Performs semantic enrichment and chunking
- Stores knowledge in vector databases
- Uses multi-agent LLM orchestration for contextual Q&A and reasoning

---

## 🧩 Core Features

- 🔐 **Secure Automated Web Crawling**: Playwright + BeautifulSoup, supports JS-heavy, dynamic portals.
- 🧹 **Text Normalization & Metadata Tagging**: Format-aware parsing, NER, sentiment analysis.
- 🧠 **Vector Embedding & Chunking**: Semantic chunking using transformer models, embeddings stored in FAISS/Chroma.
- 📁 **Deduplication**: Content hashing with SHA-256.
- 🤖 **Modular Agentic Pipeline**: Built using LangChain, CrewAI, and LCEL with Query, Retrieval, Reasoning, and Grading Agents.
- 💬 **LLM-Based Response Generation**: Factual, traceable, and context-aware replies with post-generation grading.
- 💾 **Semantic Memory Cache**: Caches Q&A pairs for repeated queries and faster access.
- 🛡️ **Enterprise-Ready**: Backend-first, secure, modular, and scalable.

---

## 📈 Outcomes & Benefits

- 🔍 Fast, intelligent access to internal documents
- 🧑‍💼 Reduced manual effort from SMEs
- 📚 Improved traceability and knowledge reuse
- 🏢 Foundation for scalable AI-based enterprise assistants

---

## 🛠️ Tech Stack

| Category         | Tools & Frameworks                                 |
|------------------|----------------------------------------------------|
| Web Crawling     | Playwright, BeautifulSoup                          |
| NLP & Cleaning   | PDFMiner, python-docx, PyMuPDF, NER, Regex         |
| Embedding        | OpenAI Embeddings, SentenceTransformers            |
| Vector DB        | FAISS, Chroma                                      |
| Orchestration    | LangChain, CrewAI, LCEL                            |
| Backend Language | Python                                             |
| LLM              | Azure OpenAI (GPT-4 / GPT-4o recommended)          |

---

## 📁 Project Structure

📦 enterprise-rag-assistant
├── .env.example
├── app/
│ └── main.py # Main entrypoint for orchestrated RAG flow
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```
### 2. Create a Virtual Environment
``` bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### 3. Install Dependencies
``` bash
pip install -r requirements.txt
```
### 4. Add Azure OpenAI Credentials
#### Create a .env file in the root directory with the following content:
```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```
▶️ How to Run
Option 1: Run the End-to-End Agentic Assistant
``` bash
python app/main.py
```
This script:

<li> Ingests documents using the secure crawler</li>

<li> Performs metadata tagging and chunking </li>

<li> Embeds content and stores it in the vector DB </li>

<li> Starts a multi-agent RAG loop to answer user queries </li>

👨‍💼 Developed By
Darshit  Pithadia
M.Sc. Data Science – SVKM’s NMIMS University
In collaboration with <b> Worley Services India  Pvt. Ltd </b>.

