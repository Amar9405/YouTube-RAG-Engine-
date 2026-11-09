# 🎥 TubeTalk AI: Advanced YouTube RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Groq](https://img.shields.io/badge/Groq-Llama--3-orange)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-purple)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to chat with any YouTube video. By extracting video transcripts and using an advanced retrieval pipeline, this app generates accurate, context-aware answers without hallucinations.

## ✨ Key Features

*   **URL & ID Parsing:** Paste any YouTube link (full URL, short link, or raw ID) and the app extracts the video automatically.
*   **Multi-Language Support:** Automatically fetches English (`en`) or Hindi (`hi`) transcripts.
*   **Advanced Retrieval Engine:**
    *   **Multi-Query Retrieval:** Uses an LLM to rewrite vague user questions into 3 variations, improving search recall.
    *   **Cross-Encoder Reranker:** Uses `BAAI/bge-reranker-base` to re-rank the top 20 chunks and passes only the absolute best 4 to the LLM, drastically reducing hallucinations.
*   **Lightning-Fast Generation:** Powered by **Groq** (using `openai/gpt-oss-120b`), achieving sub-second response times compared to standard APIs.
*   **Interactive UI:** A sleek Streamlit interface featuring persistent chat memory and an "Under the Hood" expander to see the exact text chunks used to generate the answer.
*   **RAGAS Evaluation:** Includes a robust `evaluation.py` script using the RAGAS framework to mathematically score Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## 🏗️ Architecture

1.  **Ingestion:** `youtube-transcript-api` pulls captions, and `RecursiveCharacterTextSplitter` breaks them into 1000-character chunks.
2.  **Indexing:** `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) converts chunks to vectors, stored in a local `FAISS` database.
3.  **Retrieval:** 
    *   `MultiQueryRetriever` expands the question.
    *   `ContextualCompressionRetriever` + `CrossEncoderReranker` filters the noise and keeps the top 4 chunks.
4.  **Generation:** LangChain pipes the prompt and context to Groq's LLM.
5.  **UI:** Streamlit handles the chat state and user interface.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
