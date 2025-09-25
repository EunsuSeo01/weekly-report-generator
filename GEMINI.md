# GEMINI.md

## Project Overview

This project is an AI-powered weekly report generator. It automatically creates weekly work reports by summarizing data from various sources like Slack, Notion, OneDrive, and Outlook.

The project consists of:

*   **Backend:** A FastAPI server that uses LangChain and a vector database (FAISS with Hugging Face embeddings) to process and summarize documents.
*   **Frontend:** A choice of two frontends:
    *   A Streamlit application.
    *   A Vue.js application.
*   **Database:** A PostgreSQL database with pgvector extension for storing vector embeddings, managed via Docker.

## Building and Running

### 1. Backend

The backend is a FastAPI application.

**To install dependencies:**

```bash
pip install -r backend/requirements.txt
```

**To run the backend server:**

```bash
uvicorn backend.main:app --reload
```

### 2. Frontend

There are two frontend options: Streamlit and Vue.js.

**To run the Streamlit frontend:**

```bash
streamlit run fastapi-project/streamlit_app.py
```

**To run the Vue.js frontend:**

```bash
cd fastapi-project/front
npm install
npm run dev
```

### 3. Database

The project uses a PostgreSQL database with the pgvector extension, managed with Docker.

**To start the database:**

```bash
docker-compose up -d
```

## Development Conventions

*   The backend is written in Python using FastAPI.
*   The frontend is available in two options: Streamlit (Python) and Vue.js (JavaScript).
*   The project uses a vector database for similarity search, which is a core part of the AI functionality.
*   API keys and other sensitive information are stored as placeholders in the source code. These should be replaced with actual values for the application to work.
