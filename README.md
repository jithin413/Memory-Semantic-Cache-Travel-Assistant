# AI Travel Assistant with Memory and Semantic Caching

An intelligent AI-powered Travel Assistant built using FastAPI, LangGraph, Redis, Sentence Transformers, and Google Gemini.

The system enhances travel recommendations by combining user memory, request fingerprinting, and semantic caching to avoid unnecessary LLM calls and improve response latency.

---

## Overview

Traditional AI applications often send every user query directly to an LLM, even when the same or a semantically similar query has already been processed.

This project addresses that problem by introducing:

*   User-specific memory
*   Exact query identification using request fingerprints
*   Semantic similarity-based cache retrieval
*   Redis-based response caching
*   LangGraph workflow orchestration

When a user submits a query, the system first retrieves relevant memories and checks whether a similar query has already been processed. If a suitable cached response exists, it is returned instead of making another expensive LLM call.

---

## Features

*   **Personalized recommendations:** Tailors travel advice based on user memory retrieval.
*   **Query fingerprinting:** Uses SHA-256 for exact query identification.
*   **Semantic caching:** Uses Sentence Transformers and Cosine similarity for near-duplicate query detection.
*   **Performance optimization:** Redis-based response caching and reduced unnecessary LLM calls.
*   **Workflow orchestration:** Powered by LangGraph.
*   **Generative AI:** Gemini-powered response generation with model latency comparison.
*   **Robust API:** FastAPI REST API with input validation and error handling.

---

## Tech Stack

*   **Python**
*   **Frameworks:** FastAPI, Uvicorn
*   **AI/LLM:** LangGraph, LangChain, Google Gemini API, Sentence Transformers, Scikit-learn
*   **Database/Caching:** Redis
*   **Data Validation:** Pydantic

---

## Architecture

The application follows the workflow below:

```text
                    USER QUERY
                         |
                         v
                +----------------+
                |  MEMORY NODE   |
                | Retrieve User  |
                | Preferences    |
                +--------+-------+
                         |
                         v
                +----------------+
                | FINGERPRINT    |
                | SHA-256 Hash   |
                +--------+-------+
                         |
                         v
                +----------------+
                |  CACHE CHECK   |
                | Semantic Search|
                +--------+-------+
                         |
              +----------+----------+
              |                     |
        CACHE HIT              CACHE MISS
              |                     |
              v                     v
      +---------------+      +---------------+
      | Cached Response|      |   LLM NODE    |
      | Return Result  |      | Gemini Model  |
      +-------+-------+      +-------+-------+
              |                      |
              |                      v
              |              +---------------+
              |              | STORE CACHE   |
              |              | Redis +       |
              |              | Embeddings    |
              |              +-------+-------+
              |                      |
              +----------+-----------+
                         |
                         v
                  FINAL RESPONSE
