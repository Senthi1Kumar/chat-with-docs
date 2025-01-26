# Chat with Ellie.ai Documentation 🔍🤖

A simple RAG-powered assistant for querying Ellie.ai's technical documentation using natural language.

![Project Architecture diagram](assets/imgs/ellie-rag-arch.png)

## Features
- Natural language queries over documentation
- Source citation with document references
- Conversation history tracking
- Hallucination safeguards
- Performance metrics monitoring

## Prerequisites
- Python 3.10+
- Groq API Key (set as `GROQ_API_KEY` in environment)
- Milvus DB

## Installation
```bash
git clone https://github.com/Senthi1Kumar/chat-with-docs.git
cd ellie-doc-chat
python -m venv .venv
# source .venv/bin/activate or .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
1. Copy example config:
```bash
cp config/settings.example.yaml config/settings.yaml
cp config/.env.example config.env
```
2. Update with your:
   - Documentation URLs
   - Milvus connection details
   - Device preferences (CPU/GPU)

## Usage
1. Ingest documentation:
```bash
python index_docs.py
```
2. Start chat interface:
```bash
streamlit run app.py
```

## Evaluation Framework [WIP]
- Faithfulness scoring
- Context relevance metrics
- Hallucination detection
- Retrieval performance tracking
- Latency monitoring

## License
MIT License
