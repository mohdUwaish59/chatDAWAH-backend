# ChatDawah Backend API

RAG-powered Q&A chatbot API with Qdrant Cloud vector database and HuggingFace LLM.

## Features

- FastAPI backend
- Qdrant Cloud vector database
- FastEmbed for efficient embeddings
- HuggingFace LLM (OpenAI-compatible API)
- Multiple LLM providers (HuggingFace / OpenAI)
- Semantic search with context retrieval

## Tech Stack

- **Framework**: FastAPI
- **Vector DB**: Qdrant Cloud
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **LLM**: HuggingFace Router (OpenAI-compatible)
- **Deployment**: Render

## Environment Variables

Required environment variables:

```env
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=your_hf_token
HUGGINGFACE_MODEL=openai/gpt-oss-20b:groq
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
MAX_TOKENS=1000
TEMPERATURE=0.7
TOP_K=5
```

## Installation

```bash
pip install -r requirements.txt
```

## Local Development

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your credentials

# Run the server
python main.py
```

Server runs on http://localhost:8000

## API Endpoints

### Health Check
```
GET /health
```

### Statistics
```
GET /stats
```

### Query
```
POST /query
Content-Type: application/json

{
  "question": "Your question here",
  "top_k": 5
}
```

### API Documentation
```
GET /docs
```

## Deployment

See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for deployment instructions.

### Render Deployment

1. Push to GitHub
2. Connect to Render
3. Add environment variables
4. Deploy

## Project Structure

```
backend/
├── app/
│   ├── api/          # API routes
│   ├── core/         # Configuration
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   └── utils/        # Utilities
├── data/             # Dataset
├── main.py           # Entry point
├── requirements.txt  # Dependencies
└── render.yaml       # Render config
```

## Developer

**Mohammad Uwaish**  
GitHub: [@mohammaduwaish](https://github.com/mohammaduwaish)
