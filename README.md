```
# Health check
curl http://127.0.0.1:8000/api/health/

# Text search
curl -X POST http://127.0.0.1:8000/api/recommend/ \
  -H "Content-Type: application/json" \
  -d '{"type": "text_search", "query": "gaming laptop RTX", "n_recommendations": 3}'

# Similar laptops
curl -X POST http://127.0.0.1:8000/api/recommend/ \
  -d '{"type": "similar", "laptop_id": 42, "n_recommendations": 5}'

# Retrain & reload without restarting Django
python main.py
curl -X POST http://127.0.0.1:8000/api/admin/reload/
```

# 🧠 Laptop Recommendation System (Django + FAISS + Sentence Transformers)

Production-style semantic laptop recommendation API built with:

- Django 5
- FAISS (vector similarity search)
- Sentence-Transformers (all-MiniLM-L6-v2)
- MLflow experiment tracking
- Versioned model registry

Supports:
- Semantic text search
- Item-to-item similarity
- Preference-based filtering
- Personalized recommendations
- Hot model reload without restarting server

---

## 📁 Project Structure

rec_system/
├── config/              # Django project
├── api/                 # API views & ModelManager
├── src/                 # ML logic (encoder, KNN, engine)
├── pipelines/           # Training pipeline
├── models/              # Trained model artifacts (ignored in git)
├── extracted_data/      # Feature store (ignored)
├── mlruns/              # MLflow logs (ignored)
└── main.py              # Training entrypoint

---

## 🚀 Setup

### 1️⃣ Create virtual environment

```bash
uv venv
uv sync

uv run python main.py

cd config
uv run python manage.py makemigrations
uv run python manage.py migrate
uv run python manage.py runserver