# LennyHub RAG - Resume Tomorrow

## ✅ Completed
- [x] Cloned repository to `/Users/shawn/proj/lennyhub-rag`
- [x] Created Python 3.13 virtual environment (`.venv`)
- [x] Installed all dependencies
- [x] Modified all 9 Python files to use Claude + Voyage AI instead of OpenAI

## ⏳ Remaining Steps

### 1. Add API Keys to `.env`
Edit `/Users/shawn/proj/lennyhub-rag/.env` and replace the placeholder values:

```
ANTHROPIC_API_KEY=sk-ant-...your-actual-key...
VOYAGE_API_KEY=pa-...your-actual-key...
```

**Get Voyage AI key (free):** https://www.voyageai.com/

### 2. Run Setup Script
```bash
cd /Users/shawn/proj/lennyhub-rag
source .venv/bin/activate
python setup_rag.py --quick   # First 10 transcripts, ~5 min
```

Or for full dataset:
```bash
python setup_rag.py --parallel  # All 297 transcripts, ~25-35 min
```

### 3. Launch Web Interface
```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Notes
- Using **Claude Sonnet 4** for LLM
- Using **Voyage-3** for embeddings (1024 dimensions)
- Qdrant vector database runs locally (auto-started by setup script)
