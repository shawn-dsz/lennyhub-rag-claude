# Installation Guide - LennyHub RAG

A RAG (Retrieval-Augmented Generation) system built on Lenny's Podcast transcripts using RAG-Anything.

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 2. Installation

#### Option A: Using pip (Recommended)

```bash
# Navigate to the project directory
cd lennyhub-rag

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
# Create a new conda environment
conda create -n lennyhub python=3.11 -y
conda activate lennyhub

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file and add your API key
nano .env
```

Add this line to your `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 4. Build the RAG System

Run the build script to index the transcripts:

```bash
python build_transcript_rag.py
```

This will:
- Process both transcript files (~170KB total)
- Create embeddings using OpenAI's text-embedding-3-small
- Build a knowledge graph with entities and relationships
- Store everything in `./rag_storage/`

**Expected time:** 2-3 minutes
**Expected cost:** ~$0.25 in OpenAI API calls

### 5. Query the System

#### Interactive Mode (Recommended)

```bash
python query_rag.py --interactive
```

Then type your questions and get instant answers!

#### Single Question Mode

```bash
python query_rag.py "What is a curiosity loop?"
```

#### Example Queries Mode

```bash
python query_rag.py
```

## Verification

To verify your installation is working:

```bash
# Check Python version
python --version  # Should be 3.9+

# Check if raganything is installed
pip show raganything

# Test the query script
python query_rag.py "What is a curiosity loop and how does it work?"
```

## Troubleshooting

### ModuleNotFoundError: No module named 'raganything'

```bash
pip install --upgrade raganything
```

### OPENAI_API_KEY not set

Make sure your `.env` file exists and contains:
```
OPENAI_API_KEY=sk-...your-key...
```

### ImportError: cannot import name 'openai_complete_if_cache'

This is already fixed in the provided scripts. If you encounter this, make sure you're using the latest versions:
```bash
pip install --upgrade lightrag-hku openai
```

### Rate Limiting

If you hit OpenAI rate limits:
- Wait a few minutes and try again
- Upgrade your OpenAI account tier
- The system caches responses, so re-running won't cost extra

### Memory Issues

If you encounter memory issues during indexing:
- Close other applications
- Process one transcript at a time by modifying `build_transcript_rag.py`

## Project Structure

```
lennyhub-rag/
├── data/
│   ├── Ada Chen Rekhi.txt
│   └── Adam Fishman.txt
├── rag_storage/              # Generated after building
│   ├── graph_chunk_entity_relation.graphml
│   ├── vdb_entities.json
│   ├── vdb_relationships.json
│   └── ...
├── build_transcript_rag.py   # Build the RAG system
├── query_rag.py              # Query interface
├── sample_questions.txt      # 70+ example questions
├── requirements.txt          # Python dependencies
├── .env                      # Your API keys (create this)
├── .env.example              # Template for .env
├── INSTALL.md                # This file
└── README_TRANSCRIPT_RAG.md  # Full documentation
```

## Next Steps

1. Review `sample_questions.txt` for 70+ example questions
2. Read `README_TRANSCRIPT_RAG.md` for detailed documentation
3. Start querying the system with your own questions!

## Cost Estimate

- **Initial build:** ~$0.25
  - Embeddings: ~$0.02 per transcript
  - Knowledge graph extraction: ~$0.20 total
- **Per query:** ~$0.001-0.01 (depending on complexity)
- **Cached queries:** Free (responses are cached)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the full documentation in `README_TRANSCRIPT_RAG.md`
3. Make sure all dependencies are installed: `pip list | grep -E "raganything|lightrag|openai"`

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade raganything lightrag-hku openai
```

Note: After upgrading, you may need to rebuild the RAG system if the data format changed.
