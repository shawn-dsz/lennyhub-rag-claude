# Technical Overview - LennyHub RAG

A deep dive into the architecture, vector database, and knowledge graph implementation of the LennyHub RAG system.

## 🏗️ Architecture Overview

LennyHub RAG is built on two main frameworks:
- **RAG-Anything**: Multimodal document processing pipeline
- **LightRAG**: Knowledge graph-based retrieval system

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Interface                         │
│                     (query_rag.py)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG-Anything                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              LightRAG Core                            │  │
│  │  ┌────────────┐  ┌───────────┐  ┌────────────────┐  │  │
│  │  │ Embeddings │  │   LLM     │  │ Knowledge Graph│  │  │
│  │  │  (OpenAI)  │  │(GPT-4o-mini)│ │   (GraphML)   │  │  │
│  │  └────────────┘  └───────────┘  └────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Storage Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Vector Store │  │  KV Storage  │  │  Graph Storage  │  │
│  │(NanoVectorDB)│  │    (JSON)    │  │   (GraphML)     │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🗄️ Vector Database: NanoVectorDB

### What is NanoVectorDB?

NanoVectorDB is a **lightweight, in-memory vector database** designed for fast similarity search. It's the default vector store in LightRAG.

**Key Features:**
- **In-memory**: Fast access, no external database required
- **Persistent**: Saves to JSON files on disk
- **Compressed storage**: Uses Float16 + zlib + Base64 encoding
- **Cosine similarity**: Default metric for vector comparison
- **Async operations**: Non-blocking queries and updates

### Storage Files

The system uses **three separate vector databases**:

```
rag_storage/
├── vdb_entities.json          # 1.2MB - Entity embeddings
├── vdb_relationships.json     # 1.2MB - Relationship embeddings
└── vdb_chunks.json            # 307KB - Text chunk embeddings
```

**Why three databases?**
- **Entities**: Named entities extracted from text (people, organizations, concepts)
- **Relationships**: Connections between entities (works-at, created-by, etc.)
- **Chunks**: Raw text segments for direct content retrieval

### Vector Format

Each vector is stored in compressed format:
1. Convert float32 → float16 (50% size reduction)
2. Compress with zlib
3. Encode with Base64 for JSON storage
4. Store with metadata (id, created_at, distance)

**Example:**
```json
{
  "__id__": "Ada Chen Rekhi",
  "vector": "eJyNVktu3DAM...",  // Compressed embedding
  "__created_at__": 1768466228,
  "entity_type": "person",
  "description": "Executive coach..."
}
```

### Embedding Model

- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimensions**: 1536
- **Cost**: ~$0.02 per 1M tokens
- **Max tokens**: 8192 per request

### Alternative Vector Databases

LightRAG supports multiple vector DB backends:
- **NanoVectorDB** (default) - In-memory, JSON storage
- **Qdrant** - Production-grade vector DB
- **Milvus** - Scalable vector search
- **PostgreSQL + pgvector** - SQL-based vector storage
- **MongoDB Atlas** - Document + vector search
- **FAISS** - Facebook's similarity search library

## 🕸️ Knowledge Graph

### Graph Structure

The knowledge graph is stored in **GraphML format** (107KB):

```
rag_storage/
└── graph_chunk_entity_relation.graphml
```

GraphML is an XML-based format compatible with NetworkX, Neo4j, and other graph tools.

### Graph Components

#### Nodes (Entities)
Each node represents an entity with:
- **entity_id**: Unique identifier
- **entity_type**: person, organization, concept, method, etc.
- **description**: LLM-generated entity description
- **source_id**: References to source chunks
- **file_path**: Origin transcript
- **created_at**: Timestamp

**Example Entity:**
```xml
<node id="Ada Chen Rekhi">
  <data key="entity_type">person</data>
  <data key="description">
    Executive coach and co-founder of Notejoy...
  </data>
  <data key="source_id">chunk-3ef01b14...</data>
  <data key="file_path">Ada Chen Rekhi.txt</data>
</node>
```

#### Edges (Relationships)
Each edge represents a relationship with:
- **source/target**: Connected entity IDs
- **description**: Relationship type and context
- **weight**: Relationship strength (0-1)
- **keywords**: Associated keywords
- **source_id**: References to source chunks

**Example Relationship:**
```xml
<edge source="Ada Chen Rekhi" target="Curiosity Loops">
  <data key="description">
    Ada Chen Rekhi created and teaches the Curiosity Loops framework
  </data>
  <data key="weight">0.95</data>
  <data key="keywords">framework, method, decision-making</data>
</edge>
```

### Current Graph Stats

For the two transcripts:
- **101 nodes** (entities)
- **97 edges** (relationships)
- **18 text chunks** (semantic segments)

### Entity Types

Common entity types extracted:
- **person**: Ada Chen Rekhi, Adam Fishman, Lenny
- **organization**: Microsoft, SurveyMonkey, LinkedIn, Lyft, Patreon
- **concept**: Curiosity Loops, Explore & Exploit, PMF Framework
- **method**: Values Exercise, Growth Competency Model
- **location**: Silicon Valley
- **product**: Notejoy

## 🔍 Query Modes Explained

### 1. Naive Search
Simple keyword-based search without semantic understanding.
```python
response = await rag.aquery("curiosity loop", mode="naive")
```
- Fastest but least accurate
- Good for exact term matches
- No vector or graph usage

### 2. Local Search
Uses entity and relationship embeddings for local context.
```python
response = await rag.aquery("What is a curiosity loop?", mode="local")
```

**Process:**
1. Generate query embedding
2. Find similar entities in `vdb_entities.json`
3. Find similar relationships in `vdb_relationships.json`
4. Retrieve associated chunks
5. Synthesize answer with LLM

**Best for:** Specific factual questions about entities or concepts

### 3. Global Search
Uses knowledge graph traversal for broader understanding.
```python
response = await rag.aquery("career advice", mode="global")
```

**Process:**
1. Extract keywords from query
2. Find relevant entities in graph
3. Traverse graph to find connected concepts
4. Aggregate information across multiple hops
5. Synthesize comprehensive answer

**Best for:** Broad topics requiring multiple concepts

### 4. Hybrid Search (Recommended)
Combines local and global approaches.
```python
response = await rag.aquery("What advice does Ada give?", mode="hybrid")
```

**Process:**
1. Run local search → get specific entities/relationships
2. Run global search → get broader context
3. Merge results with deduplication
4. Rank by relevance
5. Synthesize unified answer

**Best for:** Most queries - balances specificity and context

### 5. Mix Search
Integrates knowledge graph and vector retrieval.
```python
response = await rag.aquery("career frameworks", mode="mix")
```

Similar to hybrid but with different weighting and merging strategies.

## 🗂️ Storage Architecture

### Complete Storage Structure

```
rag_storage/
├── Vector Stores (3.7MB total)
│   ├── vdb_entities.json          # 1.2MB - Entity vectors
│   ├── vdb_relationships.json     # 1.2MB - Relationship vectors
│   └── vdb_chunks.json            # 307KB - Chunk vectors
│
├── Knowledge Graph
│   └── graph_chunk_entity_relation.graphml  # 107KB - Entity graph
│
├── Key-Value Stores (JSON)
│   ├── kv_store_full_docs.json         # 154KB - Original documents
│   ├── kv_store_text_chunks.json       # 100KB - Chunked text
│   ├── kv_store_full_entities.json     # 2.6KB - Entity metadata
│   ├── kv_store_full_relations.json    # 6.3KB - Relationship metadata
│   ├── kv_store_entity_chunks.json     # 23KB - Entity-chunk mapping
│   ├── kv_store_relation_chunks.json   # 24KB - Relation-chunk mapping
│   └── kv_store_doc_status.json        # 2.9KB - Processing status
│
└── Cache
    └── kv_store_llm_response_cache.json  # 856KB - LLM response cache
```

### Storage Breakdown

| Component | Size | Purpose |
|-----------|------|---------|
| Vector embeddings | 3.7MB | Semantic search |
| Knowledge graph | 107KB | Entity relationships |
| Full documents | 154KB | Source text |
| Text chunks | 100KB | Semantic segments |
| Metadata | 58KB | Entity/relation info |
| LLM cache | 856KB | Response caching |
| **Total** | **~5MB** | Complete system |

## 🔄 Query Processing Pipeline

### Step-by-Step Query Flow

```
User Query: "What is a curiosity loop?"
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Query Preprocessing              │
│    - Extract keywords               │
│    - Generate query embedding       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. Entity Search (Local)            │
│    - Search vdb_entities.json       │
│    - Top-k similar entities         │
│    - Threshold: cosine > 0.2        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. Relationship Search (Local)      │
│    - Search vdb_relationships.json  │
│    - Find entity connections        │
│    - Get relationship descriptions  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. Graph Traversal (Global)         │
│    - Load GraphML knowledge graph   │
│    - Find related entities          │
│    - Traverse edges (1-2 hops)      │
│    - Aggregate connected concepts   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 5. Chunk Retrieval                  │
│    - Map entities → chunks          │
│    - Search vdb_chunks.json         │
│    - Retrieve relevant text         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 6. Context Merging                  │
│    - Deduplicate results            │
│    - Rank by relevance              │
│    - Truncate to token limit        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 7. LLM Synthesis                    │
│    - Build prompt with context      │
│    - Generate answer (GPT-4o-mini)  │
│    - Cache response                 │
└────────────┬────────────────────────┘
             │
             ▼
    Return Answer to User
```

### Query Example

**Input:** "What is a curiosity loop and how does it work?"

**Processing:**
1. **Keywords extracted:** curiosity loop, mechanism, process
2. **Entities found (top-k=40):**
   - "Curiosity loop" (entity_type: concept)
   - "Ada Chen Rekhi" (entity_type: person)
   - "Decision making" (entity_type: concept)
3. **Relationships found:**
   - Ada → created → Curiosity loops
   - Curiosity loops → helps_with → Decision making
   - Curiosity loops → requires → Specific questions
4. **Chunks retrieved:** 16 text segments from Ada's transcript
5. **Answer synthesized** with full context about definition, steps, and examples

## 🎯 Performance Characteristics

### Query Performance

| Query Mode | Speed | Accuracy | Use Case |
|------------|-------|----------|----------|
| naive | 50ms | Low | Exact matches |
| local | 200ms | Medium | Specific entities |
| global | 300ms | High | Broad topics |
| hybrid | 400ms | Highest | General queries |
| mix | 350ms | High | Complex queries |

### Cost Analysis

**Initial Build:**
- Embeddings: ~$0.04 (170KB text)
- Entity extraction: ~$0.20 (GPT-4o-mini)
- **Total: ~$0.25**

**Per Query:**
- Embedding: ~$0.0001
- LLM synthesis: ~$0.001-0.01
- **Average: ~$0.002 per query**

**Cached Query:** $0 (responses cached)

### Scalability

Current setup handles:
- ✅ 2 transcripts (~170KB text)
- ✅ 101 entities
- ✅ 97 relationships
- ✅ 18 chunks

Estimated limits (without optimization):
- 📊 ~50 transcripts (~4MB text)
- 📊 ~2,500 entities
- 📊 ~2,000 relationships
- 📊 ~500 chunks

For larger datasets:
- Switch to Qdrant or Milvus
- Use PostgreSQL for KV storage
- Enable reranking models
- Implement chunking strategies

## 🔧 Configuration Options

### Vector Database Options

```python
# Default: NanoVectorDB (in-memory)
config = RAGAnythingConfig(working_dir="./rag_storage")

# Switch to Qdrant (scalable)
from qdrant_client import QdrantClient
lightrag_kwargs = {
    "vector_storage": "QdrantVectorDBStorage",
    "vector_db_storage_cls_kwargs": {
        "url": "http://localhost:6333",
        "collection_name": "lennyhub"
    }
}

# Switch to PostgreSQL (SQL-based)
lightrag_kwargs = {
    "vector_storage": "PostgreSQLStorage",
    "vector_db_storage_cls_kwargs": {
        "connection_string": "postgresql://user:pass@localhost/db"
    }
}
```

### LLM Options

```python
# Default: GPT-4o-mini
async def llm_model_func(prompt, **kwargs):
    return await openai_complete_if_cache(
        "gpt-4o-mini", prompt, **kwargs
    )

# Switch to GPT-4
async def llm_model_func(prompt, **kwargs):
    return await openai_complete_if_cache(
        "gpt-4", prompt, **kwargs
    )

# Switch to local model (Ollama)
from lightrag.llm.ollama import ollama_complete
async def llm_model_func(prompt, **kwargs):
    return await ollama_complete(
        "llama3", prompt, **kwargs
    )
```

### Embedding Options

```python
# Default: text-embedding-3-small (1536 dims)
async def embedding_func(texts):
    return await openai_embed(
        texts, model="text-embedding-3-small"
    )

# Switch to text-embedding-3-large (3072 dims, better quality)
async def embedding_func(texts):
    return await openai_embed(
        texts, model="text-embedding-3-large"
    )
```

## 🚀 Advanced Features

### Response Caching

All LLM responses are cached in `kv_store_llm_response_cache.json`:
- **Key**: Hash of (mode + query + context)
- **Value**: LLM response
- **Benefit**: Instant responses for repeated queries, $0 cost

### Reranking (Optional)

Enable reranking for better result quality:
```python
lightrag_kwargs = {
    "enable_rerank": True,
    "rerank_model_func": rerank_model  # Custom reranker
}
```

Reranking improves relevance but adds ~100ms latency.

### Streaming Responses

For long answers, enable streaming:
```python
async for chunk in rag.aquery_stream(question, mode="hybrid"):
    print(chunk, end="", flush=True)
```

### Custom Entity Extraction

Customize entity types:
```python
custom_entity_types = [
    "FRAMEWORK", "TOOL", "METRIC", "STRATEGY"
]
# Configure in LightRAG initialization
```

## 📊 Monitoring & Debugging

### Check Storage Stats

```python
# Check vector database sizes
print(f"Entities: {len(rag.entities_vdb)}")
print(f"Relationships: {len(rag.relationships_vdb)}")
print(f"Chunks: {len(rag.chunks_vdb)}")

# Check graph size
graph = rag.lightrag.chunk_entity_relation_graph
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

### Query Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# See detailed query processing
response = await rag.aquery(question, mode="hybrid")
```

### Clear Cache

```bash
# Clear LLM response cache
rm rag_storage/kv_store_llm_response_cache.json

# Rebuild entire system
rm -rf rag_storage/
python build_transcript_rag.py
```

## 🔗 References

- **RAG-Anything**: https://github.com/HKUDS/RAG-Anything
- **LightRAG**: https://github.com/HKUDS/LightRAG
- **NanoVectorDB**: https://github.com/gusye1234/nano-vectordb
- **GraphML Format**: http://graphml.graphdrawing.org/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

---

**Last Updated:** January 2026
