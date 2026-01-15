# Adding New Transcripts to LennyHub RAG

A guide for adding more podcast transcripts and updating the RAG system.

## 🚀 Quick Answer

```bash
# 1. Add new transcript files to the data folder
cp "New Transcript.txt" data/

# 2. Re-run the build script
python build_transcript_rag.py

# 3. Done! Query the updated system
python query_rag.py --interactive
```

## 📝 Detailed Steps

### Step 1: Prepare Your Transcript

Place your new transcript file(s) in the `data/` folder:

```bash
data/
├── Ada Chen Rekhi.txt          # Existing
├── Adam Fishman.txt            # Existing
├── New Guest Name.txt          # ← New file
└── Another Guest.txt           # ← Another new file
```

**File format requirements:**
- Plain text (.txt) format
- UTF-8 encoding
- Any size (the system will chunk it automatically)
- Filename will be used as document ID

### Step 2: Update the RAG System

Run the build script:

```bash
python build_transcript_rag.py
```

**What happens:**
1. Scans the `data/` folder for all `.txt` files
2. Checks which documents are already indexed
3. **Processes only new documents** (incremental update)
4. Extracts entities and relationships
5. Generates embeddings
6. Updates the knowledge graph
7. Saves to `rag_storage/`

### Step 3: Verify the Update

Check that your new transcripts were indexed:

```python
python query_rag.py "What does [New Guest Name] talk about?"
```

Or inspect the storage:

```bash
# Check document count
python -c "
import json
with open('rag_storage/kv_store_full_docs.json') as f:
    docs = json.load(f)
    print(f'Total documents: {len(docs)}')
    print('Documents:', list(docs.keys()))
"
```

## 🔄 Incremental vs Full Rebuild

### Incremental Update (Default)

The system automatically detects new files and only processes them:

**Pros:**
- ✅ Faster (only processes new content)
- ✅ Cheaper (only pays for new embeddings)
- ✅ Preserves existing data
- ✅ Automatic deduplication

**When it works:**
- Adding new transcript files
- Files not previously processed

**Time & Cost:**
- New transcript (~85KB): ~1-2 minutes, ~$0.15

### Full Rebuild (Optional)

If you want to completely rebuild from scratch:

```bash
# Delete existing storage
rm -rf rag_storage/

# Rebuild everything
python build_transcript_rag.py
```

**When to do a full rebuild:**
- Changing LLM model (GPT-4o-mini → GPT-4)
- Changing embedding model
- Changing chunk size or strategy
- Fixing data quality issues
- Major version upgrade of RAG-Anything

**Time & Cost:**
- All transcripts: ~3-5 minutes per transcript, ~$0.15 each

## 📊 System Updates After Adding Transcripts

### What Gets Updated

| Storage File | Updates |
|--------------|---------|
| `vdb_entities.json` | ✅ New entities added |
| `vdb_relationships.json` | ✅ New relationships added |
| `vdb_chunks.json` | ✅ New chunks added |
| `graph_chunk_entity_relation.graphml` | ✅ Graph expanded |
| `kv_store_full_docs.json` | ✅ New documents added |
| `kv_store_llm_response_cache.json` | ✅ Cache preserved |

### Example Growth

**Before (2 transcripts):**
- 101 entities
- 97 relationships
- 18 chunks
- 5MB total storage

**After adding 3 more transcripts:**
- ~250 entities (+149)
- ~240 relationships (+143)
- ~45 chunks (+27)
- ~12MB total storage (+7MB)

## 🎯 Best Practices

### 1. Prepare Transcripts

**Clean the text:**
```bash
# Remove timestamps if they're noisy
sed 's/([0-9][0-9]:[0-9][0-9]:[0-9][0-9])://g' transcript.txt > clean.txt

# Remove speaker labels if consistent format
# Keep them if they help identify who said what
```

**File naming:**
```bash
# Good naming (guest name)
data/Sarah Tavel.txt
data/Kevin Hale.txt

# Also good (topic or episode number)
data/Episode-123-Growth-Tactics.txt
data/Product-Market-Fit-Deep-Dive.txt

# Avoid special characters
data/Sarah_Tavel.txt          # Good
data/Sarah & Kevin (123).txt # Bad (special chars)
```

### 2. Batch Processing

If adding many transcripts, process them in batches:

```bash
# Add first batch
cp transcript1.txt transcript2.txt data/
python build_transcript_rag.py

# Wait for completion, then add next batch
cp transcript3.txt transcript4.txt data/
python build_transcript_rag.py
```

**Why batch?**
- Monitor for errors after each batch
- Control API costs
- Easier to debug issues

### 3. Monitor Processing

Watch the output for errors:

```bash
python build_transcript_rag.py 2>&1 | tee processing.log
```

Look for:
- ✅ "Successfully indexed [filename]"
- ❌ Error messages
- 📊 Statistics (entities found, chunks created)

### 4. Test After Adding

Always test queries on new content:

```bash
# Test that new content is retrievable
python query_rag.py "What does [new guest] say about [topic]?"

# Test that old content still works
python query_rag.py "What is a curiosity loop?"
```

## ⚠️ Common Issues

### Issue 1: File Not Processing

**Problem:** New file not detected

**Solutions:**
```bash
# Check file is in correct location
ls -la data/*.txt

# Check file permissions
chmod 644 data/"New Transcript.txt"

# Check for hidden characters in filename
ls -la data/ | cat -A
```

### Issue 2: Duplicate Content

**Problem:** Same content indexed twice

**Solution:**
The system uses document IDs to prevent duplicates:
- Document ID = `transcript-{filename_stem}`
- Same filename = replaces old version
- Different filename = new document

To replace existing transcript:
```bash
# Remove old version
rm data/"Old Name.txt"

# Add new version with same or different name
cp "Updated Transcript.txt" data/"Old Name.txt"

# Rebuild
python build_transcript_rag.py
```

### Issue 3: Out of Memory

**Problem:** Too many transcripts at once

**Solution:**
```python
# Edit build_transcript_rag.py
# Process files one at a time instead of all at once

for transcript_file in transcript_files:
    # Process one file
    await rag.insert_content_list(...)

    # Optional: Clear some memory
    import gc
    gc.collect()
```

### Issue 4: High API Costs

**Problem:** Unexpected OpenAI charges

**Solution:**
```bash
# Check what will be processed first
python -c "
from pathlib import Path
import json

# Check existing docs
with open('rag_storage/kv_store_full_docs.json') as f:
    existing = set(json.load(f).keys())

# Check all files
all_files = list(Path('data').glob('*.txt'))
new_files = [f for f in all_files
             if f'transcript-{f.stem}' not in existing]

print(f'Will process {len(new_files)} new files:')
for f in new_files:
    size_kb = f.stat().st_size / 1024
    print(f'  - {f.name} ({size_kb:.1f} KB, est. cost: ${size_kb * 0.002:.3f})')
"
```

## 💡 Advanced: Selective Processing

### Process Specific Files

Modify `build_transcript_rag.py` to process only specific files:

```python
# Original: processes all .txt files
transcript_files = list(transcript_dir.glob("*.txt"))

# Modified: process only specific files
specific_files = ["New Guest.txt", "Another Guest.txt"]
transcript_files = [
    transcript_dir / f for f in specific_files
    if (transcript_dir / f).exists()
]
```

### Skip Already Processed Files

The system already does this automatically, but you can verify:

```python
# Check if a document is already indexed
import json

with open('rag_storage/kv_store_full_docs.json') as f:
    docs = json.load(f)

filename = "Ada Chen Rekhi"
doc_id = f"transcript-{filename}"

if doc_id in docs:
    print(f"✓ {filename} is already indexed")
else:
    print(f"✗ {filename} needs to be processed")
```

## 📈 Scaling Considerations

### Storage Growth

Each transcript (~85KB) adds approximately:
- **Vectors**: ~1.5-2MB (entities + relationships + chunks)
- **Graph**: ~50-100KB (new entities and edges)
- **Documents**: ~85KB (original text)
- **Cache**: Grows with unique queries

**Total per transcript**: ~2-3MB

### Performance Impact

| Transcripts | Storage | Query Time | Build Time |
|-------------|---------|------------|------------|
| 1-10 | ~25MB | 200-400ms | 2-3 min each |
| 10-50 | ~125MB | 300-600ms | 2-3 min each |
| 50-100 | ~250MB | 400-800ms | 2-3 min each |
| 100+ | ~500MB+ | Consider optimization | 2-3 min each |

### When to Optimize

Consider switching to production vector DBs at:
- **50+ transcripts**: Switch to Qdrant or Milvus
- **100+ transcripts**: Use PostgreSQL for KV storage
- **500+ transcripts**: Enable sharding and distributed storage

## 🔧 Customizing the Build Process

### Modify Chunking Strategy

Edit `build_transcript_rag.py`:

```python
# Default: automatic chunking
await rag.insert_content_list(...)

# Custom: split by character (e.g., by speaker)
await rag.insert_content_list(
    content_list=content_list,
    split_by_character="\n\n",  # Split on double newline
    split_by_character_only=True
)
```

### Change Entity Extraction

Customize what gets extracted:

```python
# In build_transcript_rag.py, add to lightrag_kwargs:
lightrag_kwargs = {
    "entity_extract_max_gleaning": 2,  # Default: 1
    "max_entity_tokens": 128,          # Default: varies
}
```

### Parallel Processing

Process multiple files simultaneously:

```python
# In build_transcript_rag.py
import asyncio

# Process all files in parallel
tasks = [
    rag.insert_content_list(content_list, file_path=str(f))
    for f, content_list in zip(transcript_files, content_lists)
]
await asyncio.gather(*tasks)
```

**Caution:** This uses more API quota and may hit rate limits.

## 📚 Quick Reference

```bash
# Add transcripts
cp *.txt data/

# Update RAG (incremental)
python build_transcript_rag.py

# Full rebuild
rm -rf rag_storage/ && python build_transcript_rag.py

# Check what's indexed
python -c "import json; print(list(json.load(open('rag_storage/kv_store_full_docs.json')).keys()))"

# Test new content
python query_rag.py "query about new content"
```

## ❓ FAQ

**Q: Do I need to delete old data before adding new transcripts?**
A: No! The system handles incremental updates automatically.

**Q: Can I remove a transcript from the RAG?**
A: Not easily with current setup. Best approach: delete `rag_storage/` and rebuild without that file.

**Q: How long does it take to process one transcript?**
A: ~2-3 minutes per transcript (~85KB), depending on OpenAI API speed.

**Q: Can I add non-English transcripts?**
A: Yes! OpenAI's models support 100+ languages. The system will work with any UTF-8 text.

**Q: What if I update an existing transcript?**
A: Use the same filename. The system will replace the old version.

**Q: Can I add other document types (PDF, DOCX)?**
A: The current script only handles .txt files. You would need to convert other formats to plain text first, or modify the script to use RAG-Anything's document parsing features.

---

**Next Steps:**
- Add your new transcripts to `data/`
- Run `python build_transcript_rag.py`
- Test with new queries!
