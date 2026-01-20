"""
Quick RAG build script - processes first 10 transcripts for testing

Usage:
    python build_rag_quick.py
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.anthropic import anthropic_complete_if_cache, anthropic_embed
from lightrag.utils import EmbeddingFunc
from qdrant_config import get_lightrag_kwargs
import numpy as np

# Load environment variables
load_dotenv()

MAX_TRANSCRIPTS = 10  # Process only first 10 transcripts

async def main():
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("\nPlease set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    if not os.getenv("VOYAGE_API_KEY"):
        print("ERROR: VOYAGE_API_KEY environment variable not set!")
        print("\nPlease set your Voyage AI API key:")
        print("  export VOYAGE_API_KEY='your-api-key-here'")
        return

    print("=" * 70)
    print("QUICK RAG BUILD - First 10 Transcripts (Claude + Voyage AI)")
    print("=" * 70)
    print()

    # Configure RAG system
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    # Set up LLM and embedding functions (using Claude + Voyage AI)
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await anthropic_complete_if_cache(
            "claude-sonnet-4-20250514",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_tokens=4096,
            **kwargs
        )

    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await anthropic_embed(texts, model="voyage-3")

    # Get Qdrant configuration
    lightrag_kwargs = get_lightrag_kwargs()

    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # Voyage-3 uses 1024 dimensions
            max_token_size=8192,
            func=embedding_func
        ),
        lightrag_kwargs=lightrag_kwargs
    )

    # Ensure LightRAG is initialized
    await rag._ensure_lightrag_initialized()

    # Get transcript files
    transcript_dir = Path("./data")
    all_files = sorted(list(transcript_dir.glob("*.txt")))[:MAX_TRANSCRIPTS]

    print(f"\nFound {len(all_files)} transcripts to process:")
    for i, file in enumerate(all_files, 1):
        print(f"  {i}. {file.name}")

    print(f"\n{'=' * 70}")
    print("Processing transcripts...")
    print(f"{'=' * 70}\n")

    # Process each transcript
    for i, transcript_file in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] Processing: {transcript_file.name}")

        # Read transcript content
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Insert into RAG system
        content_list = [{
            "type": "text",
            "text": content,
            "page_idx": 0
        }]

        await rag.insert_content_list(
            content_list=content_list,
            file_path=str(transcript_file),
            doc_id=f"transcript-{transcript_file.stem}"
        )
        print(f"✓ Successfully indexed\n")

    print(f"{'=' * 70}")
    print("RAG system built successfully!")
    print(f"{'=' * 70}\n")

    # Test with a sample question
    print("Testing with sample question...\n")
    print(f"{'=' * 70}")
    print("Question: What is a curiosity loop and how does it work?")
    print(f"{'=' * 70}\n")

    response = await rag.aquery(
        "What is a curiosity loop and how does it work?",
        mode="hybrid"
    )

    print("Answer:")
    print(response)
    print()

    print(f"{'=' * 70}")
    print("Build complete!")
    print(f"{'=' * 70}\n")

    print("To query the system:")
    print('  python query_rag.py "Your question here"')
    print("  python query_rag.py --interactive")
    print()

    # Close RAG system
    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
