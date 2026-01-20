"""
Build RAG system from podcast transcripts using RAG-Anything

Requirements:
- ANTHROPIC_API_KEY environment variable must be set
- VOYAGE_API_KEY environment variable must be set (for embeddings)

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    export VOYAGE_API_KEY="your-voyage-key"
    python build_transcript_rag.py
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

# Load environment variables from .env file
load_dotenv()

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

    # Configure RAG system
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,  # Not needed for text transcripts
        enable_table_processing=False,  # Not needed for text transcripts
        enable_equation_processing=False,  # Not needed for text transcripts
    )

    # Set up LLM and embedding functions (using Claude + Voyage AI)
    print("Setting up LLM (Claude) and embedding (Voyage AI) functions...")

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await anthropic_complete_if_cache(
            "claude-sonnet-4-20250514",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_tokens=4096,
            **kwargs
        )

    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await anthropic_embed(
            texts,
            model="voyage-3"
        )

    # Get Qdrant configuration
    lightrag_kwargs = get_lightrag_kwargs()

    # Initialize RAG system with LLM functions
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

    # Read transcript files
    transcript_dir = Path("./data")
    transcript_files = list(transcript_dir.glob("*.txt"))

    print(f"\nFound {len(transcript_files)} transcript files:")
    for file in transcript_files:
        print(f"  - {file.name}")

    # Ensure LightRAG is initialized
    await rag._ensure_lightrag_initialized()

    # Process each transcript
    print("\nProcessing transcripts and building RAG...")
    for transcript_file in transcript_files:
        print(f"\nProcessing: {transcript_file.name}")

        # Read transcript content
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Insert into RAG system using LightRAG's insert method
        # Format content as a list with text items
        content_list = [
            {
                "type": "text",
                "text": content,
                "page_idx": 0
            }
        ]

        await rag.insert_content_list(
            content_list=content_list,
            file_path=str(transcript_file),
            doc_id=f"transcript-{transcript_file.stem}"
        )
        print(f"✓ Successfully indexed {transcript_file.name}")

    print("\n" + "="*60)
    print("RAG system built successfully!")
    print("="*60)

    # Sample questions for Ada Chen Rekhi transcript
    sample_questions = [
        "What is a curiosity loop and how does it work?",
        "What are Ada's personal values?",
        "What advice does Ada give about building an early career?",
        "What is the 'eating your vegetables' concept?",
        "Should you start a company with your partner?",
        "What is the explore and exploit framework for career development?",
        "What is Adam Fishman's growth competency model?",
        "What are the four main components of the growth competency model?",
        "Why is onboarding important for growth?",
        "What is the PMF framework for choosing a company to work at?",
    ]

    print("\nTesting RAG with sample questions...\n")

    # Test with a few sample questions
    for i, question in enumerate(sample_questions[:3], 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}")

        # Query the RAG system
        response = await rag.aquery(question, mode="hybrid")
        print(f"\nAnswer:\n{response}\n")

    print("\n" + "="*60)
    print("Sample questions saved to: sample_questions.txt")
    print("="*60)

    # Save all sample questions to a file
    with open("sample_questions.txt", "w") as f:
        f.write("Sample Questions for Transcript RAG\n")
        f.write("="*60 + "\n\n")
        for i, question in enumerate(sample_questions, 1):
            f.write(f"{i}. {question}\n")

        f.write("\n\nHow to query the RAG:\n")
        f.write("-" * 60 + "\n")
        f.write("from raganything import RAGAnything, RAGAnythingConfig\n")
        f.write("import asyncio\n\n")
        f.write("config = RAGAnythingConfig(working_dir='./rag_storage')\n")
        f.write("rag = RAGAnything(config=config)\n\n")
        f.write("# Query the RAG\n")
        f.write("response = await rag.aquery('Your question here', mode='hybrid')\n")
        f.write("print(response)\n")

    print("\nYou can query the RAG system using the code shown in sample_questions.txt")

    # Close RAG system
    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
