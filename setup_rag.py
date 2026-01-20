#!/usr/bin/env python3
"""
Automated RAG Setup Script

This script handles the complete setup process:
1. Checks if Qdrant is installed
2. Starts Qdrant server
3. Builds embeddings from transcripts (sequential or parallel)
4. Tests the RAG system

Usage:
    python setup_rag.py                          # Process all transcripts (sequential)
    python setup_rag.py --quick                  # Process first 10 transcripts
    python setup_rag.py --max 50                 # Process first 50 transcripts
    python setup_rag.py --max 50 --parallel      # Process 50 transcripts in parallel (5x faster)
    python setup_rag.py --parallel --workers 10  # Custom concurrency (default: 5)
"""

import os
import sys
import time
import asyncio
import argparse
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv()


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 70)


def check_qdrant_installed():
    """Check if Qdrant binary is installed"""
    qdrant_bin = Path.home() / ".qdrant" / "qdrant"
    return qdrant_bin.exists()


def install_qdrant():
    """Install Qdrant binary"""
    print("Installing Qdrant locally...")
    install_script = Path("./install_qdrant_local.sh")

    if not install_script.exists():
        print("ERROR: install_qdrant_local.sh not found!")
        print("Please ensure you're running this script from the lennyhub-rag directory.")
        return False

    # Make script executable
    os.chmod(install_script, 0o755)

    # Run installation script
    result = subprocess.run(
        ["bash", str(install_script)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Qdrant installed successfully!")
        return True
    else:
        print("ERROR: Qdrant installation failed!")
        print(result.stderr)
        return False


def is_qdrant_running():
    """Check if Qdrant server is running"""
    try:
        # Qdrant's root endpoint returns version info
        response = requests.get("http://localhost:6333/", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_qdrant():
    """Start Qdrant server"""
    print("Starting Qdrant server...")
    start_script = Path("./start_qdrant.sh")

    if not start_script.exists():
        print("ERROR: start_qdrant.sh not found!")
        return False

    # Make script executable
    os.chmod(start_script, 0o755)

    # Run start script
    result = subprocess.run(
        ["bash", str(start_script)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Qdrant started successfully!")
        return True
    else:
        # Check if it's already running
        if "already running" in result.stdout.lower():
            print("✓ Qdrant is already running!")
            return True
        print("ERROR: Failed to start Qdrant!")
        print(result.stderr)
        return False


def wait_for_qdrant(timeout=30):
    """Wait for Qdrant to be ready"""
    print("Waiting for Qdrant to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_qdrant_running():
            print("✓ Qdrant is ready!")
            return True
        time.sleep(1)
        sys.stdout.write(".")
        sys.stdout.flush()

    print("\nERROR: Qdrant failed to start within timeout!")
    return False


def check_api_key():
    """Check if Anthropic and Voyage AI API keys are set"""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("\nPlease set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nOr add it to the .env file.")
        return False
    print("✓ Anthropic API key found!")

    if not voyage_key:
        print("ERROR: VOYAGE_API_KEY environment variable not set!")
        print("\nPlease set your Voyage AI API key (for embeddings):")
        print("  export VOYAGE_API_KEY='your-api-key-here'")
        print("\nGet one at: https://www.voyageai.com/")
        print("Or add it to the .env file.")
        return False
    print("✓ Voyage AI API key found!")
    return True


def get_already_processed_docs():
    """Get list of already processed document IDs"""
    doc_status_file = Path("./rag_storage/kv_store_full_docs.json")
    if doc_status_file.exists():
        try:
            with open(doc_status_file, 'r') as f:
                docs = json.load(f)
                return set(docs.keys())
        except Exception as e:
            print(f"Warning: Could not read docs file: {e}")
    return set()


# Global progress tracking for parallel mode
processed_count = 0
total_to_process = 0
lock = asyncio.Lock()


async def process_single_transcript_parallel(rag, transcript_file, semaphore):
    """Process a single transcript with semaphore control (parallel mode)"""
    global processed_count

    async with semaphore:  # Limit concurrent processing
        doc_id = f"transcript-{transcript_file.stem}"

        try:
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
                doc_id=doc_id
            )

            # Update progress counter
            async with lock:
                processed_count += 1
                current = processed_count

            print(f"[{current}/{total_to_process}] ✓ {transcript_file.name}")
            return True, transcript_file.name

        except Exception as e:
            print(f"[ERROR] ✗ {transcript_file.name}: {str(e)}")
            return False, transcript_file.name


async def build_rag_parallel(max_transcripts=None, workers=5):
    """Build RAG system with parallel processing"""
    global total_to_process, processed_count

    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.anthropic import anthropic_complete_if_cache, anthropic_embed
    from lightrag.utils import EmbeddingFunc
    from qdrant_config import get_lightrag_kwargs
    import numpy as np

    start_time = datetime.now()

    # Configure RAG system
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    # Set up LLM and embedding functions (using Claude + Voyage AI)
    print("Setting up LLM (Claude) and embedding (Voyage AI) functions...")

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
    lightrag_kwargs = get_lightrag_kwargs(verbose=False)

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

    # Get all transcript files
    transcript_dir = Path("./data")
    all_files = sorted(list(transcript_dir.glob("*.txt")))

    # Get already processed documents
    already_processed = get_already_processed_docs()
    print(f"Already processed: {len(already_processed)} transcripts")

    # Filter out already processed
    transcript_files = []
    for file in all_files:
        doc_id = f"transcript-{file.stem}"
        if doc_id not in already_processed:
            transcript_files.append(file)

    # Apply max limit
    if max_transcripts and len(transcript_files) > max_transcripts:
        transcript_files = transcript_files[:max_transcripts]

    total_to_process = len(transcript_files)

    if total_to_process == 0:
        print("\n✓ All transcripts already processed!")
        print(f"Total documents in system: {len(already_processed)}")
        rag.close()
        return True

    print(f"\nWill process: {total_to_process} new transcript(s)")
    print(f"Processing {workers} transcripts at a time (parallel mode)...")
    print("\nStarting parallel processing...")
    print("=" * 70 + "\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(workers)

    # Process all transcripts in parallel
    tasks = [
        process_single_transcript_parallel(rag, file, semaphore)
        for file in transcript_files
    ]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Calculate statistics
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
    failed = total_to_process - successful

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("PARALLEL PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Successfully processed: {successful}/{total_to_process}")
    print(f"Failed: {failed}")
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    if successful > 0:
        print(f"Average: {duration/successful:.1f} seconds per transcript")
    print(f"Total documents in system: {len(already_processed) + successful}")
    print("=" * 70)

    # Close RAG system
    rag.close()

    return successful > 0


async def build_rag(max_transcripts=None):
    """Build RAG system with sequential processing"""
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.anthropic import anthropic_complete_if_cache, anthropic_embed
    from lightrag.utils import EmbeddingFunc
    from qdrant_config import get_lightrag_kwargs
    import numpy as np

    # Configure RAG system
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    # Set up LLM and embedding functions (using Claude + Voyage AI)
    print("Setting up LLM (Claude) and embedding (Voyage AI) functions...")

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
    lightrag_kwargs = get_lightrag_kwargs(verbose=False)

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

    # Get transcript files
    transcript_dir = Path("./data")
    all_files = sorted(list(transcript_dir.glob("*.txt")))

    # Get already processed documents
    already_processed = get_already_processed_docs()
    print(f"Already processed: {len(already_processed)} transcripts")

    # Filter out already processed
    transcript_files = []
    for file in all_files:
        doc_id = f"transcript-{file.stem}"
        if doc_id not in already_processed:
            transcript_files.append(file)

    if max_transcripts:
        transcript_files = transcript_files[:max_transcripts]

    print(f"\nFound {len(transcript_files)} transcript(s) to process")

    if len(transcript_files) == 0:
        print("\n✓ All transcripts already processed!")
        print(f"Total documents in system: {len(already_processed)}")
        rag.close()
        return True

    # Ensure LightRAG is initialized
    await rag._ensure_lightrag_initialized()

    # Process each transcript sequentially
    print("\nProcessing transcripts sequentially...")
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"\n[{i}/{len(transcript_files)}] Processing: {transcript_file.name}")

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
        print(f"✓ Successfully indexed!")

    print_header("RAG System Built Successfully!")

    # Test with a sample question
    print("Testing RAG system with sample question...\n")
    print("Question: What is a curiosity loop and how does it work?")
    print("-" * 70 + "\n")

    try:
        response = await rag.aquery(
            "What is a curiosity loop and how does it work?",
            mode="hybrid"
        )
        print("Answer:")
        print(response)
    except Exception as e:
        print(f"Warning: Test query failed: {e}")

    # Close RAG system
    rag.close()

    return True


async def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup RAG system with Qdrant")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Process only first 10 transcripts (for quick testing)"
    )
    parser.add_argument(
        "--max",
        type=int,
        help="Maximum number of transcripts to process"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (5x faster, processes multiple transcripts at once)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers for parallel mode (default: 5, max: 10)"
    )
    args = parser.parse_args()

    # Determine max transcripts
    max_transcripts = None
    if args.quick:
        max_transcripts = 10
    elif args.max:
        max_transcripts = args.max

    # Validate workers
    if args.workers > 10:
        print("Warning: Max workers is 10 to avoid rate limiting. Using 10.")
        args.workers = 10
    elif args.workers < 1:
        print("Warning: Workers must be at least 1. Using 1.")
        args.workers = 1

    mode_text = "PARALLEL" if args.parallel else "SEQUENTIAL"
    print_header(f"RAG-Anything Setup with Local Qdrant ({mode_text})")

    if args.parallel:
        print(f"⚡ Parallel mode enabled with {args.workers} workers")
        print(f"   This is ~{args.workers}x faster than sequential mode!\n")

    # Step 1: Check if Qdrant is installed
    print_step(1, "Checking Qdrant Installation")
    if not check_qdrant_installed():
        print("Qdrant not found. Installing...")
        if not install_qdrant():
            print("\nSetup failed! Please install Qdrant manually:")
            print("  ./install_qdrant_local.sh")
            return 1
    else:
        print("✓ Qdrant is already installed!")

    # Step 2: Check API keys
    print_step(2, "Checking Anthropic and Voyage AI API Keys")
    if not check_api_key():
        return 1

    # Step 3: Start Qdrant
    print_step(3, "Starting Qdrant Server")
    if is_qdrant_running():
        print("✓ Qdrant is already running!")
    else:
        if not start_qdrant():
            return 1
        if not wait_for_qdrant():
            return 1

    # Step 4: Build RAG with embeddings
    print_step(4, "Building RAG System and Creating Embeddings")

    mode_desc = f"first {max_transcripts} transcripts" if max_transcripts else "all transcripts"
    processing_mode = f"parallel ({args.workers} workers)" if args.parallel else "sequential"
    print(f"Processing {mode_desc} in {processing_mode} mode...\n")

    try:
        if args.parallel:
            success = await build_rag_parallel(max_transcripts=max_transcripts, workers=args.workers)
        else:
            success = await build_rag(max_transcripts=max_transcripts)

        if not success:
            return 1
    except Exception as e:
        print(f"\nERROR: Failed to build RAG system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Final summary
    print_header("Setup Complete!")
    print("Your RAG system is ready to use!\n")
    print("Next steps:")
    print("  1. Query the system:")
    print('     python query_rag.py "Your question here"')
    print("     python query_rag.py --interactive\n")
    print("  2. Launch visual interface:")
    print("     ./run_streamlit.sh\n")
    print("  3. Query with sources:")
    print('     python query_with_sources.py "Your question"\n')
    print("  4. Check Qdrant status:")
    print("     ./status_qdrant.sh")
    print("     curl http://localhost:6333/\n")
    print("  5. View dashboard:")
    print("     http://localhost:6333/dashboard\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
