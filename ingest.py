"""
Parse PDFs, split text, embed chunks, and persist them to a local JSON store.
"""

from __future__ import annotations

import hashlib
import json
import os
from getpass import getpass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import fitz
from openai import OpenAI
from zhipuai import ZhipuAI

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
VECTOR_STORE_FILE = Path(__file__).parent / "vector_store.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 100


def get_embedding_provider() -> str:
    """Resolve the configured embedding provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("ZHIPUAI_API_KEY"):
        return "zhipuai"
    return "openai"


def get_embedding_model() -> str:
    """Resolve the embedding model name."""
    if get_embedding_provider() == "zhipuai":
        return os.getenv("ZHIPUAI_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "embedding-3"))
    return os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))


def get_embedding_base_url() -> str:
    """Resolve an optional OpenAI-compatible embedding base URL."""
    return os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).strip()


def compute_md5(file_path: Path) -> str:
    """Compute the file MD5 for deduplication."""
    md5 = hashlib.md5()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def normalize_text(text: str) -> str:
    """Lightly clean extracted text."""
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract plain text from each PDF page."""
    pages: list[dict] = []

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = normalize_text(page.get_text("text"))
            if not text:
                continue

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "text": text,
                }
            )

    return pages


def split_pages_to_chunks(pages: Iterable[dict]) -> list[dict]:
    """Split page text into chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for chunk_text in split_texts:
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            chunks.append(
                {
                    "text": cleaned_chunk,
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

    return chunks


def chunked(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    """Yield fixed-size batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_api_key(api_key: str | None = None) -> str:
    """Resolve the embedding API key."""
    if api_key:
        return api_key

    env_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if env_key:
        return env_key

    input_key = getpass("Please enter the embedding API key: ").strip()
    if not input_key:
        raise ValueError("Missing embedding API key.")
    return input_key


def build_embedding_client(api_key: str) -> OpenAI | ZhipuAI:
    """Build the embedding client for the configured provider."""
    provider = get_embedding_provider()
    if provider == "zhipuai":
        return ZhipuAI(api_key=api_key)
    if get_embedding_base_url():
        return OpenAI(api_key=api_key, base_url=get_embedding_base_url())
    return OpenAI(api_key=api_key)


def load_vector_store() -> dict:
    """Load the local vector store file."""
    if not VECTOR_STORE_FILE.exists():
        return {
            "embedding_provider": get_embedding_provider(),
            "embedding_model": get_embedding_model(),
            "documents": [],
        }

    with VECTOR_STORE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_vector_store(store: dict) -> None:
    """Persist the local vector store."""
    with VECTOR_STORE_FILE.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False)


def should_skip_file(store: dict, pdf_path: Path, file_md5: str) -> bool:
    """Skip ingest if the file MD5 matches existing stored chunks."""
    for item in store.get("documents", []):
        metadata = item.get("metadata", {})
        if metadata.get("source") == pdf_path.name and metadata.get("file_md5") == file_md5:
            return True
    return False


def delete_existing_file_chunks(store: dict, pdf_path: Path) -> None:
    """Delete existing chunks for the same source file."""
    store["documents"] = [
        item
        for item in store.get("documents", [])
        if item.get("metadata", {}).get("source") != pdf_path.name
    ]


def embed_batch(client: OpenAI | ZhipuAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = client.embeddings.create(
        model=get_embedding_model(),
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_file(store: dict, embedding_client: OpenAI | ZhipuAI, pdf_path: Path) -> int:
    """Ingest one PDF into the local vector store."""
    file_md5 = compute_md5(pdf_path)

    if should_skip_file(store, pdf_path, file_md5):
        print(f"Skipping unchanged file: {pdf_path.name}")
        return 0

    delete_existing_file_chunks(store, pdf_path)

    pages = extract_pdf_pages(pdf_path)
    if not pages:
        print(f"No text extracted from: {pdf_path.name}")
        return 0

    chunks = split_pages_to_chunks(pages)
    if not chunks:
        print(f"No chunks generated from: {pdf_path.name}")
        return 0

    for batch in chunked(chunks, EMBED_BATCH_SIZE):
        texts = [item["text"] for item in batch]
        embeddings = embed_batch(embedding_client, texts)

        for item, embedding in zip(batch, embeddings):
            metadata = {
                **item["metadata"],
                "file_md5": file_md5,
            }
            store["documents"].append(
                {
                    "id": f"{pdf_path.stem}:{file_md5}:{metadata['page']}:{metadata['chunk_index']}",
                    "text": item["text"],
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    print(f"Ingested file: {pdf_path.name} ({len(chunks)} chunks)")
    return len(chunks)


def ingest_pdfs(pdf_paths: Iterable[Path] | None = None, api_key: str | None = None) -> int:
    """Ingest multiple PDFs and persist the updated vector store."""
    files = list(pdf_paths) if pdf_paths is not None else sorted(DATA_DIR.glob("*.pdf"))
    if not files:
        return 0

    resolved_api_key = get_api_key(api_key)
    embedding_client = build_embedding_client(resolved_api_key)
    store = load_vector_store()
    store["embedding_provider"] = get_embedding_provider()
    store["embedding_model"] = get_embedding_model()

    total_chunks = 0
    for pdf_path in files:
        total_chunks += ingest_file(store, embedding_client, pdf_path)

    save_vector_store(store)
    return total_chunks


def list_knowledge_base_documents() -> list[dict[str, int | str]]:
    """Return source files and chunk counts from the vector store."""
    store = load_vector_store()
    summary: dict[str, int] = {}
    for item in store.get("documents", []):
        source = item.get("metadata", {}).get("source")
        if source:
            summary[source] = summary.get(source, 0) + 1

    return [
        {"source": source, "chunk_count": chunk_count}
        for source, chunk_count in sorted(summary.items())
    ]


def main() -> None:
    """CLI entrypoint."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}.")
        return

    total_chunks = ingest_pdfs(pdf_files)
    print(f"Total ingested chunks: {total_chunks}")


if __name__ == "__main__":
    main()
