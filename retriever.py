"""
Local JSON-based retrieval for the IoT RAG app.
"""

from __future__ import annotations

import json
import math
import os
import re
from getpass import getpass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from zhipuai import ZhipuAI


load_dotenv()

VECTOR_STORE_FILE = Path(__file__).parent / "vector_store.json"
RERANK_CANDIDATES = 15


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


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever:
    """Read the local vector store and provide retrieval methods."""

    def __init__(
        self,
        embedding_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.embedding_provider = get_embedding_provider()
        self.embedding_model = embedding_model or get_embedding_model()
        self.api_key = api_key or self._get_api_key()

        if self.embedding_provider == "zhipuai":
            self.client = ZhipuAI(api_key=self.api_key)
        elif get_embedding_base_url():
            self.client = OpenAI(api_key=self.api_key, base_url=get_embedding_base_url())
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.store = self._load_store()

    def _load_store(self) -> dict:
        """Load vector data from disk."""
        if not VECTOR_STORE_FILE.exists():
            raise ValueError("Vector store not found. Run ingest.py first.")
        with VECTOR_STORE_FILE.open("r", encoding="utf-8") as file:
            store = json.load(file)
        if not store.get("documents"):
            raise ValueError("Vector store is empty. Upload a PDF and ingest it first.")
        return store

    def _get_api_key(self) -> str:
        """Resolve the embedding API key."""
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

    def _embed_query(self, query: str) -> list[float]:
        """Embed the user query."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize mixed Chinese / English content for BM25."""
        return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Run dense retrieval and return Top-K chunks."""
        if not query.strip():
            return []

        query_embedding = self._embed_query(query)
        scored: list[dict[str, Any]] = []

        for item in self.store.get("documents", []):
            similarity = cosine_similarity(query_embedding, item["embedding"])
            metadata = item.get("metadata", {})
            scored.append(
                {
                    "text": item.get("text"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "score": similarity,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def search_with_rerank(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve candidates first, then rerank with BM25."""
        if not query.strip():
            return []

        candidates = self.search(query=query, k=max(RERANK_CANDIDATES, k))
        if not candidates:
            return []

        tokenized_corpus = [self._tokenize(item["text"]) for item in candidates]
        tokenized_query = self._tokenize(query)

        if not any(tokenized_corpus) or not tokenized_query:
            return candidates[:k]

        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)

        reranked = []
        for item, bm25_score in zip(candidates, bm25_scores):
            reranked.append(
                {
                    **item,
                    "score": float(bm25_score),
                    "_dense_score": item["score"],
                }
            )

        reranked.sort(key=lambda item: (item["score"], item["_dense_score"]), reverse=True)
        return [
            {
                "text": item["text"],
                "source": item["source"],
                "page": item["page"],
                "score": item["score"],
            }
            for item in reranked[:k]
        ]


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Backward-compatible wrapper."""
    retriever = Retriever()
    return retriever.search(query=query, k=top_k)
