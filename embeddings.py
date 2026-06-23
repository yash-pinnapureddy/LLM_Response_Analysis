"""Utility script for generating OpenAI embeddings and computing similarity."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
client = OpenAI()


def get_embedding(text: str, model: str = MODEL_NAME) -> list[float]:
    """Return an embedding for a single text input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two vector-like inputs."""
    vec1_arr = np.asarray(vec1, dtype=float)
    vec2_arr = np.asarray(vec2, dtype=float)

    norm1 = np.linalg.norm(vec1_arr)
    norm2 = np.linalg.norm(vec2_arr)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input vectors must not be all zeros")

    return float(np.dot(vec1_arr, vec2_arr) / (norm1 * norm2))


def compare_texts(text1: str, text2: str, model: str = MODEL_NAME) -> float:
    """Encode two texts and return their cosine similarity."""
    embedding1 = get_embedding(text1, model=model)
    embedding2 = get_embedding(text2, model=model)
    return cosine_similarity(embedding1, embedding2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between two texts using OpenAI embeddings."
    )
    parser.add_argument("text1", help="First text input")
    parser.add_argument("text2", help="Second text input")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Embedding model to use (default: text-embedding-3-small)",
    )

    args = parser.parse_args()
    similarity = compare_texts(args.text1, args.text2, model=args.model)
    print(f"Cosine similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()
