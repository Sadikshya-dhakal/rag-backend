"""Two selectable chunking strategies, both implementing a shared `Chunker`
protocol so new strategies can be dropped in without touching call sites.

1. FixedSizeChunker        - fixed-width sliding window over raw characters
                              with configurable overlap. Simple, predictable,
                              language-agnostic.
2. RecursiveParagraphChunker - splits on paragraph boundaries first, then
                              sentences, then hard character limits, only
                              recursing into a smaller separator when a piece
                              is still too large. Produces chunks that respect
                              natural text boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.document import ChunkingStrategy


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


class FixedSizeChunker:
    """Sliding window over characters: [0, size), [size - overlap, 2*size - overlap), ..."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(index=index, text=piece))
                index += 1
            if end == length:
                break
            start += stride
        return chunks


class RecursiveParagraphChunker:
    """Paragraph-first recursive splitter.

    Splits on blank lines (paragraphs). Any paragraph still longer than
    `chunk_size` is recursively split on sentence boundaries, then on plain
    whitespace as a last resort. Adjacent small paragraphs are packed
    together up to `chunk_size`, with the last `chunk_overlap` characters of
    a chunk carried into the next one for context continuity.
    """

    _SEPARATORS = ["\n\n", r"(?<=[.!?])\s+", " "]

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        pieces = self._recursive_split(text, separator_idx=0)
        packed = self._pack(pieces)
        return [Chunk(index=i, text=p) for i, p in enumerate(packed)]

    def _recursive_split(self, text: str, separator_idx: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size or separator_idx >= len(self._SEPARATORS):
            return [text]

        separator = self._SEPARATORS[separator_idx]
        parts = [p for p in re.split(separator, text) if p and p.strip()]
        if len(parts) <= 1:
            return self._recursive_split(text, separator_idx + 1)

        result: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                result.extend(self._recursive_split(part, separator_idx + 1))
            else:
                result.append(part.strip())
        return result

    def _pack(self, pieces: list[str]) -> list[str]:
        """Greedily pack consecutive small pieces into ~chunk_size windows,
        carrying a small overlap tail forward for context continuity.
        """
        packed: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                packed.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{tail}\n\n{piece}".strip() if tail else piece
            else:
                # Single piece already <= chunk_size guaranteed by recursion,
                # but guard anyway.
                packed.append(piece)
                current = ""

        if current:
            packed.append(current)
        return packed


def get_chunker(strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int) -> Chunker:
    if strategy == ChunkingStrategy.FIXED_SIZE:
        return FixedSizeChunker(chunk_size, chunk_overlap)
    if strategy == ChunkingStrategy.RECURSIVE_PARAGRAPH:
        return RecursiveParagraphChunker(chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
