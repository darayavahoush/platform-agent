"""Retrieval layer for MissionModule: embeds `kb/*.md` docs with
sentence-transformers and searches them with FAISS (both already in
requirements.txt). Small corpus (a handful of short docs) so a flat index
is plenty — swap IndexFlatIP for an IVF/HNSW index if the KB grows past a
few thousand chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    source: str


class Retriever:
    def __init__(
        self,
        kb_path: str = "member4_llm_rag/kb",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
    ):
        from sentence_transformers import SentenceTransformer  # lazy import
        import faiss
        import numpy as np

        self._np = np
        self._faiss = faiss
        self.kb_path = Path(kb_path)
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = self._load_and_chunk(chunk_size)

        if self.chunks:
            embeddings = self.model.encode(
                [c.text for c in self.chunks], normalize_embeddings=True
            ).astype("float32")
            self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized dot product
            self.index.add(embeddings)
        else:
            self.index = None

    def _load_and_chunk(self, chunk_size: int) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not self.kb_path.exists():
            return chunks

        paths = sorted(self.kb_path.glob("*.md")) + sorted(self.kb_path.glob("*.txt"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            buf = ""
            for para in paragraphs:
                if len(buf) + len(para) > chunk_size and buf:
                    chunks.append(Chunk(text=buf.strip(), source=path.name))
                    buf = ""
                buf += para + "\n\n"
            if buf.strip():
                chunks.append(Chunk(text=buf.strip(), source=path.name))
        return chunks

    def retrieve(self, query: str, k: int = 3) -> List[Chunk]:
        if not self.chunks or self.index is None:
            return []
        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        k = min(k, len(self.chunks))
        _, idx = self.index.search(query_emb, k)
        return [self.chunks[i] for i in idx[0] if i != -1]