import re
from collections import Counter
from typing import List, Dict


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for Chinese and English mixed text.
    It extracts Chinese characters and English/number words.
    """
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return chinese_chars + english_words


def score_chunk(question: str, chunk_text: str) -> float:
    """
    Score a chunk by token overlap.
    This is a lightweight local retrieval method for MVP demonstration.
    """
    question_tokens = tokenize(question)
    chunk_tokens = tokenize(chunk_text)

    if not question_tokens or not chunk_tokens:
        return 0.0

    question_counter = Counter(question_tokens)
    chunk_counter = Counter(chunk_tokens)

    overlap = 0
    for token, count in question_counter.items():
        overlap += min(count, chunk_counter.get(token, 0))

    unique_bonus = len(set(question_tokens) & set(chunk_tokens)) * 0.5
    normalized_score = (overlap + unique_bonus) / max(len(question_tokens), 1)

    return round(normalized_score, 4)


def retrieve_top_k(question: str, chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve top-k chunks by simple keyword overlap score.
    """
    scored_chunks = []

    for chunk in chunks:
        score = score_chunk(question, chunk.get("text", ""))
        scored_chunk = dict(chunk)
        scored_chunk["score"] = score
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    positive_results = [item for item in scored_chunks if item["score"] > 0]
    if positive_results:
        return positive_results[:top_k]

    return scored_chunks[:top_k]
