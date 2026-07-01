import re
from typing import List, Dict

import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from services import embedding_service


# 常见中文停用词 + 标点，切词后过滤掉，避免“的/是/公司”这类高频词干扰排序。
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "也", "就", "都", "而", "及",
    "与", "或", "一个", "没有", "我们", "你们", "他们", "这个", "那个", "什么",
    "哪些", "如何", "怎么", "可以", "需要", "对于", "关于", "以及", "一些",
    "这", "那", "着", "吗", "呢", "吧", "啊", "把", "被", "让", "给", "为",
    "请问", "请", "问",
}

# 混合检索权重：向量（语义）占比略高于 BM25（字面）。
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


def tokenize(text: str) -> List[str]:
    """
    用 jieba 对中英文混合文本分词，并过滤停用词与纯标点。
    对比旧版“中文按单字切”，这里切出的是词（如“AI工具”“人工复核”），匹配更精准。
    """
    tokens = []
    for token in jieba.lcut(text.lower()):
        token = token.strip()
        # 只保留含中文/字母/数字的 token，丢掉空白和标点
        if not token or not re.search(r"[一-鿿A-Za-z0-9]", token):
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _bm25_scores(question: str, chunks: List[Dict[str, str]]) -> np.ndarray:
    """按片段顺序返回 BM25 原始分数（字面匹配）。"""
    corpus_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
    query_tokens = tokenize(question)
    if not query_tokens or not any(corpus_tokens):
        return np.zeros(len(chunks))
    bm25 = BM25Okapi(corpus_tokens)
    return np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)


def _vector_scores(question: str, chunks: List[Dict[str, str]]) -> np.ndarray:
    """按片段顺序返回向量余弦相似度（语义匹配）。向量已归一化，点积即余弦。"""
    chunk_vectors = embedding_service.embed_texts([c.get("text", "") for c in chunks])
    query_vector = embedding_service.embed_query(question)
    return chunk_vectors @ query_vector


def _normalize(scores: np.ndarray) -> np.ndarray:
    """按最大值归一到 [0, 1]，让 BM25 和向量分数能相加。全 0 时保持全 0。"""
    top = scores.max()
    if top <= 0:
        return np.zeros_like(scores)
    return scores / top


def retrieve_top_k(question: str, chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
    """
    混合检索：BM25（字面）+ bge 向量（语义）加权融合后取 top_k。

    - 字面匹配保证关键词精确命中；
    - 语义匹配能召回换了说法的内容（如“报销”≈“费用申请”）；
    - 若环境未装向量依赖，自动退回纯 BM25，应用仍可运行。
    """
    if not chunks:
        return []

    bm25_raw = _bm25_scores(question, chunks)

    if embedding_service.is_available():
        vector_raw = _vector_scores(question, chunks)
        combined = BM25_WEIGHT * _normalize(bm25_raw) + VECTOR_WEIGHT * _normalize(vector_raw)
    else:
        combined = bm25_raw

    scored_chunks = []
    for chunk, score in zip(chunks, combined):
        scored_chunk = dict(chunk)
        scored_chunk["score"] = round(float(score), 4)
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # 优先返回有正分（真正匹配上）的片段；都为 0 时才退回前 top_k
    positive_results = [item for item in scored_chunks if item["score"] > 0]
    if positive_results:
        return positive_results[:top_k]

    return scored_chunks[:top_k]
