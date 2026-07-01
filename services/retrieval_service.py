import re
from typing import List, Dict

import jieba
from rank_bm25 import BM25Okapi


# 常见中文停用词 + 标点，切词后过滤掉，避免“的/是/公司”这类高频词干扰排序。
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "也", "就", "都", "而", "及",
    "与", "或", "一个", "没有", "我们", "你们", "他们", "这个", "那个", "什么",
    "哪些", "如何", "怎么", "可以", "需要", "对于", "关于", "以及", "一些",
    "这", "那", "着", "吗", "呢", "吧", "啊", "把", "被", "让", "给", "为",
    "请问", "请", "问",
}


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


def retrieve_top_k(question: str, chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
    """
    用 BM25 对片段排序，返回相关度最高的 top_k 个片段。

    BM25 是搜索引擎常用的排序算法，相比旧版的“token 重叠计数”多做了两件事：
      1. IDF 加权：越罕见的词越能定位答案，权重越高；
      2. 长度归一化：长片段不会因为字数多就天然占便宜。
    """
    if not chunks:
        return []

    # 对每个片段分词，作为 BM25 的语料库
    corpus_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
    query_tokens = tokenize(question)

    # 问题为空，或所有片段都切不出有效词，退回原始顺序（保持可用）
    if not query_tokens or not any(corpus_tokens):
        return [dict(chunk, score=0.0) for chunk in chunks[:top_k]]

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        scored_chunk = dict(chunk)
        scored_chunk["score"] = round(float(score), 4)
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # 优先返回有正分（真正匹配上）的片段；都为 0 时才退回前 top_k
    positive_results = [item for item in scored_chunks if item["score"] > 0]
    if positive_results:
        return positive_results[:top_k]

    return scored_chunks[:top_k]
