"""
本地向量（embedding）服务。

用智源开源的中文向量模型 bge-small-zh-v1.5 把文本转成向量，
让检索能理解“语义相近”而不只是“字面相同”（如“报销”≈“费用申请”）。

模型跑在本地、不花钱、不需要 API。若环境未安装 sentence-transformers，
is_available() 返回 False，检索层会自动退回纯 BM25，保证应用仍可运行。
"""
from typing import List

import numpy as np


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# bge 官方建议：给“查询”加一段指令前缀，检索短句时效果更好；片段本身不加。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None            # 懒加载的模型实例
_available = None        # 是否安装了依赖（None=未检测）
_cache = {}              # {文本: 向量}，避免重复编码同一段文字


def is_available() -> bool:
    """检测运行环境是否装了 sentence-transformers。"""
    global _available
    if _available is None:
        try:
            import sentence_transformers  # noqa: F401
            _available = True
        except Exception:
            _available = False
    return _available


def _get_model():
    """首次调用时加载模型（约 100MB，首次会自动下载并缓存到本地）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """把一组文本编码为归一化向量（便于用点积直接算余弦相似度）。带缓存。"""
    model = _get_model()
    missing = [t for t in texts if t not in _cache]
    if missing:
        vectors = model.encode(missing, normalize_embeddings=True)
        for text, vector in zip(missing, vectors):
            _cache[text] = np.asarray(vector, dtype=np.float32)
    return np.array([_cache[t] for t in texts])


def embed_query(question: str) -> np.ndarray:
    """编码查询语句（自动加上 bge 推荐的指令前缀）。"""
    return embed_texts([QUERY_INSTRUCTION + question])[0]
