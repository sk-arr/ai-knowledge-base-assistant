"""
检索评测脚本：用 eval/eval_set.json 里的用例，量化混合检索的效果。

指标：
- Top-1 命中率：期望关键词出现在排名第 1 的片段里
- Top-3 命中率：期望关键词出现在前 3 个片段里
- 拒答准确率：无关问题（expect_relevant=false）被正确判为“未找到相关内容”

用法（在项目根目录）：
    python eval/evaluate.py
"""
import json
import os
import sys

# 允许从项目根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_service import split_text_into_chunks  # noqa: E402
from services.retrieval_service import retrieve_top_k, has_relevant  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP_K = 3


def load_chunks(doc_paths):
    """加载并切分所有文档，片段标记来源。"""
    chunks = []
    for rel_path in doc_paths:
        path = os.path.join(ROOT, rel_path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for chunk in split_text_into_chunks(text):
            chunk["source"] = os.path.basename(rel_path)
            chunks.append(chunk)
    return chunks


def main():
    with open(os.path.join(os.path.dirname(__file__), "eval_set.json"), encoding="utf-8") as f:
        spec = json.load(f)

    chunks = load_chunks(spec["docs"])
    print(f"文档 {len(spec['docs'])} 个，切出片段 {len(chunks)} 个，用例 {len(spec['cases'])} 条\n")

    pos_total = top1_hits = top3_hits = 0
    src_total = src_hits = 0
    neg_total = neg_correct = 0

    for case in spec["cases"]:
        question = case["question"]
        results = retrieve_top_k(question, chunks, top_k=TOP_K)

        if case.get("expect_relevant") is False:
            # 负例：应判为不相关
            neg_total += 1
            ok = not has_relevant(results)
            neg_correct += int(ok)
            print(f"[拒答] {'✓' if ok else '✗'}  {question}")
            continue

        # 正例：期望关键词应出现在检索结果里
        pos_total += 1
        keyword = case["expect_keyword"]
        texts = [r.get("text", "") for r in results]
        hit1 = keyword in texts[0] if texts else False
        hit3 = any(keyword in t for t in texts)
        top1_hits += int(hit1)
        top3_hits += int(hit3)

        # 来源路由：Top-1 命中的片段是否来自期望的文档
        src_mark = ""
        expect_source = case.get("expect_source")
        if expect_source:
            src_total += 1
            src_ok = bool(results) and results[0].get("source") == expect_source
            src_hits += int(src_ok)
            src_mark = f"，来源={'对' if src_ok else '错'}"

        mark = "✓" if hit3 else "✗"
        print(f"[检索] {mark}  {question}  (期望含“{keyword}”，Top1={'中' if hit1 else '否'}{src_mark})")

    print("\n=== 汇总 ===")
    if pos_total:
        print(f"Top-1 命中率：{top1_hits}/{pos_total} = {top1_hits / pos_total:.0%}")
        print(f"Top-3 命中率：{top3_hits}/{pos_total} = {top3_hits / pos_total:.0%}")
    if src_total:
        print(f"来源路由命中率：{src_hits}/{src_total} = {src_hits / src_total:.0%}")
    if neg_total:
        print(f"拒答准确率：{neg_correct}/{neg_total} = {neg_correct / neg_total:.0%}")


if __name__ == "__main__":
    main()
