from typing import List, Dict


def generate_mock_answer(question: str, chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Generate a stable mock answer based on retrieved chunks.
    This is not a real LLM call. It is designed for MVP demonstration.
    """
    if not question.strip():
        return {
            "conclusion": "请输入问题后再生成回答。",
            "evidence": "当前没有可用问题。",
            "next_step": "上传文档并输入与文档内容相关的问题。",
        }

    if not chunks:
        return {
            "conclusion": "当前没有检索到可用文档片段。",
            "evidence": "请先上传 TXT 或 MD 文档。",
            "next_step": "上传资料后重新提问。",
        }

    best_chunk = chunks[0].get("text", "").replace("\n", " ")
    conclusion = (
        f"根据已上传文档，问题“{question}”可以优先参考检索到的高相关片段。"
        "当前版本使用本地检索与 Mock 回答，用于展示 RAG 问答流程。"
    )

    evidence = (
        "系统已根据问题从文档中检索出相关内容。"
        f"最高相关片段摘要：{best_chunk[:180]}"
        + ("..." if len(best_chunk) > 180 else "")
    )

    next_step = (
        "建议人工复核引用片段后再使用答案。后续可接入 Claude、OpenAI 或 DeepSeek API，"
        "把检索片段作为上下文传入模型，生成更自然的回答。"
    )

    return {
        "conclusion": conclusion,
        "evidence": evidence,
        "next_step": next_step,
    }


def build_markdown_answer(question: str, answer: Dict[str, str], chunks: List[Dict[str, str]]) -> str:
    """
    Build Markdown content for export.
    """
    lines = [
        "# AI企业知识库问答结果",
        "",
        "## 问题",
        question,
        "",
        "## 简要结论",
        answer.get("conclusion", ""),
        "",
        "## 依据摘要",
        answer.get("evidence", ""),
        "",
        "## 建议下一步",
        answer.get("next_step", ""),
        "",
        "## 引用来源",
    ]

    for index, chunk in enumerate(chunks, start=1):
        lines.extend(
            [
                "",
                f"### 引用片段 {index}",
                f"- 片段编号：{chunk.get('chunk_id', '')}",
                f"- 相关分数：{chunk.get('score', '')}",
                "",
                chunk.get("text", ""),
            ]
        )

    return "\n".join(lines)
