import os
import json
from typing import List, Dict


# 大模型只允许返回这三个字段，正好接住前端的三段式展示（结论/依据/下一步）。
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string", "description": "简要结论，直接回答问题"},
        "evidence": {"type": "string", "description": "依据摘要，指出结论来自哪些资料内容"},
        "next_step": {"type": "string", "description": "建议下一步或人工复核提示"},
    },
    "required": ["conclusion", "evidence", "next_step"],
    "additionalProperties": False,
}

# 强约束“只根据资料回答、无依据就说没找到”，对应项目主打的可信输出/防幻觉。
SYSTEM_PROMPT = (
    "你是企业知识库问答助手。只能依据用户提供的【资料片段】回答问题，"
    "不得编造资料中不存在的内容。若资料无法回答，请在 conclusion 中明确说明"
    "“文档中未找到相关依据”。evidence 要引用资料中的关键信息，next_step 给出"
    "人工复核或补充资料的建议。全部用简体中文。"
)

# 默认用最新的 Claude Opus 4.8；可用环境变量 KB_MODEL 覆盖。
MODEL = os.getenv("KB_MODEL", "claude-opus-4-8")


def _llm_enabled() -> bool:
    """有 API key 且未强制 Mock 时，走真实大模型。"""
    if os.getenv("KB_FORCE_MOCK") == "1":
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _generate_llm_answer(question: str, chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """把检索到的片段作为上下文传给 Claude，生成结构化回答。"""
    import anthropic

    context = "\n\n".join(
        f"[资料片段{i}] {chunk.get('text', '')}" for i, chunk in enumerate(chunks, start=1)
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": f"以下是检索到的资料：\n{context}\n\n请回答问题：{question}",
            }
        ],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def generate_answer(question: str, chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """
    统一入口：优先真实大模型，失败或未配置 key 时自动回退到 Mock，保证演示不中断。
    """
    if not question.strip() or not chunks:
        # 空问题/无片段的提示语与 Mock 一致，直接复用
        return generate_mock_answer(question, chunks)

    # 相关度门槛：检索到的片段都不够相关，就直说没找到——
    # 既避免把无关内容当证据（防幻觉），也省掉一次无意义的大模型调用。
    from services import retrieval_service

    if not retrieval_service.has_relevant(chunks):
        return {
            "conclusion": "文档中未找到与该问题相关的内容。",
            "evidence": "检索到的片段与问题相关度过低，未作为回答依据。",
            "next_step": "请换一种说法提问，或上传包含相关内容的文档后重试。",
        }

    if _llm_enabled():
        try:
            return _generate_llm_answer(question, chunks)
        except Exception as error:  # 网络/鉴权/解析等任何异常都优雅降级
            fallback = generate_mock_answer(question, chunks)
            fallback["conclusion"] = (
                f"（大模型调用失败，已回退本地 Mock：{error}）" + fallback["conclusion"]
            )
            return fallback

    return generate_mock_answer(question, chunks)


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
