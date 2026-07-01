import streamlit as st

from services.document_service import (
    read_uploaded_file,
    split_text_into_chunks,
    get_document_stats,
)
from services.retrieval_service import retrieve_top_k
from services.answer_service import generate_answer, build_markdown_answer
from services.answer_service import MODEL as ANSWER_MODEL, _llm_enabled
from utils.storage import save_qa_record, load_history


st.set_page_config(
    page_title="AI企业知识库问答助手",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.10), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef6f5 100%);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        margin-bottom: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(15, 118, 110, 0.12);
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    }
    .brand {
        font-weight: 850;
        color: #0f172a;
        letter-spacing: 0.02em;
    }
    .topbar-right {
        color: #0f766e;
        font-weight: 700;
        font-size: 13px;
    }
    .console-shell {
        display: grid;
        grid-template-columns: 0.98fr 1.38fr;
        gap: 22px;
        align-items: stretch;
        margin-bottom: 22px;
    }
    .left-panel {
        padding: 28px;
        border-radius: 28px;
        background: linear-gradient(180deg, #ffffff 0%, #f0fdfa 100%);
        border: 1px solid rgba(20, 184, 166, 0.20);
        box-shadow: 0 18px 44px rgba(13, 148, 136, 0.10);
    }
    .right-panel {
        padding: 28px;
        border-radius: 28px;
        background: #0f172a;
        color: #f8fafc;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
        position: relative;
        overflow: hidden;
    }
    .right-panel:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.16);
        right: -70px;
        top: -80px;
    }
    .eyebrow {
        color: #0f766e;
        font-size: 13px;
        font-weight: 850;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 40px;
        font-weight: 900;
        line-height: 1.18;
        color: #0f172a;
        margin-bottom: 14px;
    }
    .hero-copy {
        color: #475569;
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 18px;
    }
    .tag-light {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        margin: 0 8px 8px 0;
        background: #ccfbf1;
        color: #115e59;
        border: 1px solid #99f6e4;
        font-size: 13px;
        font-weight: 760;
    }
    .assistant-title {
        font-size: 21px;
        font-weight: 850;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
    }
    .assistant-line {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.75;
        margin-bottom: 18px;
        position: relative;
        z-index: 1;
    }
    .flow-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin: 15px 0;
        position: relative;
        z-index: 1;
    }
    .flow-num {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        background: #14b8a6;
        color: #ffffff;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-weight: 850;
        flex-shrink: 0;
    }
    .flow-text b {
        color: #ffffff;
    }
    .flow-text {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.55;
    }
    .workspace-grid {
        display: grid;
        grid-template-columns: 0.88fr 1.12fr;
        gap: 20px;
        margin-top: 10px;
    }
    .section-label {
        font-size: 13px;
        color: #0f766e;
        font-weight: 850;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 10px 0 8px 0;
    }
    .section-title {
        font-size: 24px;
        color: #0f172a;
        font-weight: 900;
        margin-bottom: 10px;
    }
    .hint {
        color: #64748b;
        font-size: 14px;
        line-height: 1.65;
        margin-bottom: 12px;
    }
    .status-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }
    .doc-card {
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        border-radius: 16px;
        padding: 13px;
        margin: 10px 0;
    }
    .answer-panel {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }
    .answer-block {
        border-left: 4px solid #14b8a6;
        padding: 12px 14px;
        margin: 12px 0;
        background: #f8fafc;
        border-radius: 0 14px 14px 0;
    }
    .answer-block h4 {
        margin: 0 0 8px 0;
        color: #0f172a;
    }
    .answer-block p {
        margin: 0;
        color: #475569;
        line-height: 1.7;
    }
    .value-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        min-height: 145px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .value-card h4 {
        margin: 0 0 10px 0;
        color: #0f172a;
    }
    .value-card p {
        margin: 0;
        color: #475569;
        line-height: 1.65;
        font-size: 14px;
    }
    .history-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 16px;
    }
    @media (max-width: 900px) {
        .console-shell, .workspace-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_value_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="value-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_block(title: str, content: str) -> None:
    st.markdown(
        f"""
        <div class="answer-block">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
if "answer" not in st.session_state:
    st.session_state.answer = None
if "question" not in st.session_state:
    st.session_state.question = ""


st.markdown(
    """
    <div class="topbar">
        <div class="brand">KnowledgeOps Assistant</div>
        <div class="topbar-right">Local Retrieval · Source-grounded Answer · MVP</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="console-shell">
        <div class="left-panel">
            <div class="eyebrow">Enterprise Knowledge Console</div>
            <div class="hero-title">AI企业知识库问答助手</div>
            <div class="hero-copy">
                上传内部资料，系统自动切分为知识片段，并根据问题检索相关内容，生成带引用来源的回答。
            </div>
            <span class="tag-light">RAG思路</span>
            <span class="tag-light">文档问答</span>
            <span class="tag-light">知识库检索</span>
            <span class="tag-light">引用来源</span>
            <span class="tag-light">Mock稳定演示</span>
        </div>
        <div class="right-panel">
            <div class="assistant-title">工作流不是生成内容，而是先找依据</div>
            <div class="assistant-line">
                这个页面刻意设计成企业知识库控制台，和短视频生产工具区分开：更强调文档、检索、证据和复核。
            </div>
            <div class="flow-item">
                <div class="flow-num">1</div>
                <div class="flow-text"><b>上传资料</b><br>解析 TXT / MD 文档，生成可检索知识片段。</div>
            </div>
            <div class="flow-item">
                <div class="flow-num">2</div>
                <div class="flow-text"><b>检索证据</b><br>根据问题返回 Top-3 相关片段，而不是直接编答案。</div>
            </div>
            <div class="flow-item">
                <div class="flow-num">3</div>
                <div class="flow-text"><b>输出答案</b><br>展示结论、依据、下一步建议和引用来源。</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 控制台说明")
    st.markdown("### 项目定位")
    st.write("面向企业内部资料的轻量化知识库问答助手。")

    st.markdown("### 适合场景")
    st.markdown(
        """
        - 企业制度问答
        - 内部资料检索
        - 项目文档查询
        - 培训材料问答
        - 面试项目演示
        """
    )

    st.markdown("### 技术栈")
    st.markdown(
        """
        - Python
        - Streamlit
        - 文本切分
        - 本地检索
        - RAG 思路
        - Markdown 导出
        """
    )

    st.markdown("### 当前版本")
    if _llm_enabled():
        st.success(f"回答引擎：真实大模型（{ANSWER_MODEL}）+ 混合检索。")
    else:
        st.info("回答引擎：本地检索 + Mock 演示。设置 ANTHROPIC_API_KEY 后自动切换为真实大模型。")


input_col, output_col = st.columns([0.92, 1.08], gap="large")

with input_col:
    st.markdown('<div class="section-label">STEP 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">上传资料</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">上传 TXT 或 MD 文档。建议先用 sample_docs/company_ai_policy.md 测试。</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "上传 TXT 或 MD 文档",
        type=["txt", "md"],
        help="建议上传公司制度、项目说明、培训资料等纯文本文件。",
    )

    if uploaded_file is not None:
        text = read_uploaded_file(uploaded_file)
        chunks = split_text_into_chunks(text)
        stats = get_document_stats(uploaded_file.name, text, chunks)

        st.session_state.document_text = text
        st.session_state.chunks = chunks
        st.session_state.doc_stats = stats
        st.session_state.answer = None
        st.session_state.retrieved_chunks = []

        st.success("文档已解析并切分。")

    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown(
            f"""
            <div class="doc-card">
                <b>当前文档：</b>{stats["filename"]}<br>
                <b>字数：</b>{stats["char_count"]}　
                <b>知识片段：</b>{stats["chunk_count"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("预览知识片段"):
            for chunk in st.session_state.chunks[:3]:
                st.markdown(f"**{chunk['chunk_id']}**")
                st.write(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

    st.markdown('<div class="section-label">STEP 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">输入问题</div>', unsafe_allow_html=True)

    question = st.text_area(
        "问题",
        placeholder="例如：公司使用 AI 工具时有哪些注意事项？",
        height=130,
    )

    if st.button(
        "检索资料并生成回答",
        type="primary",
        use_container_width=True,
        disabled=not bool(st.session_state.chunks),
    ):
        st.session_state.question = question.strip()
        retrieved_chunks = retrieve_top_k(question, st.session_state.chunks, top_k=3)
        answer = generate_answer(question, retrieved_chunks)

        st.session_state.retrieved_chunks = retrieved_chunks
        st.session_state.answer = answer

        save_qa_record(
            {
                "question": question.strip(),
                "answer": answer,
                "chunks": retrieved_chunks,
                "document": st.session_state.doc_stats,
            }
        )

        st.success("已完成检索并生成回答。")

    if not st.session_state.chunks:
        st.info("请先上传文档，再输入问题。")

    st.markdown("### 最近问答")
    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.caption("暂无历史记录。")
    else:
        for index, item in enumerate(history[:5], start=1):
            title = item.get("question", "未命名问题")[:24]
            with st.expander(f"{index}. {title}"):
                st.caption(item.get("created_at", ""))
                st.markdown("**问题**")
                st.write(item.get("question", ""))
                st.markdown("**简要结论**")
                st.write(item.get("answer", {}).get("conclusion", ""))
    st.markdown("</div>", unsafe_allow_html=True)

with output_col:
    st.markdown('<div class="section-label">STEP 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">回答与证据</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">回答会同时展示依据摘要和引用片段，方便人工复核。</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.answer:
        answer = st.session_state.answer
        chunks = st.session_state.retrieved_chunks

        st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
        render_answer_block("简要结论", answer.get("conclusion", ""))
        render_answer_block("依据摘要", answer.get("evidence", ""))
        render_answer_block("建议下一步", answer.get("next_step", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        markdown = build_markdown_answer(
            st.session_state.question,
            answer,
            chunks,
        )
        st.download_button(
            "导出当前问答结果 Markdown",
            data=markdown,
            file_name="knowledge_base_answer.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown("### 引用来源")
        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"引用片段 {index}｜{chunk.get('chunk_id', '')}｜相关分数 {chunk.get('score', '')}"):
                st.write(chunk.get("text", ""))
    else:
        st.info("上传文档并完成检索后，这里会展示回答、依据摘要和引用来源。")


st.markdown("---")
st.markdown("## 项目价值")
value_cols = st.columns(4)
value_cards = [
    (
        "业务痛点",
        "企业内部资料分散，制度、项目说明、培训材料查询成本较高。",
    ),
    (
        "检索优先",
        "先检索相关片段，再生成回答，强调依据而不是直接生成。",
    ),
    (
        "可信输出",
        "回答同时展示引用片段，便于人工复核，降低无依据回答风险。",
    ),
    (
        "后续扩展",
        "后续可接入真实大模型 API、向量数据库、权限管理和多文档知识库。",
    ),
]

for col, (title, body) in zip(value_cols, value_cards):
    with col:
        render_value_card(title, body)
