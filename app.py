import streamlit as st

from services.document_service import (
    read_uploaded_file,
    split_text_into_chunks,
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --fg: #EDEDEF;
        --fg-muted: #8A8F98;
        --accent: #5E6AD2;
        --accent-bright: #6872D9;
        --border: rgba(255,255,255,0.06);
        --border-hover: rgba(255,255,255,0.10);
    }

    html, body, .stApp, [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Layer 1 — deep-space base gradient */
    .stApp {
        background: radial-gradient(ellipse at top, #0a0a0f 0%, #050506 50%, #020203 100%);
        color: var(--fg);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        position: relative;
        z-index: 1;
    }

    /* Layers 2-4 — grid overlay + floating ambient light blobs */
    .ambient { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
    .ambient .grid {
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 64px 64px;
        mask-image: radial-gradient(ellipse at 50% 0%, #000 40%, transparent 80%);
    }
    .ambient .blob { position: absolute; border-radius: 50%; filter: blur(140px); }
    .ambient .b1 {
        width: 900px; height: 1000px; top: -320px; left: calc(50% - 450px);
        background: radial-gradient(circle, rgba(94,106,210,0.28), transparent 70%);
        animation: float 9s ease-in-out infinite;
    }
    .ambient .b2 {
        width: 620px; height: 820px; top: 18%; left: -220px;
        background: radial-gradient(circle, rgba(168,85,247,0.16), transparent 70%);
        animation: float 11s ease-in-out infinite reverse;
    }
    .ambient .b3 {
        width: 540px; height: 720px; top: 42%; right: -180px;
        background: radial-gradient(circle, rgba(59,130,246,0.13), transparent 70%);
        animation: float 10s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-26px); }
    }

    /* Top bar — glass hairline */
    .topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 18px; margin-bottom: 22px; border-radius: 14px;
        background: linear-gradient(to bottom, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid var(--border);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06), 0 2px 20px rgba(0,0,0,0.4);
        backdrop-filter: blur(12px);
    }
    .brand { font-weight: 700; color: var(--fg); letter-spacing: -0.01em; }
    .topbar-right {
        color: var(--accent-bright); font-weight: 600; font-size: 12px;
        letter-spacing: 0.06em; font-family: ui-monospace, monospace;
    }

    /* Hero console — two glass panels */
    .console-shell {
        display: grid; grid-template-columns: 0.98fr 1.38fr; gap: 22px;
        align-items: stretch; margin-bottom: 26px;
    }
    .left-panel {
        padding: 32px; border-radius: 20px;
        background: linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid var(--border);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06), 0 8px 40px rgba(0,0,0,0.4);
    }
    .right-panel {
        padding: 32px; border-radius: 20px; color: var(--fg);
        background: linear-gradient(to bottom, rgba(94,106,210,0.10), rgba(255,255,255,0.02));
        border: 1px solid rgba(94,106,210,0.20);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06), 0 8px 40px rgba(0,0,0,0.5), 0 0 80px rgba(94,106,210,0.08);
        position: relative; overflow: hidden;
    }
    .right-panel:after {
        content: ""; position: absolute; width: 320px; height: 320px; border-radius: 50%;
        background: radial-gradient(circle, rgba(94,106,210,0.28), transparent 70%);
        right: -100px; top: -110px; filter: blur(20px);
    }
    .eyebrow {
        color: var(--accent-bright); font-size: 12px; font-weight: 600;
        letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 14px;
        font-family: ui-monospace, monospace;
    }
    .hero-title {
        font-size: 44px; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1;
        margin-bottom: 16px;
        background: linear-gradient(to bottom, #ffffff, rgba(255,255,255,0.7));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-copy { color: var(--fg-muted); font-size: 15px; line-height: 1.8; margin-bottom: 20px; }
    .tag-light {
        display: inline-block; padding: 6px 12px; border-radius: 999px; margin: 0 8px 8px 0;
        background: rgba(94,106,210,0.08); color: #c7ccf2;
        border: 1px solid rgba(94,106,210,0.3); font-size: 12px; font-weight: 500;
    }
    .assistant-title { font-size: 20px; font-weight: 600; color: var(--fg); margin-bottom: 12px; position: relative; z-index: 1; }
    .assistant-line { color: var(--fg-muted); font-size: 14px; line-height: 1.75; margin-bottom: 18px; position: relative; z-index: 1; }
    .flow-item { display: flex; gap: 12px; align-items: flex-start; margin: 15px 0; position: relative; z-index: 1; }
    .flow-num {
        width: 30px; height: 30px; border-radius: 9px;
        background: linear-gradient(to bottom, #6872D9, #5E6AD2); color: #fff;
        display: inline-flex; justify-content: center; align-items: center;
        font-weight: 600; flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(94,106,210,0.4), inset 0 1px 0 0 rgba(255,255,255,0.2);
    }
    .flow-text { color: var(--fg-muted); font-size: 14px; line-height: 1.55; }
    .flow-text b { color: var(--fg); font-weight: 600; }

    /* Section labels & titles */
    .section-label {
        font-size: 12px; color: var(--accent-bright); font-weight: 600;
        letter-spacing: 0.16em; text-transform: uppercase; margin: 10px 0 8px 0;
        font-family: ui-monospace, monospace;
    }
    .section-title {
        font-size: 24px; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 10px;
        background: linear-gradient(to bottom, #ffffff, rgba(255,255,255,0.75));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hint { color: var(--fg-muted); font-size: 14px; line-height: 1.65; margin-bottom: 12px; }

    /* Cards & panels — glass */
    .doc-card {
        background: rgba(94,106,210,0.06); border: 1px solid rgba(94,106,210,0.2);
        border-radius: 14px; padding: 14px 16px; margin: 12px 0;
        color: var(--fg-muted); font-size: 14px;
    }
    .doc-card b { color: var(--fg); }
    .answer-panel {
        background: linear-gradient(to bottom, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid var(--border); border-radius: 16px; padding: 6px 20px 14px;
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.05), 0 8px 40px rgba(0,0,0,0.4);
    }
    .answer-block {
        border-left: 3px solid var(--accent); padding: 12px 16px; margin: 14px 0;
        background: rgba(255,255,255,0.03); border-radius: 0 12px 12px 0;
    }
    .answer-block h4 { margin: 0 0 8px 0; color: var(--fg); font-size: 14px; font-weight: 600; }
    .answer-block p { margin: 0; color: var(--fg-muted); line-height: 1.7; }
    .value-card {
        background: linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid var(--border); border-radius: 16px; padding: 20px; min-height: 150px;
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.05), 0 2px 20px rgba(0,0,0,0.35);
        transition: transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s, border-color 0.25s;
    }
    .value-card:hover {
        transform: translateY(-4px); border-color: var(--border-hover);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.08), 0 12px 50px rgba(0,0,0,0.5), 0 0 60px rgba(94,106,210,0.12);
    }
    .value-card h4 { margin: 0 0 10px 0; color: var(--fg); font-size: 16px; font-weight: 600; }
    .value-card p { margin: 0; color: var(--fg-muted); font-size: 14px; line-height: 1.6; }
    .history-card {
        background: rgba(255,255,255,0.03); border: 1px solid var(--border);
        border-radius: 14px; padding: 14px;
    }

    /* --- Streamlit native widgets --- */
    h1, h2, h3, h4, h5, h6 { color: var(--fg); letter-spacing: -0.01em; }
    hr { border-color: var(--border) !important; }

    .stButton > button, .stDownloadButton > button {
        border-radius: 8px; font-weight: 600; border: 1px solid transparent;
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out, background 0.2s ease-out;
        background: rgba(255,255,255,0.05); color: var(--fg);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.08);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: rgba(255,255,255,0.09); border-color: var(--border-hover);
    }
    .stButton > button:active, .stDownloadButton > button:active { transform: scale(0.98); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(to bottom, #6872D9, #5E6AD2); color: #fff; border: none;
        box-shadow: 0 0 0 1px rgba(94,106,210,0.5), 0 4px 14px rgba(94,106,210,0.35), inset 0 1px 0 0 rgba(255,255,255,0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(to bottom, #7079e0, #6872D9);
        box-shadow: 0 0 0 1px rgba(94,106,210,0.6), 0 6px 22px rgba(94,106,210,0.45), inset 0 1px 0 0 rgba(255,255,255,0.25);
    }
    .stButton > button[kind="primary"][disabled],
    .stButton > button[kind="primary"]:disabled {
        background: rgba(255,255,255,0.05); color: var(--fg-muted);
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.05); opacity: 0.7;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0f0f13; border: 1px dashed rgba(255,255,255,0.12); border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(94,106,210,0.4); }

    .stTextArea textarea, .stTextInput input {
        background: #0f0f13 !important; border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 8px !important; color: var(--fg) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(94,106,210,0.25) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important; border-radius: 12px !important;
        background: rgba(255,255,255,0.03); overflow: hidden;
    }
    [data-testid="stExpander"] summary:hover { color: var(--accent-bright); }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); backdrop-filter: blur(6px); }

    @media (prefers-reduced-motion: reduce) {
        .ambient .blob { animation: none !important; }
    }
    @media (max-width: 900px) {
        .console-shell { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 环境光晕背景层：固定在内容之下，网格 + 三个缓慢漂浮的光斑，营造电影级氛围光
st.markdown(
    """
    <div class="ambient">
        <div class="grid"></div>
        <div class="blob b1"></div>
        <div class="blob b2"></div>
        <div class="blob b3"></div>
    </div>
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
        <div class="topbar-right">Hybrid Retrieval · Grounded Generation · Anti-Hallucination</div>
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
                上传内部资料，系统按结构切分为知识片段，用「BM25 + 向量」混合检索相关内容，
                再交给大模型生成带引用来源的回答；检索不到足够相关的内容时直接说「未找到」。
            </div>
            <span class="tag-light">混合检索</span>
            <span class="tag-light">向量语义</span>
            <span class="tag-light">真实大模型</span>
            <span class="tag-light">引用来源</span>
            <span class="tag-light">防幻觉</span>
        </div>
        <div class="right-panel">
            <div class="assistant-title">工作流不是生成内容，而是先找依据</div>
            <div class="assistant-line">
                这个页面刻意设计成企业知识库控制台，和短视频生产工具区分开：更强调文档、检索、证据和复核。
            </div>
            <div class="flow-item">
                <div class="flow-num">1</div>
                <div class="flow-text"><b>上传资料</b><br>解析 TXT / MD / PDF / DOCX，按标题段落切分为知识片段。</div>
            </div>
            <div class="flow-item">
                <div class="flow-num">2</div>
                <div class="flow-text"><b>混合检索</b><br>BM25 字面 + bge 向量语义，加权召回 Top-K 相关片段。</div>
            </div>
            <div class="flow-item">
                <div class="flow-num">3</div>
                <div class="flow-text"><b>生成答案</b><br>大模型据片段生成结论/依据/下一步；无依据则不作答。</div>
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
        - Python / Streamlit
        - jieba + BM25 检索
        - bge 向量 + 混合召回
        - 结构感知切分
        - Claude 生成（可降级 Mock）
        - 相关度门槛防幻觉
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
        '<div class="hint">支持 TXT / MD / PDF / DOCX，可多选。建议先用 sample_docs 里的文档测试。</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "上传 TXT / MD / PDF / DOCX 文档（可多选）",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="可一次选择多个公司制度、项目说明、培训资料等文档，跨文档统一检索。",
    )

    # 用「文件名+大小」做签名，仅在上传集合变化时重新解析，避免每次交互重复切分
    signature = [(f.name, f.size) for f in uploaded_files]
    if uploaded_files and signature != st.session_state.get("upload_signature"):
        all_chunks = []
        total_chars = 0
        names = []
        for uploaded_file in uploaded_files:
            text = read_uploaded_file(uploaded_file)
            total_chars += len(text)
            names.append(uploaded_file.name)
            for chunk in split_text_into_chunks(text):
                chunk["source"] = uploaded_file.name  # 标记片段来源文档
                all_chunks.append(chunk)

        label = (
            f"{len(names)} 个文档：{', '.join(names)}"
            if len(names) > 1
            else (names[0] if names else "未命名文档")
        )
        st.session_state.chunks = all_chunks
        st.session_state.doc_stats = {
            "filename": label,
            "char_count": str(total_chars),
            "chunk_count": str(len(all_chunks)),
        }
        st.session_state.upload_signature = signature
        st.session_state.answer = None
        st.session_state.retrieved_chunks = []

        st.success(f"已解析 {len(names)} 个文档，共切出 {len(all_chunks)} 个知识片段。")

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
            for chunk in st.session_state.chunks[:5]:
                source = chunk.get("source", "")
                st.markdown(f"**{chunk['chunk_id']}**" + (f" · {source}" if source else ""))
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
            source = chunk.get("source", "")
            title = f"引用片段 {index}"
            if source:
                title += f"｜{source}"
            title += f"｜{chunk.get('chunk_id', '')}｜相关分数 {chunk.get('score', '')}"
            with st.expander(title):
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
        "混合检索",
        "BM25 字面 + bge 向量语义加权召回，换了说法也能找到，支持多文档跨库。",
    ),
    (
        "可信输出",
        "先检索后生成、标注引用来源；相关度不足直接说未找到，从源头防幻觉。",
    ),
    (
        "可量化",
        "内置检索评测集，Top-1 / 来源路由 / 拒答准确率均可一键跑出数字。",
    ),
]

for col, (title, body) in zip(value_cols, value_cards):
    with col:
        render_value_card(title, body)
