import io
import re
from typing import List, Dict, Tuple


# Markdown 标题行：1~6 个 # 加空格加标题文字
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _read_text_bytes(raw_bytes: bytes) -> str:
    """纯文本/Markdown：优先 UTF-8，失败退回 GBK。"""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("gbk", errors="ignore")


def _read_pdf(raw_bytes: bytes) -> str:
    """PDF：逐页抽取文本，页间用空行分隔（便于按段落切分）。"""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _read_docx(raw_bytes: bytes) -> str:
    """DOCX：抽取段落文本，并把 Word 标题样式转成 Markdown 标题，以复用结构切分。"""
    import docx

    document = docx.Document(io.BytesIO(raw_bytes))
    parts = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "") if para.style else ""
        if style_name.startswith("Heading"):
            digits = "".join(ch for ch in style_name if ch.isdigit())
            level = min(max(int(digits) if digits else 1, 1), 6)
            parts.append("#" * level + " " + text)
        else:
            parts.append(text)
    return "\n\n".join(parts)


def read_uploaded_file(uploaded_file) -> str:
    """
    按扩展名读取上传文件为纯文本：支持 TXT / MD / PDF / DOCX。
    """
    if uploaded_file is None:
        return ""

    raw_bytes = uploaded_file.getvalue()
    name = (getattr(uploaded_file, "name", "") or "").lower()

    if name.endswith(".pdf"):
        return _read_pdf(raw_bytes)
    if name.endswith(".docx"):
        return _read_docx(raw_bytes)
    return _read_text_bytes(raw_bytes)


def normalize_text(text: str) -> str:
    """
    统一换行、去掉行尾空白，并把 3 个以上连续空行压成一个空行。
    与旧版不同：保留单个空行，以便按段落切分（旧版会删掉所有空行）。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    按 Markdown 标题把文档分节，返回 [(面包屑, 正文)]。
    面包屑是标题层级路径，如“公司规范 > 使用范围”；标题前的内容归到空面包屑。
    """
    sections: List[Tuple[str, str]] = []
    heading_stack: List[Tuple[int, str]] = []
    body_lines: List[str] = []
    breadcrumb = ""

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((breadcrumb, body))

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            body_lines.clear()
            level = len(match.group(1))
            title = match.group(2).strip()
            # 维护标题栈：遇到某级标题时，丢掉同级及更深的旧标题，再压入当前标题
            heading_stack[:] = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, title))
            breadcrumb = " > ".join(t for _, t in heading_stack)
        else:
            body_lines.append(line)

    flush()
    return sections


def _char_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """对超长段落的兜底：按字符切，带 overlap 重叠。"""
    pieces = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _pack_paragraphs(body: str, chunk_size: int, overlap: int) -> List[str]:
    """
    把段落贪心打包成不超过 chunk_size 的片段，尽量不拆散段落。
    单个段落本身超长时，才对它按字符切。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_char_split(para, chunk_size, overlap))
            continue

        if current and len(current) + 1 + len(para) > chunk_size:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para

    if current:
        pieces.append(current)
    return pieces


def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 60) -> List[Dict[str, str]]:
    """
    结构感知切分：先按 Markdown 标题分节，再按段落打包成片段。
    相比旧版“固定字数硬切”，不会把句子/段落拦腰截断，且每个片段带所在小节标题，
    检索能命中标题关键词，引用来源也更清晰。
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Dict[str, str]] = []
    index = 1
    for breadcrumb, body in _split_into_sections(text):
        for piece in _pack_paragraphs(body, chunk_size, overlap):
            # 面包屑并入片段文本：既利于检索命中标题，也让引用更好读
            chunk_text = f"【{breadcrumb}】\n{piece}" if breadcrumb else piece
            chunks.append(
                {
                    "chunk_id": f"片段 {index}",
                    "heading": breadcrumb,
                    "text": chunk_text,
                }
            )
            index += 1

    return chunks


def get_document_stats(filename: str, text: str, chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Return basic document statistics.
    """
    return {
        "filename": filename or "未命名文档",
        "char_count": str(len(text)),
        "chunk_count": str(len(chunks)),
    }
