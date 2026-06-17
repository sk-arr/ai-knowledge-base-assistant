from typing import List, Dict


def read_uploaded_file(uploaded_file) -> str:
    """
    Read TXT or MD uploaded file content as UTF-8 text.
    """
    if uploaded_file is None:
        return ""

    raw_bytes = uploaded_file.getvalue()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("gbk", errors="ignore")


def normalize_text(text: str) -> str:
    """
    Clean text while keeping paragraph structure.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_text_into_chunks(text: str, chunk_size: int = 450, overlap: int = 80) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks.
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 1
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"片段 {chunk_index}",
                    "text": chunk_text,
                    "start": str(start),
                    "end": str(end),
                }
            )
            chunk_index += 1

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

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
