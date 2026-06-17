from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json


HISTORY_PATH = Path("data/history.json")


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_qa_record(record: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_history()
    record["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, record)

    HISTORY_PATH.write_text(
        json.dumps(history[:10], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
