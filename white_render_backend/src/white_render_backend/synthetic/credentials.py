"""Load a backend-only credential without logging document contents."""
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from ..domain import EngineError


def load_api_key():
    filename = os.getenv("MINIMAX_API_KEY_FILE")
    config = Path(__file__).resolve().parents[3] / "minimax.local.json"
    try:
        if not filename and config.is_file():
            filename = str(config.parent / json.loads(config.read_text(encoding="utf-8-sig"))["credential_file"])
        if not filename:
            return os.getenv("MINIMAX_API_KEY", "")
        with zipfile.ZipFile(filename) as doc:
            xml = ET.fromstring(doc.read("word/document.xml"))
        text = "".join(node.text or "" for node in xml.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        keys = set(re.findall(r"sk-[A-Za-z0-9_-]{20,}", text))
        if len(keys) != 1:
            raise ValueError("Expected one key")
        return keys.pop()
    except Exception:
        raise EngineError("H3_CREDENTIAL_FILE", "Cannot load exactly one key from the configured backend credential document", status=503) from None
