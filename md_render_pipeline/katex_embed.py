from __future__ import annotations

import base64
import pathlib
import re
from typing import Dict

CSS_URL_RE = re.compile(r"url\(([^)]+)\)")

MIME_TYPES = {
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".eot": "application/vnd.ms-fontobject",
}


def load_katex_assets(root_dir: pathlib.Path) -> Dict[str, str]:
    root = pathlib.Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"KaTeX assets directory not found: {root}")

    css_path = root / "katex.min.css"
    js_main_path = root / "katex.min.js"
    auto_render_path = root / "contrib" / "auto-render.min.js"

    for path in (css_path, js_main_path):
        if not path.exists():
            raise FileNotFoundError(f"Required KaTeX asset missing: {path}")

    css_text = css_path.read_text(encoding="utf-8")
    css_inlined = _inline_css_urls(css_text, css_path.parent)
    js_main = js_main_path.read_text(encoding="utf-8")
    js_auto = auto_render_path.read_text(encoding="utf-8") if auto_render_path.exists() else ""

    return {
        "css": css_inlined,
        "js_main": js_main,
        "js_auto": js_auto,
    }


def _inline_css_urls(css_text: str, base_dir: pathlib.Path) -> str:
    def repl(match: re.Match) -> str:
        raw_url = match.group(1).strip().strip("'\"")
        if raw_url.startswith("data:"):
            return match.group(0)
        asset_path = (base_dir / raw_url).resolve()
        if not asset_path.exists():
            # Leave as-is to avoid breaking CSS; caller may not need this font.
            return match.group(0)
        data = asset_path.read_bytes()
        mime = MIME_TYPES.get(asset_path.suffix.lower(), "application/octet-stream")
        encoded = base64.b64encode(data).decode("ascii")
        return f"url(data:{mime};base64,{encoded})"

    return CSS_URL_RE.sub(repl, css_text)
