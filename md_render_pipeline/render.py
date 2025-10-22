# render.py
import hashlib
import html
import os
import pathlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import markdown2
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from .chromium import RenderSession
from .config import RenderSettings
from .html_template import HTML_TMPL
from .katex_embed import load_katex_assets

MARKDOWN_EXTRAS = [
    "fenced-code-blocks",
    "tables",
    "task_list",
    "strike",
    "footnotes",
]

# 解析 Markdown 前先把公式替换成占位符，避免 markdown2 将下划线解析成 <em>
_markdown = markdown2.Markdown(extras=MARKDOWN_EXTRAS)
# 1. 代码高亮主题：Stata 深色（蓝黑调）
_PYGMENTS_STYLE_NAME = "stata-dark"
_PYGMENTS_INLINE_FORMATTER = HtmlFormatter(style=_PYGMENTS_STYLE_NAME, nowrap=True)
_PYGMENTS_STYLE = HtmlFormatter(style=_PYGMENTS_STYLE_NAME).get_style_defs('.codehilite')
CODE_BLOCK_RE = re.compile(r"<pre><code(?: class=\"language-([\w#+\-\.]+)\")?>(.*?)</code></pre>", re.DOTALL)
MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)")

LATEX_UNICODE_REPLACEMENTS = {
    "\\mathcal{L}": "𝓛",
    "\\mathcal{l}": "𝓁",
    "\\sum": "∑",
    "\\int": "∫",
    "\\log": "log",
    "\\theta": "θ",
    "\\Theta": "Θ",
    "\\nabla": "∇",
    "\\partial": "∂",
    "\\cdot": "·",
    "\\times": "×",
    "\\leq": "≤",
    "\\geq": "≥",
    "\\rightarrow": "→",
    "\\to": "→",
    "\\left": "",
    "\\right": "",
    "\\,": " ",
}


def md_to_html(md_text: str) -> Tuple[str, str]:
    sanitized_text, math_placeholders = _extract_math_placeholders(md_text)
    html_out = _markdown.convert(sanitized_text)
    html_out = _restore_math_placeholders(html_out, math_placeholders)
    html_out = _normalize_task_lists(html_out)
    html_out, pygments_css = _apply_code_highlight(html_out)
    return html_out, pygments_css


def _apply_code_highlight(html_text: str) -> Tuple[str, str]:
    def repl(match: re.Match) -> str:
        lang = match.group(1)
        code = html.unescape(match.group(2))
        lexer = _get_lexer(lang, code)
        highlighted = highlight(code, lexer, _PYGMENTS_INLINE_FORMATTER)
        lang_class = f" language-{lang}" if lang else ""
        return f'<pre class="codehilite{lang_class}"><code>{highlighted}</code></pre>'

    highlighted_html = CODE_BLOCK_RE.sub(repl, html_text)
    return highlighted_html, _PYGMENTS_STYLE


def _get_lexer(lang: Optional[str], code: str):
    if lang:
        try:
            return get_lexer_by_name(lang, stripall=False)
        except ClassNotFound:
            pass
    try:
        return guess_lexer(code)
    except ClassNotFound:
        return TextLexer(stripall=False)


def _render_math(html_text: str, mode: str) -> str:
    if mode == "katex":
        # 保留原始 $...$ / $$...$$，后续由 KaTeX auto-render 处理
        return html_text

    def block_repl_unicode(match: re.Match) -> str:
        latex_src = html.unescape(match.group(1).strip())
        rendered = _latex_to_unicode(latex_src)
        return f'<div class="math-block math-unicode">{rendered}</div>'

    def inline_repl_unicode(match: re.Match) -> str:
        latex_src = html.unescape(match.group(1).strip())
        rendered = _latex_to_unicode(latex_src)
        return f'<span class="math-inline math-unicode">{rendered}</span>'

    html_with_blocks = MATH_BLOCK_RE.sub(block_repl_unicode, html_text)
    html_with_inline = MATH_INLINE_RE.sub(inline_repl_unicode, html_with_blocks)
    return html_with_inline


def _latex_to_unicode(expr: str) -> str:
    out = html.escape(expr, quote=False)
    for src, tgt in LATEX_UNICODE_REPLACEMENTS.items():
        out = out.replace(src, tgt)

    def _strip_braces(text: str) -> str:
        if text.startswith("{") and text.endswith("}"):
            return text[1:-1]
        return text

    def _replace_sub(match: re.Match) -> str:
        content = _strip_braces(match.group(1))
        return f"<sub>{content}</sub>"

    def _replace_sup(match: re.Match) -> str:
        content = _strip_braces(match.group(1))
        return f"<sup>{content}</sup>"

    out = re.sub(r"_({[^}]+}|.)", _replace_sub, out)
    out = re.sub(r"\^({[^}]+}|.)", _replace_sup, out)
    # 清除残余的大括号
    out = out.replace("{", "").replace("}", "")
    return out


BLOCK_PLACEHOLDER_FMT = "\uE000MB{idx}\uE001"
INLINE_PLACEHOLDER_FMT = "\uE000MI{idx}\uE001"


def _extract_math_placeholders(md_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    placeholders: List[Tuple[str, str]] = []

    def block_repl(match: re.Match) -> str:
        placeholder = BLOCK_PLACEHOLDER_FMT.format(idx=len(placeholders))
        placeholders.append((placeholder, match.group(0)))
        return placeholder

    text = MATH_BLOCK_RE.sub(block_repl, md_text)

    def inline_repl(match: re.Match) -> str:
        placeholder = INLINE_PLACEHOLDER_FMT.format(idx=len(placeholders))
        placeholders.append((placeholder, match.group(0)))
        return placeholder

    text = MATH_INLINE_RE.sub(inline_repl, text)
    return text, placeholders


def _restore_math_placeholders(html_text: str, placeholders: List[Tuple[str, str]]) -> str:
    for placeholder, expr in placeholders:
        html_text = html_text.replace(placeholder, html.escape(expr, quote=False))
    return html_text


_TASK_LI_BULLET_RE = re.compile(
    r'(<li\b[^>]*class=(?:"[^"]*task-list-item[^"]*"|\'[^\']*task-list-item[^\']*\')[^>]*>\s*)(?:&bull;|&#8226;|&#x2022;|•)\s*',
    re.IGNORECASE,
)

_LI_CHECKBOX_RE = re.compile(
    r'<li([^>]*)>(\s*<input\s+type="checkbox"[^>]*>)',
    re.IGNORECASE,
)


_TASK_LI_BULLET_RE = re.compile(
    r'(<li\b[^>]*class=(?:"[^"]*task-list-item[^"]*"|\'[^\']*task-list-item[^\']*\')[^>]*>\s*)(?:&bull;|&#8226;|&#x2022;|•)\s*',
    re.IGNORECASE,
)


def _normalize_task_lists(html_text: str) -> str:
    def _ensure_task_class(match: re.Match) -> str:
        attrs, input_html = match.groups()
        attrs_new = attrs
        if "task-list-item" not in attrs_new:
            if 'class="' in attrs_new:
                attrs_new = re.sub(
                    r'class="([^"]*)"',
                    lambda m: f'class="task-list-item {m.group(1)}"',
                    attrs_new,
                    count=1,
                )
            elif "class='" in attrs_new:
                attrs_new = re.sub(
                    r"class='([^']*)'",
                    lambda m: f"class='task-list-item {m.group(1)}'",
                    attrs_new,
                    count=1,
                )
            else:
                attrs_new += ' class="task-list-item"'
        return f"<li{attrs_new}>{input_html}"

    html_text = _LI_CHECKBOX_RE.sub(_ensure_task_class, html_text)

    cleaned, _ = _TASK_LI_BULLET_RE.subn(r"\1", html_text)
    return cleaned

def render_html_to_pdf_bytes(
    html: str,
    cfg: RenderSettings,
    *,
    session: Optional[RenderSession] = None,
) -> bytes:
    if session is not None:
        page = session.new_page()
        owns_session = False
    else:
        session = RenderSession()
        session.start()
        page = session.new_page()
        owns_session = True

    try:
        page.set_viewport_size({"width": cfg.page_size_px[0], "height": cfg.page_size_px[1]})
        page.emulate_media(media="print")
        # 必须使用 wait_until="networkidle" 确保网络请求有机会完成
        page.set_content(html, wait_until="networkidle")
        if cfg.math_renderer == "katex":
            try:
                page.wait_for_function(
                    "document.body.classList.contains('katex-rendered')",
                    timeout=10000,
                )
            except Exception as e:
                print(f"Warning: Timed out waiting for KaTeX to render: {e}")
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            prefer_css_page_size=cfg.prefer_css_page_size,
            scale=cfg.chromium_scale,
        )
    finally:
        page.close()
        if owns_session:
            session.close()
    return pdf_bytes

def pdf_bytes_to_pngs(
    pdf_bytes: bytes,
    out_dir: str,
    dpi: int,
    *,
    single_file: bool = False,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        cmd = ["pdftocairo", "-png", "-r", str(dpi)]
        if single_file:
            cmd.append("-singlefile")
        cmd.extend([pdf_path, os.path.join(out_dir, "page")])
        subprocess.check_call(cmd)
    if single_file:
        raw = sorted(str(p) for p in pathlib.Path(out_dir).glob("page.png"))
    else:
        raw = sorted(str(p) for p in pathlib.Path(out_dir).glob("page-*.png"))

    paths = raw
    if paths:
        for i, p in enumerate(paths, 1):
            newp = os.path.join(out_dir, f"page_{i:04d}.png")
            os.replace(p, newp)
        paths = [os.path.join(out_dir, f"page_{i:04d}.png") for i in range(1, len(paths) + 1)]
    return paths

def fingerprint_cfg(cfg: RenderSettings) -> str:
    payload = (
        f"{cfg.dpi}|{cfg.page_size_px}|{cfg.margins_px}|{cfg.font_family}|"
        f"{cfg.code_font}|{cfg.font_size_px}|{cfg.line_height}|{cfg.alignment}|"
        f"{cfg.hyphens}|{cfg.ligatures}|{cfg.chromium_scale}"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()

def render_markdown(
    md_text: str,
    cfg: RenderSettings,
    *,
    session: Optional[RenderSession] = None,
) -> Tuple[List[str], str]:
    html_full = compose_html_doc(md_text, cfg)
    pdf_bytes = render_html_to_pdf_bytes(html_full, cfg, session=session)
    pngs = pdf_bytes_to_pngs(
        pdf_bytes,
        cfg.out_dir,
        cfg.dpi,
        single_file=cfg.pdf_single_file,
    )
    return pngs, fingerprint_cfg(cfg)


def compose_html_doc(md_text: str, cfg: RenderSettings) -> str:
    html_body, pygments_css = md_to_html(md_text)

    extras_css: List[str] = []
    extras_scripts: List[str] = []

    math_mode = cfg.math_renderer.lower()
    html_body = _render_math(html_body, math_mode)
    if math_mode == "katex":
        if not cfg.katex_assets_dir:
            raise ValueError("RenderSettings.katex_assets_dir must be set when math_renderer='katex'")
        assets = load_katex_assets(Path(cfg.katex_assets_dir))
        extras_css.append(assets["css"])
        extras_scripts.append(f"<script>{assets['js_main']}</script>")
        js_auto = assets.get("js_auto")
        if not js_auto:
            raise ValueError("KaTeX auto-render script not found in katex_assets_dir/contrib/auto-render.min.js")
        extras_scripts.append(f"<script>{js_auto}</script>")
        extras_scripts.append(
            "<script>document.addEventListener('DOMContentLoaded',function(){renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},{left:'$',right:'$',display:false},{left:'\\\\(',right:'\\\\)',display:false}],throwOnError:false});document.body.classList.add('katex-rendered');});</script>"
        )
    elif math_mode != "unicode":
        raise ValueError(f"Unsupported math_renderer: {cfg.math_renderer}")

    html_full = HTML_TMPL.render(
        html=html_body,
        page_w=cfg.page_size_px[0],
        page_h=cfg.page_size_px[1],
        m_top=cfg.margins_px[0],
        m_right=cfg.margins_px[1],
        m_bottom=cfg.margins_px[2],
        m_left=cfg.margins_px[3],
        font_family=cfg.font_family,
        code_font=cfg.code_font,
        font_size=cfg.font_size_px,
        line_height=cfg.line_height,
        align=cfg.alignment,
        hyphens=cfg.hyphens,
        bg=cfg.background_color,
        pygments_css=pygments_css,
        extra_css="\n".join(extras_css),
        extra_scripts="\n".join(extras_scripts),
    )
    return html_full


# Utility used when math_renderer='unicode'
