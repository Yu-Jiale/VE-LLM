import os, tempfile, hashlib, pathlib, subprocess
from typing import List, Tuple, Optional
from markdown2 import markdown
from .chromium import RenderSession
from .config import RenderSettings
from .html_template import HTML_TMPL

def md_to_html(md_text: str) -> str:
    return markdown(md_text, extras=["fenced-code-blocks", "tables"])

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
        page.set_content(html, wait_until="networkidle")
        try:
            page.wait_for_function(
                "document.fonts && document.fonts.status === 'loaded'",
                timeout=5000,
            )
        except Exception:
            pass
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

def pdf_bytes_to_pngs(pdf_bytes: bytes, out_dir: str, dpi: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        cmd = ["pdftocairo", "-png", "-r", str(dpi), pdf_path, os.path.join(out_dir, "page")]
        subprocess.check_call(cmd)
    paths = sorted(str(p) for p in pathlib.Path(out_dir).glob("page-*.png"))
    if paths:
        for i, p in enumerate(paths, 1):
            newp = os.path.join(out_dir, f"page_{i:04d}.png")
            os.replace(p, newp)
        paths = [os.path.join(out_dir, f"page_{i:04d}.png") for i in range(1, len(paths)+1)]
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
    html_body = md_to_html(md_text)
    html_full = HTML_TMPL.render(
        html=html_body,
        page_w=cfg.page_size_px[0], page_h=cfg.page_size_px[1],
        m_top=cfg.margins_px[0], m_right=cfg.margins_px[1],
        m_bottom=cfg.margins_px[2], m_left=cfg.margins_px[3],
        font_family=cfg.font_family, code_font=cfg.code_font,
        font_size=cfg.font_size_px, line_height=cfg.line_height,
        align=cfg.alignment, hyphens=cfg.hyphens, bg=cfg.background_color
    )
    pdf_bytes = render_html_to_pdf_bytes(html_full, cfg, session=session)
    pngs = pdf_bytes_to_pngs(pdf_bytes, cfg.out_dir, cfg.dpi)
    return pngs, fingerprint_cfg(cfg)
