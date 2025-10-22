import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

PAGE_SIZE_MM: Dict[str, Tuple[int, int]] = {
    "A4": (210, 297),
    "LETTER": (216, 279),
    "LEGAL": (216, 356),
}

DEFAULT_DPI = 192
DEFAULT_PAGE_SIZE_PX = (1728, 2432)
DEFAULT_MARGINS_PX = (96, 96, 96, 96)
DEFAULT_FONT_SIZE_PX = 28


def _mm_to_px(mm: int, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))


def _resolve_page_size_px(data: Dict[str, Any], dpi: int) -> Tuple[int, int]:
    if "page_size_px" in data:
        width, height = data["page_size_px"]
        return int(width), int(height)
    if "page_size_mm" in data:
        width_mm, height_mm = data["page_size_mm"]
        return _mm_to_px(width_mm, dpi), _mm_to_px(height_mm, dpi)
    if "page_size" in data:
        key = str(data["page_size"]).upper()
        if key not in PAGE_SIZE_MM:
            raise ValueError(f"Unsupported page_size '{data['page_size']}'. Supported: {', '.join(PAGE_SIZE_MM)}")
        width_mm, height_mm = PAGE_SIZE_MM[key]
        return _mm_to_px(width_mm, dpi), _mm_to_px(height_mm, dpi)
    scale = dpi / DEFAULT_DPI
    return int(DEFAULT_PAGE_SIZE_PX[0] * scale), int(DEFAULT_PAGE_SIZE_PX[1] * scale)


def _resolve_margins_px(data: Dict[str, Any], dpi: int) -> Tuple[int, int, int, int]:
    if "margins_px" in data:
        top, right, bottom, left = data["margins_px"]
        return int(top), int(right), int(bottom), int(left)
    if "margins_mm" in data:
        top_mm, right_mm, bottom_mm, left_mm = data["margins_mm"]
        return (
            _mm_to_px(top_mm, dpi),
            _mm_to_px(right_mm, dpi),
            _mm_to_px(bottom_mm, dpi),
            _mm_to_px(left_mm, dpi),
        )
    if "margins" in data:
        margins = data["margins"]
        if isinstance(margins, dict):
            return (
                int(margins.get("top", DEFAULT_MARGINS_PX[0])),
                int(margins.get("right", DEFAULT_MARGINS_PX[1])),
                int(margins.get("bottom", DEFAULT_MARGINS_PX[2])),
                int(margins.get("left", DEFAULT_MARGINS_PX[3])),
            )
        if isinstance(margins, (list, tuple)) and len(margins) == 4:
            top, right, bottom, left = margins
            return int(top), int(right), int(bottom), int(left)
    scale = dpi / DEFAULT_DPI
    return (
        int(DEFAULT_MARGINS_PX[0] * scale),
        int(DEFAULT_MARGINS_PX[1] * scale),
        int(DEFAULT_MARGINS_PX[2] * scale),
        int(DEFAULT_MARGINS_PX[3] * scale),
    )


def _resolve_font_size_px(data: Dict[str, Any], dpi: int) -> int:
    if "font_size_px" in data:
        return int(data["font_size_px"])
    if "font_size_pt" in data:
        return int(round((data["font_size_pt"] / 72.0) * dpi))
    if "font_size" in data:
        return int(round((data["font_size"] / 72.0) * dpi))
    return DEFAULT_FONT_SIZE_PX

@dataclass
class RenderSettings:
    # Page & rasterization
    dpi: int = 192
    page_size_px: Tuple[int, int] = (1728, 2432)
    margins_px: Tuple[int, int, int, int] = (96, 96, 96, 96)  # top, right, bottom, left
    background_color: str = "#FFFFFF"

    # Typography
    font_family: str = "Noto Sans CJK SC"
    code_font: str = "JetBrains Mono"
    font_size_px: int = 28
    line_height: float = 1.55
    alignment: str = "left"  # 'left' | 'justify' | 'center'
    hyphens: str = "none"
    ligatures: str = "none"

    # Chromium / PDF
    chromium_scale: float = 1.0
    a4_mm: Tuple[int, int] = (210, 297)  # For completeness; Playwright uses built-ins
    prefer_css_page_size: bool = True
    pdf_single_file: bool = False
    math_renderer: str = "unicode"  # "unicode" | "katex"
    katex_assets_dir: Optional[str] = None

    # Output
    out_dir: str = "./render_out"
    doc_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderSettings":
        dpi = int(data.get("dpi", cls.dpi))
        page_size_px = _resolve_page_size_px(data, dpi)
        margins_px = _resolve_margins_px(data, dpi)
        font_size_px = _resolve_font_size_px(data, dpi)

        return cls(
            dpi=dpi,
            page_size_px=page_size_px,
            margins_px=margins_px,
            background_color=data.get("background_color", cls.background_color),
            font_family=data.get("font_family", cls.font_family),
            code_font=data.get("code_font", cls.code_font),
            font_size_px=font_size_px,
            line_height=float(data.get("line_height", cls.line_height)),
            alignment=data.get("alignment", cls.alignment),
            hyphens=data.get("hyphens", cls.hyphens),
            ligatures=data.get("ligatures", cls.ligatures),
            chromium_scale=float(data.get("chromium_scale", cls.chromium_scale)),
            prefer_css_page_size=bool(data.get("prefer_css_page_size", cls.prefer_css_page_size)),
            pdf_single_file=bool(data.get("pdf_single_file", cls.pdf_single_file)),
            math_renderer=str(data.get("math_renderer", cls.math_renderer)).lower(),
            katex_assets_dir=data.get("katex_assets_dir"),
            out_dir=data.get("out_dir", cls.out_dir),
            doc_id=data.get("doc_id"),
        )


def load_render_settings(path: str) -> RenderSettings:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Render config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RenderSettings.from_dict(data)


def apply_overrides(settings: RenderSettings, **overrides: Any) -> RenderSettings:
    return replace(settings, **overrides)
