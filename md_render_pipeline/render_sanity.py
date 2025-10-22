import argparse
import os
from pathlib import Path
from typing import Optional

from .config import RenderSettings, apply_overrides, load_render_settings
from .render import render_markdown, md_to_html
from .html_template import HTML_TMPL


DEFAULT_SAMPLE = """# 渲染断行测试

这段文字专门用于验证 Markdown → HTML → PNG 渲染是否存在超出右边界的问题。我们需要覆盖以下情况：

1. **长段落**：含有大量连续的中文字符，例如“这是一个非常长非常长非常长的句子，我们希望它能够在页面宽度内自然折行而不会溢出”。请重复这个句子以观察折行。
2. **带有空格的长句**：ThisIsAnExtremelyLongEnglishWordDesignedToTestWhetherTheRendererWillWrapItProperlyWithoutSpaces。
3. **列表项**： 
   - 这是一个测试项，同样包含一段很长的中文，用来观察列表环境中的换行行为，确保不会出现右侧溢出。
   - Another bullet with a very long English sentence that should wrap gracefully even if it exceeds the typical page width limitations.
4. **代码块**：

```python
def long_line():
    return "这是一个非常非常非常非常非常非常非常长的字符串，用于测试代码块内部的断行。"
```

请肉眼检查生成的 PNG 是否存在内容被截断或跑出页面之外的情况。
"""


def build_cfg(out_dir: str, config_path: Optional[str]) -> RenderSettings:
    if config_path:
        base_cfg = load_render_settings(config_path)
    else:
        base_cfg = RenderSettings()
    return apply_overrides(base_cfg, out_dir=out_dir)


def save_html(md_text: str, cfg: RenderSettings, html_path: Path) -> None:
    html_body = md_to_html(md_text)
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
    )
    html_path.write_text(html_full, encoding="utf-8")


def main(args: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="渲染管线断行 smoke 测试")
    parser.add_argument(
        "--markdown",
        type=str,
        help="待渲染的 Markdown 文件路径；若省略则使用内置样例。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./md_render_pipeline/render_out",
        help="PNG 输出目录（默认: ./md_render_pipeline/render_out）",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="同时保存渲染用的完整 HTML，便于浏览器直接查看。",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="渲染配置 JSON 文件路径。",
    )
    ns = parser.parse_args(args=args)

    if ns.markdown:
        md_path = Path(ns.markdown)
        if not md_path.exists():
            raise FileNotFoundError(f"markdown 文件不存在: {md_path}")
        md_text = md_path.read_text(encoding="utf-8")
    else:
        md_text = DEFAULT_SAMPLE

    cfg = build_cfg(ns.out_dir, ns.config)
    os.makedirs(cfg.out_dir, exist_ok=True)

    pngs, cfg_fp = render_markdown(md_text, cfg)
    print(f"[render] 输出页面数量: {len(pngs)}  配置指纹: {cfg_fp}")
    for path in pngs:
        print(f" - {path}")

    if ns.save_html:
        html_path = Path(cfg.out_dir) / "render_debug.html"
        save_html(md_text, cfg, html_path)
        print(f"[render] 已保存 HTML 到: {html_path}")


if __name__ == "__main__":
    main()
