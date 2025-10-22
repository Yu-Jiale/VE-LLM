# html_template.py
from jinja2 import Template

HTML_TMPL = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' data:;">

<style>
  @page { size: A4; margin: 0; }
  html, body {
    margin: 0;
    padding: 0;
    background: #f7f9fc;
    font-family: "{{ font_family }}", "Noto Sans", sans-serif;
  }
  .page {
    box-sizing: border-box;
    width: {{ page_w }}px;
    max-width: 100%;
    min-height: {{ page_h }}px;
    padding: {{ m_top }}px {{ m_right }}px {{ m_bottom }}px {{ m_left }}px;
    font-size: {{ font_size }}px;
    line-height: {{ line_height }};
    color: #111;
    text-align: {{ align }};
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
    white-space: normal;
    background: {{ bg }};
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
    border-radius: 24px;
  }
  .page, .page * {
    box-sizing: border-box;
    max-width: 100%;
    hyphens: {{ hyphens }};
    word-break: break-word;
    overflow-wrap: anywhere;
  }
  h1, h2, h3, h4, h5 {
    margin: 0.8em 0 0.4em 0;
    font-weight: 600;
    color: #1f2933;
  }
  h1 { font-size: {{ (font_size * 1.6)|round(2) }}px; border-bottom: 2px solid #dce3f0; padding-bottom: 0.3em; }
  h2 { font-size: {{ (font_size * 1.35)|round(2) }}px; }
  h3 { font-size: {{ (font_size * 1.2)|round(2) }}px; }
  p, li { margin: 0 0 0.7em 0; color: #2e3944; }
  ul, ol { padding-left: 1.6em; }

  code, pre {
    font-family: "{{ code_font }}", "JetBrains Mono", monospace;
    font-size: {{ (font_size * 0.85)|round(2) }}px;
  }
  code {
    background: rgba(15, 23, 42, 0.12);
    padding: 0.2em 0.4em;
    border-radius: 6px;
  }
  pre {
    white-space: pre-wrap;
    background: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
    box-shadow: none;
    overflow: visible;
  }
  pre code {
    background: transparent;
    padding: 0;
    font-size: inherit;
  }

  blockquote {
    margin: 1em 0;
    padding: 0.6em 1em;
    border-left: 4px solid #4f46e5;
    background: rgba(99, 102, 241, 0.08);
    color: #3730a3;
    border-radius: 8px;
  }

  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
  }
  th, td {
    padding: 10px 14px;
    text-align: left;
  }
  th {
    background: rgba(37, 99, 235, 0.12);
    color: #1d4ed8;
    font-weight: 600;
  }
  tr:nth-child(even) td {
    background: rgba(15, 23, 42, 0.025);
  }

  img {
    max-width: 100%;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
  }

  /* 2. 强化任务列表样式，确保移除圆点 */
  ul.task-list,
  ol.task-list {
    list-style: none !important;
    padding-left: 0;
    margin-left: 0;
  }
  ul.task-list > li,
  ol.task-list > li,
  ul li.task-list-item,
  ol li.task-list-item {
    list-style: none !important;
    margin: 0.35em 0;
  }
  ul.task-list > li {
    display: flex;
    align-items: flex-start;
    gap: 0.6em;
    position: relative;
  }
  ul.task-list > li > ul,
  ul.task-list > li > ol,
  ol.task-list > li > ul,
  ol.task-list > li > ol {
    margin: 0.35em 0 0.35em 1.6em;
    padding-left: 1.2em;
  }
  ul.task-list > li > ul li,
  ul.task-list > li > ol li,
  ol.task-list > li > ul li,
  ol.task-list > li > ol li,
  ul li.task-list-item > ul li,
  ul li.task-list-item > ol li {
    list-style: none !important;
    display: flex;
    align-items: flex-start;
    gap: 0.4em;
  }
  ul.task-list > li > ul li::marker,
  ul.task-list > li > ol li::marker,
  ol.task-list > li > ul li::marker,
  ol.task-list > li > ol li::marker {
    content: "";
  }
  ul.task-list > li::marker,
  ol.task-list > li::marker,
  ul li.task-list-item::marker,
  ol li.task-list-item::marker {
    content: "";
    display: none;
  }
  ul.task-list li::before,
  ol.task-list li::before,
  ul li.task-list-item::before,
  ol li.task-list-item::before {
    content: none !important;
  }
  ul.task-list li input[type="checkbox"],
  ol.task-list li input[type="checkbox"],
  ul li.task-list-item input[type="checkbox"],
  ol li.task-list-item input[type="checkbox"] {
    flex-shrink: 0;
    width: 1.1em;
    height: 1.1em;
    margin-top: 0.2em;
    cursor: pointer;
    accent-color: #2563eb;
  }
  ul.task-list li label,
  ol.task-list li label,
  ul li.task-list-item label,
  ol li.task-list-item label {
    display: inline-flex;
    align-items: flex-start;
    gap: 0.6em;
    margin: 0;
  }
  ul.task-list li > p,
  ol.task-list li > p,
  ul li.task-list-item > p,
  ol li.task-list-item > p {
    margin: 0;
  }

  .codehilite {
    background: linear-gradient(135deg, #101c2d, #12263f);
    color: #d7e3ff;
    padding: 1.1em 1.2em;
    border-radius: 16px;
    border: 1px solid rgba(36, 85, 136, 0.4);
    box-shadow: 0 14px 34px rgba(10, 24, 44, 0.45);
    overflow: auto;
  }
  .codehilite code,
  .codehilite pre {
    color: inherit;
  }
  .codehilite .highlight {
    background: transparent;
  }

  .math-block {
    margin: 1.2em 0;
    text-align: center;
  }
  .math-inline {
    display: inline-block;
    vertical-align: middle;
  }
  .math-katex[data-katex-display="true"] {
    display: block;
    text-align: center;
  }
  .math-katex[data-katex-display="false"] {
    display: inline-block;
    vertical-align: middle;
  }
  .math-unicode {
    font-family: "STIX Two Math", "Cambria Math", "Times New Roman", serif;
    font-size: {{ (font_size * 1.05)|round(2) }}px;
    color: #1f2933;
    line-height: 1.4;
  }
  .math-unicode sub {
    font-size: 0.7em;
    vertical-align: baseline;
    position: relative;
    bottom: -0.3em;
  }
  .math-unicode sup {
    font-size: 0.7em;
    vertical-align: baseline;
    position: relative;
    top: -0.4em;
  }

  .footnotes {
    margin-top: 2em;
    font-size: {{ (font_size * 0.85)|round(2) }}px;
    color: #475569;
  }
  .footnotes hr {
    border: none;
    border-top: 1px solid rgba(15, 23, 42, 0.12);
    margin-bottom: 0.8em;
  }
  .footnotes ol {
    padding-left: 1.2em;
  }
  .footnotes li {
    margin-bottom: 0.4em;
  }
  sup.footnote-ref {
    font-size: 0.75em;
    vertical-align: super;
    margin-left: 0.1em;
  }
  sup.footnote-ref a {
    color: #1d4ed8;
    text-decoration: none;
  }
  .footnotes a {
    color: #1d4ed8;
    text-decoration: none;
  }
</style>
{% if pygments_css %}
<style>
  {{ pygments_css|safe }}
</style>
{% endif %}
{% if extra_css %}
<style>
  {{ extra_css|safe }}
</style>
{% endif %}
</head>
<body>
<div class="page">
  {{ html|safe }}
</div>

{% if extra_scripts %}
{{ extra_scripts|safe }}
{% endif %}

</body>
</html>
""")
