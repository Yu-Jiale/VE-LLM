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
    background: rgba(15, 23, 42, 0.08);
    padding: 0.2em 0.4em;
    border-radius: 6px;
  }
  pre {
    white-space: pre-wrap;
    background: #1f2937;
    color: #f8fafc;
    padding: 1em;
    border-radius: 12px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
    overflow: auto;
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
</style>
</head>
<body>
<div class="page">
  {{ html|safe }}
</div>
<script>
  // Wait for web fonts if supported
  try { if (document.fonts) { document.fonts.ready.then(()=>{}); } } catch(e) {}
</script>
</body>
</html>
""")
