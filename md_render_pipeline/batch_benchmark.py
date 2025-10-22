import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .chromium import RenderSession
from .config import RenderSettings, apply_overrides, load_render_settings
from .render import render_markdown

DEFAULT_SNIPPETS: List[Tuple[str, str]] = [
    (
        "长文档综述",
        "# 视觉符号空间：项目综述\n\n"
        "## 背景简介\n"
        "- [x] 研究 Transformer 长上下文的瓶颈；\n"
        "- [x] 调研优秀的 OCR 编/解码器；\n"
        "- [ ] 统一视觉符号与语言符号。\n\n"
        "## 核心问题\n"
        "视觉符号空间能否承载长文档？需要回答三个问题：\n"
        "1. 渲染流程是否稳定；\n"
        "2. 编码序列是否紧凑；\n"
        "3. 还原文本是否保真。\n\n"
        "> 我们必须在工程效率与信息保真之间找到平衡。"
    ),
    (
        "嵌套清单与表格",
        "# 迭代计划 (Q1)\n\n"
        "### 任务列表\n"
        "- [x] 搭建渲染引擎\n"
        "  - [x] Markdown → HTML\n"
        "  - [x] HTML → PDF → PNG\n"
        "- [ ] 翻译器评估\n"
        "  - [x] 构建指标 (BLEU/ROUGE)\n"
        "  - [ ] 建立回归数据集\n\n"
        "### 资源投入\n"
        "| 模块 | 负责人 | 预估 GPU 时长 (h) | 当前进度 |\n"
        "| --- | --- | --- | --- |\n"
        "| 渲染管线 | Alice | 40 | 95% |\n"
        "| OCR 编码 | Bob | 60 | 80% |\n"
        "| 视觉符号训练 | Carol | 120 | 45% |\n"
        "| 评估与可视化 | Diana | 50 | 30% |\n"
    ),
    (
        "代码混合篇",
        "# 多语言代码片段\n\n"
        "```python\n"
        "def tokenize_visual_sequence(pages):\n"
        "    tokens = []\n"
        "    for page in pages:\n"
        "        tokens.extend(extract(page))\n"
        "    return tokens\n"
        "```\n\n"
        "```bash\n"
        "python -m md_render_pipeline.render_sanity --config configs/high_dpi.json \\\n"
        "  --markdown samples/long_doc.md --save-html\n"
        "```\n\n"
        "```json\n"
        "{\n"
        "  \"model\": \"DeepSeek-OCR\",\n"
        "  \"hidden_size\": 4096,\n"
        "  \"vision_tokens\": 16384\n"
        "}\n"
        "```"
    ),
    (
        "多语言对话",
        "# Support Transcript\n\n"
        "**User (中文)**: 请问如何把长文档转成视觉符号序列？\n\n"
        "**Engineer (English)**: First, render the Markdown into paginated PNGs. Then run the encoder to obtain dense visual tokens.\n\n"
        "**User (日本語)**: そのトークン列を保存するフォーマットは？\n\n"
        "**Engineer**: We store them as memmapped float16 arrays along with JSON metadata."
    ),
    (
        "研发会议纪要",
        "# 研发会议纪要\n\n"
        "> 日期：2025-02-10  地点：线上会议\n\n"
        "### 议程\n"
        "1. 译码回归异常\n"
        "2. 渲染性能开销\n"
        "3. 数据加密合规\n\n"
        "### 待办事项\n"
        "- [ ] 调查翻译器在 8k 以上序列长度的准确率下降问题；\n"
        "- [x] 编写高 DPI 渲染脚本；\n"
        "- [ ] 同步安全团队确认数据导出策略。\n\n"
        "### 备注\n"
        "- 需要持续追踪 GPU 占用；\n"
        "- 建议新增日志采样 (每 200 token)。"
    ),
    (
        "数学推导",
        "# 连续视觉序列的损失函数\n\n"
        "我们定义目标函数：\n\n"
        "$$\\mathcal{L} = -\\sum_{t=1}^T \\log P(v_t | v_{<t})$$\n\n"
        "其中 $v_t$ 为视觉符号。为了分析梯度，我们考虑 $\\nabla_{\\theta} \\mathcal{L}$：\n"
        "$\\nabla_{\\theta} \\mathcal{L} = -\\sum_{t=1}^T \\nabla_{\\theta} \\log P(v_t | v_{<t}; \\theta)$。\n"
        "进一步地，我们对归一化常数使用拉普拉斯近似。"
    ),
    (
        "配置与日志",
        "# 配置快照\n\n"
        "```toml\n"
        "[render]\n"
        "dpi = 480\n"
        "page_size = \"A4\"\n"
        "margins_mm = [5, 5, 5, 5]\n"
        "[[render.fonts]]\n"
        "family = \"Noto Sans CJK SC\"\n"
        "role = \"primary\"\n"
        "```\n\n"
        "```log\n"
        "[2025-02-15 10:03:01] INFO render: pages=5 dpi=480 time=3.82s\n"
        "[2025-02-15 10:03:05] INFO encoder: seq_len=9421 tokens=1536\n"
        "```"
    ),
    (
        "引用与脚注",
        "# 参考资料\n\n"
        "> “Visual tokenization is an emerging research direction that bridges document understanding and sequence modeling.”\n\n"
        "相关工作包括 Glyph[^glyph] 与 DeepSeek-OCR。\n\n"
        "[^glyph]: Glyph: Bridging Text and Image for Long-Context LLMs, 2024."
    ),
    (
        "图文混合",
        "# 渲染示例\n\n"
        "![pipeline](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='600' height='240'><rect width='600' height='240' fill='%232563eb'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='white' font-size='36'>Render Pipeline</text></svg>)\n\n"
        "> 图 1：Markdown → Chromium → PDF → PNG\n\n"
        "我们需要验证图片缩放、阴影、以及文字与图片的间距是否合理。"
    ),
    (
        "Stress 文本块",
        "# 文本压测\n\n" +
        (
            "视觉序列的统计是压测环节的关键指标。我们重复这段话以产生足够长的文本，从而测试渲染分页、换行策略、代码高亮缓存和 MathML 样式的稳定性。"
            " 在真实项目中，我们经常需要处理 100 页以上的文档，因此单页性能必须可控。"
        ) * 12
    ),
]


def run_benchmark(out_dir: str, repeat: int, snippets: List[Tuple[str, str]], config_path: Optional[str]) -> None:
    results = []
    base_cfg = load_render_settings(config_path) if config_path else RenderSettings()

    with RenderSession() as session:
        for idx in range(repeat):
            for name, md in snippets:
                sample_dir = Path(out_dir) / f"iter_{idx}" / name.replace(" ", "_")
                sample_dir.mkdir(parents=True, exist_ok=True)
                cfg = apply_overrides(base_cfg, out_dir=str(sample_dir))

                start = time.time()
                pngs, fingerprint = render_markdown(md, cfg, session=session)
                elapsed = time.time() - start
                results.append(
                    {
                        "iteration": idx,
                        "name": name,
                        "pages": len(pngs),
                        "duration_sec": elapsed,
                        "fingerprint": fingerprint,
                        "output_dir": str(sample_dir),
                    }
                )
                print(
                    f"[batch] iter={idx} name={name} pages={len(pngs)} "
                    f"time={elapsed:.3f}s fingerprint={fingerprint} dir={sample_dir}"
                )

    summary = {
        "total_runs": len(results),
        "avg_time_sec": sum(r["duration_sec"] for r in results) / max(len(results), 1),
        "max_time_sec": max((r["duration_sec"] for r in results), default=0.0),
        "min_time_sec": min((r["duration_sec"] for r in results), default=0.0),
    }

    print("========== Batch Render Summary ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")

    report_path = Path(out_dir) / "batch_render_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(
                f"iter={record['iteration']}, name={record['name']}, "
                f"pages={record['pages']}, time={record['duration_sec']:.3f}s, "
                f"fingerprint={record['fingerprint']}, dir={record['output_dir']}\n"
            )
        f.write("\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"[batch] Report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量渲染基准测试")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./md_render_pipeline/render_out/batch_benchmark",
        help="输出 PNG 的目录（默认：./md_render_pipeline/render_out/batch_benchmark）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="重复所有样本文档的次数（默认：3）",
    )
    parser.add_argument(
        "--snippets-file",
        type=str,
        help="包含测试 Markdown 内容的文件（JSON lines，每行 {\"name\": str, \"text\": str}）。",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="渲染配置 JSON 文件路径。",
    )
    args = parser.parse_args()

    snippets = DEFAULT_SNIPPETS
    if args.snippets_file:
        import json

        snippets = []
        with open(args.snippets_file, "r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                snippets.append((payload["name"], payload["text"]))

    run_benchmark(args.out_dir, args.repeat, snippets, args.config)


if __name__ == "__main__":
    main()
