import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .chromium import RenderSession
from .config import RenderSettings, apply_overrides, load_render_settings
from .render import render_markdown

DEFAULT_SNIPPETS: List[Tuple[str, str]] = [
    (
        "短段落-中文",
        "# 短段落示例\n\n这是一个简短的段落，用来测试渲染质量与字体表现。我们关注是否出现溢出与排版错位。",
    ),
    (
        "技术报告片段",
        "# 实验流程\n\n"
        "1. 渲染 Markdown，生成 PDF 与 PNG；\n"
        "2. 运行 DeepSeek-OCR 编码器提取视觉序列；\n"
        "3. 自回归模型在视觉符号空间推理；\n"
        "4. 解码器还原文本并对比原文，计算 BLEU/ROUGE。\n\n"
        "```python\n"
        "def stage(name: str) -> str:\n"
        "    return f\"当前阶段: {name}\"\n"
        "```\n",
    ),
    (
        "长篇段落",
        "# 视觉符号空间研究背景\n\n" +
        (
            "视觉符号空间旨在将长文档映射到压缩的视觉 token 序列，从而绕开 Transformer "
            "的二次方复杂度瓶颈。这一设想依托成熟的 OCR 编码器与渲染流程，需要在多页场景下保持稳定的 "
            "排版质量与高保真度。"
        ) * 8
    ),
    (
        "多语言混排",
        "# Mixed Language\n\n"
        "This document mixes English paragraphs with 中文段落 and `inline code snippets`. "
        "我们需要确保不同语言之间的字体、行高和断行行为保持一致。\n\n"
        "关键字：Token compression, Long-context reasoning, Auto-regressive modeling.",
    ),
    (
        "表格与引用",
        "# 数据对齐示例\n\n"
        "> 数据表格需要横纵方向的视觉对齐，确保数字列不会错位。\n\n"
        "| 指标 | 数值 |\n"
        "| --- | --- |\n"
        "| Precision | 0.912 |\n"
        "| Recall | 0.887 |\n"
        "| F1 | 0.899 |\n\n"
        "结论：视觉符号空间在信息压缩与还原上仍有优化空间。",
    ),
    (
        "项目周报",
        "# 项目周报\n\n"
        "## 本周进展\n"
        "- 完成 DeepSeek-OCR 翻译器保真度评估；\n"
        "- 初步搭建视觉符号空间 autoregressive 训练脚本；\n"
        "- 引入批量渲染基准，测量页面生成耗时。\n\n"
        "## 下周计划\n"
        "1. 扩充语料，覆盖更多文档样式；\n"
        "2. 优化渲染模板，强化视觉层级；\n"
        "3. 设计视觉 token 数据集落盘格式。\n\n"
        "## 风险与关注点\n"
        "- 渲染性能可能成为瓶颈，需要持续 profiling；\n"
        "- 视觉序列长度过长时，解码速度需评估。\n",
    ),
    (
        "公式混排",
        "# 数学公式示例\n\n"
        "我们在视觉空间中展示数学表达式：\n\n"
        "$$ E = mc^2 $$\n\n"
        "以及行内公式 $\\int_a^b f(x) dx$ ，测试渲染对 LaTeX 片段的兼容性（若无公式引擎，会显示原文）。\n",
    ),
    (
        "代码评审",
        "# 代码评审记录\n\n"
        "```python\n"
        "class VisualEncoder(nn.Module):\n"
        "    def forward(self, images):\n"
        "        features = self.backbone(images)\n"
        "        return self.projector(features)\n"
        "```\n\n"
        "- [ ] 检查 projector 初始化；\n"
        "- [x] 统一 dtype 为 bfloat16；\n"
        "- [ ] 添加单元测试覆盖边界输入。\n",
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
