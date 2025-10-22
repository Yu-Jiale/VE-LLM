import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch

from .config import RenderSettings, apply_overrides, load_render_settings
from .pipeline_utils import build_inputs_with_image_seq, make_zero_image_stubs
from .render import compose_html_doc, render_markdown
from deepseek_ocr_runner import encode_page_to_sequence
from deepseek_ocr_runner.decoder import decode_generate
from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
)
from transformers import AutoTokenizer


def _normalize_text(text: str) -> str:
    return text.replace("\r", "").strip()


def _tokenize_words(text: str) -> List[str]:
    return _normalize_text(text).replace("\n", " ").split()


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n or n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize_words(reference)
    hyp_tokens = _tokenize_words(hypothesis)
    if not hyp_tokens:
        return 0.0

    max_order = 4
    weights = [1.0 / max_order] * max_order
    precisions = []
    for order in range(1, max_order + 1):
        ref_counts = _ngram_counts(ref_tokens, order)
        hyp_counts = _ngram_counts(hyp_tokens, order)
        if not hyp_counts:
            precisions.append(1e-9)
            continue
        overlap = 0
        for ngram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))
        precisions.append(overlap / max(sum(hyp_counts.values()), 1e-9))

    smoothed = [p if p > 0 else 1e-9 for p in precisions]
    geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, smoothed)))

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(hyp_len, 1))
    return float(bp * geo_mean)


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            tmp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize_words(reference)
    hyp_tokens = _tokenize_words(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(hyp_tokens)
    if recall + precision == 0:
        return 0.0
    return float((2 * recall * precision) / (recall + precision))


def _levenshtein_distance(ref: str, hyp: str) -> int:
    ref = _normalize_text(ref)
    hyp = _normalize_text(hyp)
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r_char in enumerate(ref, start=1):
        curr = [i]
        for j, h_char in enumerate(hyp, start=1):
            cost = 0 if r_char == h_char else 1
            curr.append(
                min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def compute_char_accuracy(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 1.0 if not hyp else 0.0
    distance = _levenshtein_distance(ref, hyp)
    return float(max(0.0, 1.0 - distance / max(len(ref), 1)))


def evaluate_translator_fidelity(
    md_text: str,
    *,
    model_dir: str = "./models/DeepSeek-OCR",
    base_size: int = 1024,
    image_size: int = 640,
    device: Optional[str] = None,
    out_dir: str = "./md_render_pipeline/render_out",
    text_before: str = "请识别这张图片的文字：",
    text_after: str = "\n请逐行输出，不要丢字。",
    render_config_path: Optional[str] = None,
) -> Dict[str, object]:
    base_cfg = load_render_settings(render_config_path) if render_config_path else RenderSettings()
    cfg = apply_overrides(base_cfg, out_dir=out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    pages, cfg_fp = render_markdown(md_text, cfg)
    if not pages:
        raise RuntimeError("渲染失败：未生成任何页面 PNG。")

    html_path = os.path.join(cfg.out_dir, "translator_fidelity.html")
    html_full = compose_html_doc(md_text, cfg)
    with open(html_path, "w", encoding="utf-8") as f_html:
        f_html.write(html_full)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_out = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model = DeepseekOCRForCausalLM.from_pretrained(model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)

    decoded_pages: List[str] = []
    for page_idx, page_path in enumerate(pages):
        page_features = encode_page_to_sequence(
            model,
            page_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=True,
            device=device,
            out_dtype=dtype_out,
        )

        inputs_embeds, attention_mask, prefill_len = build_inputs_with_image_seq(
            model,
            tok,
            page_features.seq,
            text_before=text_before,
            text_after=text_after,
            device=device,
        )

        images_stub, _ = make_zero_image_stubs(1, base_size, image_size, device, dtype_out)
        enc_outputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prefill_lens": [prefill_len],
            "decode_images_stub": images_stub,
            "decode_images_spatial_stub": torch.tensor([[0, 0]], dtype=torch.long, device=device),
            "base_size": base_size,
            "image_size": image_size,
        }

        out_text = decode_generate(
            model,
            tok,
            enc_outputs,
            max_new_tokens=512,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )[0]
        decoded_pages.append(out_text)
        print(f"[fidelity] page {page_idx+1}/{len(pages)} decoded length={len(out_text)} chars")

    recognized_text = "\n".join(decoded_pages)
    reference_text = _normalize_text(md_text)

    metrics = {
        "bleu": compute_bleu(reference_text, recognized_text),
        "rouge_l": compute_rouge_l(reference_text, recognized_text),
        "char_accuracy": compute_char_accuracy(reference_text, recognized_text),
        "ref_length_chars": len(reference_text),
        "hyp_length_chars": len(_normalize_text(recognized_text)),
        "num_pages": len(pages),
        "cfg_fingerprint": cfg_fp,
    }

    report = {
        "metrics": metrics,
        "raw_markdown": md_text,
        "recognized_pages": decoded_pages,
        "recognized_text": recognized_text,
        "rendered_pages": pages,
        "rendered_html": html_path,
    }

    report_path = os.path.join(cfg.out_dir, "translator_fidelity.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[fidelity] metrics saved to {report_path}")
    return report


if __name__ == "__main__":
    paragraph = (
        "在这个阶段，我们主要关注 DeepSeek-OCR 翻译器的保真度。请输入长段落文本，最好混合标题、"
        "列表、代码片段和普通段落，以模拟真实文档情形。我们建议准备两页以上的内容：\n\n"
        "1. 第一部分介绍背景，例如视觉符号空间与压缩编码的动机；\n"
        "2. 第二部分具体描述实验流程，包括 Markdown 渲染、PNG 分页、视觉序列编码；\n"
        "3. 如需压力测试，可以附带一个简单的 Python 代码块：\n\n"
        "```python\n"
        "def pipeline_stage(stage: str) -> str:\n"
        "    return f\"当前阶段: {stage}, 请确认文本没有换行溢出。\"\n"
        "```\n\n"
        "完成后，请检查生成的译文是否忠实，并对比 `translator_fidelity.json` 中的指标。"
    )
    sample_md = "# 保真度评估示例\n\n" + (paragraph + "\n\n") * 8
    evaluate_translator_fidelity(sample_md, out_dir="./md_render_pipeline/render_out")
