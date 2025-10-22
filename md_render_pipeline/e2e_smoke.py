import json
import os
from typing import List, Optional, Tuple

import torch

from .config import RenderSettings, apply_overrides, load_render_settings
from .pipeline_utils import (
    build_inputs_with_image_seq,
    make_zero_image_stubs,
)
from .render import render_markdown
from deepseek_ocr_runner import encode_page_to_sequence
from deepseek_ocr_runner.decoder import decode_generate
from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
)
from transformers import AutoTokenizer

def run_e2e(
    md_text: str,
    model_dir: str = "./models/DeepSeek-OCR",
    base_size: int = 1024,
    image_size: int = 640,
    device: Optional[str] = None,
    out_dir: str = "./md_render_pipeline/render_out",
    process_all_pages: bool = False,
    text_before: str = "请识别这张图片的文字：",
    text_after: str = "\n请逐行输出，不要丢字。",
    render_config_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    base_cfg = load_render_settings(render_config_path) if render_config_path else RenderSettings()
    cfg = apply_overrides(base_cfg, out_dir=out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Render Markdown -> PNGs
    pngs, cfg_fp = render_markdown(md_text, cfg)
    print(f"[render] pages={len(pngs)}, cfg={cfg_fp}")
    if not pngs:
        raise RuntimeError("渲染失败：未生成任何页面 PNG。")

    # 2) Load OCR model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_out = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[load] model: {model_dir} on {device}")
    model = DeepseekOCRForCausalLM.from_pretrained(model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)

    target_pages = pngs if process_all_pages else [pngs[0]]
    ocr_outputs: List[str] = []

    for idx, page_path in enumerate(target_pages):
        # 3) Encode each page -> continuous visual sequence
        pf = encode_page_to_sequence(
            model,
            page_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=True,
            device=device,
            out_dtype=dtype_out,
        )
        img_seq = pf.seq
        print(f"[encode] page={idx+1}/{len(target_pages)} seq_shape={tuple(img_seq.shape)}")

        # 4) Build inputs embeds
        inputs_embeds, attention_mask, prefill_len = build_inputs_with_image_seq(
            model,
            tok,
            img_seq,
            text_before=text_before,
            text_after=text_after,
            device=device,
        )

        # 5) Decode (single-sample batch)
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

        outs = decode_generate(
            model,
            tok,
            enc_outputs,
            max_new_tokens=256,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        text = outs[0]
        ocr_outputs.append(text)
        print(f"=========== OCR OUTPUT (page {idx+1}) ===========")
        print(text)
        print("==================================")

    meta = {
        "pages": pngs,
        "cfg_fingerprint": cfg_fp,
        "ocr_text": ocr_outputs if process_all_pages else ocr_outputs[0],
        "process_all_pages": process_all_pages,
    }
    with open(os.path.join(cfg.out_dir, "e2e_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return ocr_outputs, pngs

if __name__ == "__main__":
    paragraph = (
        "这是一个用于多页 smoke 的示例段落。它会重复多次以确保渲染产生多页。"
        "请仔细检查分页之后 OCR 的稳定性以及字符保真度。\n\n"
    )
    sample_md = "# 多页 Smoke 测试\n\n" + paragraph * 40
    run_e2e(sample_md, process_all_pages=True)
