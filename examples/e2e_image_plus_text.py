# -*- coding: utf-8 -*-
"""
E2E smoke test: 连续视觉序列 + 文本 token → inputs_embeds → decode_generate()

流程：
 1) encode_page_to_sequence() 得到 (T_img, D) 的视觉序列；
 2) 用 tokenizer 把文本的前/后缀编码成嵌入；
 3) 拼成 [BOS] + text_before + <image_seq> + text_after 的 inputs_embeds；
 4) 构造 decode 需要的零图像占位（让视觉注入短路）；
 5) 调用 decode_generate(model, tokenizer, enc_outputs) 生成文本。

注意：此脚本不依赖 <image> 特殊 token 的 id；我们直接在嵌入层级拼接视觉序列。
"""

from __future__ import annotations
from typing import List, Tuple
import os
from pathlib import Path

import torch

# ===== 顶部参数（按需修改） =====
MODEL_DIR = "./models/DeepSeek-OCR"
IMAGE_PATH = "./test_document.png"

# 解码超参
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # 0=贪心
TOP_K = 0
TOP_P = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_OUT = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

# 视觉分支尺寸（与 encode 保持一致）
BASE_SIZE = 1024
IMAGE_SIZE = 640

# 文本提示：把 <image> 当作“占位符语义”，我们不依赖它的 token id
TEXT_BEFORE = "请识别这张图片的文字："
TEXT_AFTER  = "\n请逐行输出，不要丢字。"

# ===============================

from deepseek_ocr_runner import encode_page_to_sequence
from deepseek_ocr_runner.decoder import decode_generate  # 你之前的手写解码器
from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
)
from transformers import AutoTokenizer


@torch.no_grad()
def _make_zero_image_stubs(
    batch_size: int,
    base_size: int,
    image_size: int,
    device: str,
    dtype: torch.dtype,
):
    """
    构造让视觉注入短路的“零图像占位”。DeepSeek-OCR 的 forward 会检查 images[0][1] 的和是否为 0。
    """
    zeros_crop = torch.zeros((1, 3, base_size, base_size), dtype=dtype, device=device)
    zeros_ori  = torch.zeros((1, 3, image_size, image_size), dtype=dtype, device=device)
    images = [(zeros_crop, zeros_ori) for _ in range(batch_size)]
    images_spatial_crop = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
    return images, images_spatial_crop


@torch.no_grad()
def _text_to_embeds(model, tokenizer, text: str, device: str):
    """把文本编码为 input_ids，再取嵌入层得到 (T_text, H)。"""
    # 不加特殊符（BOS/EOS我们手动控制）
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return torch.zeros((0, model.config.hidden_size), dtype=model.dtype, device=device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
    embeds = model.get_input_embeddings()(ids_t)[0]  # (T,H)
    return embeds


@torch.no_grad()
def build_inputs_with_image_seq(
    model,
    tokenizer,
    image_seq: torch.Tensor,     # (T_img, H)
    text_before: str,
    text_after: str,
    device: str,
):
    """
    拼接顺序： [BOS] + text_before + image_seq + text_after
    返回：
      inputs_embeds: (1, T_total, H)
      attention_mask: (1, T_total)
      prefill_len: int
    """
    H = image_seq.shape[1]
    # BOS
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = 0  # 兜底
    bos = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    bos_embed = model.get_input_embeddings()(bos)[0]  # (1,H)

    # 文本嵌入
    emb_before = _text_to_embeds(model, tokenizer, text_before, device)
    emb_after  = _text_to_embeds(model, tokenizer, text_after, device)

    # dtype 对齐
    tgt_dtype = image_seq.dtype
    bos_embed = bos_embed.to(tgt_dtype)
    emb_before = emb_before.to(tgt_dtype)
    emb_after  = emb_after.to(tgt_dtype)

    seq = torch.cat([bos_embed, emb_before, image_seq, emb_after], dim=0)  # (T_total, H)
    attn = torch.ones((1, seq.shape[0]), dtype=torch.long, device=device)
    return seq.unsqueeze(0), attn, seq.shape[0]


def main():
    assert Path(MODEL_DIR).exists(), f"MODEL_DIR not found: {MODEL_DIR}"
    assert Path(IMAGE_PATH).exists(), f"IMAGE not found: {IMAGE_PATH}"

    print(f"[load] model/tokenizer from {MODEL_DIR}")
    model = DeepseekOCRForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 1) encode image → continuous visual sequence
    print("[encode] image → (T_img, H) visual sequence …")
    pf = encode_page_to_sequence(
        model,
        IMAGE_PATH,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=True,
        device=DEVICE,
        out_dtype=DTYPE_OUT,
    )
    img_seq = pf.seq  # (T_img, H)

    # 2) fuse text + image sequence → inputs_embeds
    print("[fuse] build inputs_embeds with text + image sequence …")
    inputs_embeds, attention_mask, prefill_len = build_inputs_with_image_seq(
        model, tok, img_seq, TEXT_BEFORE, TEXT_AFTER, DEVICE
    )

    # 3) build decode stubs (zero images) for DeepSeek-OCR forward to short-circuit vision injection
    images_stub, spatial_stub = _make_zero_image_stubs(
        batch_size=1,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        dtype=DTYPE_OUT,
    )

    # 4) 组装 enc_outputs，复用你已有的 decode_generate()
    enc_outputs = {
        "inputs_embeds": inputs_embeds,             # (1, T, H)
        "attention_mask": attention_mask,           # (1, T)
        "prefill_lens": [prefill_len],
        "decode_images_stub": images_stub,          # List[(zeros_crop, zeros_ori)]
        "decode_images_spatial_stub": torch.tensor([[0, 0]], dtype=torch.long, device=DEVICE),
        "base_size": BASE_SIZE,
        "image_size": IMAGE_SIZE,
    }

    # 5) decode
    print("[decode] generating …")
    outs = decode_generate(
        model,
        tok,
        enc_outputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    print("=========== OUTPUT ===========")
    print(outs[0])
    print("===========  DONE  ===========")

if __name__ == "__main__":
    main()