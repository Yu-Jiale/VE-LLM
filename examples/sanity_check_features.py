# -*- coding: utf-8 -*-
"""
Sanity check for the *continuous* visual encoder (features.py).

它完成 3 件事：
 1) 把一组页面图片编码成一维连续视觉序列 (T, D)；
 2) 打印每页/整批的统计信息（长度、均值/方差、向量范数分布）；
 3) 把序列作为 inputs_embeds 走一遍冻结 DeepSeek-OCR 的前向（带零图像占位），确认不崩。
"""

from __future__ import annotations
from pathlib import Path
import os
import sys
from typing import List, Tuple

import torch

# ---------------- 顶部配置（按需修改） ----------------
MODEL_DIR = "./models/DeepSeek-OCR"
IMAGE_PATHS: List[str] = [
    "./test_document.png",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
DO_FORWARD_CHECK = True
PAD_TO_MAX = True

# 与 encode 时保持一致
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
# -----------------------------------------------------

from deepseek_ocr_runner import encode_pages_to_batch
from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
)


def _assert_paths():
    ok = True
    m = Path(MODEL_DIR)
    if not (m.exists() and m.is_dir()):
        print(f"[ERR] MODEL_DIR not found: {MODEL_DIR}")
        ok = False
    for p in IMAGE_PATHS:
        if not Path(p).exists():
            print(f"[ERR] image not found: {p}")
            ok = False
    if not ok:
        sys.exit(1)


def _print_env():
    print("========== ENV ==========")
    print(f"cwd: {os.getcwd()}")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
    print(f"DEVICE: {DEVICE}  OUT_DTYPE: {OUT_DTYPE}")
    print("=========================")


def make_zero_image_stubs(
    batch_size: int,
    base_size: int,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[list, torch.Tensor]:
    """
    构造与 DeepSeek-OCR forward 兼容的“零图像占位”：
      images: List[Tuple(zeros_crop:(1,3,base,base), zeros_ori:(1,3,image,image))] * B
      images_spatial_crop: Tensor(B,2) 全 0
    满足 forward 里的条件：
      torch.sum(images[0][1]) == 0  -> 视觉分支短路
    """
    zeros_crop = torch.zeros((1, 3, base_size, base_size), dtype=dtype, device=device)
    zeros_ori  = torch.zeros((1, 3, image_size, image_size), dtype=dtype, device=device)

    images = [(zeros_crop, zeros_ori) for _ in range(batch_size)]
    images_spatial_crop = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
    return images, images_spatial_crop


def main():
    _print_env()
    _assert_paths()

    # 1) 加载冻结模型
    print(f"[load] DeepSeek-OCR from: {MODEL_DIR}")
    model = DeepseekOCRForCausalLM.from_pretrained(MODEL_DIR)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) 编码页面 → 连续序列
    print("[encode] pages → (T, D) sequences …")
    enc_out = encode_pages_to_batch(
        model,
        IMAGE_PATHS,
        device=DEVICE,
        out_dtype=OUT_DTYPE,
        pad_to_max=PAD_TO_MAX,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=CROP_MODE,
    )

    feats = enc_out["features"]
    batch_seq = enc_out["batch_seq"]
    batch_attn = enc_out["batch_attn"]
    lengths = enc_out["lengths"]
    D = enc_out["D"]

    # 3) 打印统计
    print("---------- PER-PAGE STATS ----------")
    for i, pf in enumerate(feats):
        seq = pf.seq
        t = seq.shape[0]
        mean = seq.float().mean().item()
        std = seq.float().std().item()
        norms = torch.linalg.vector_norm(seq.float(), dim=-1)
        med = norms.median().item()
        p95 = norms.quantile(0.95).item()
        print(
            f"[page {i}] T={t:<6d}  local_hw={pf.meta.local_hw}  "
            f"global_hw={pf.meta.global_hw}  crop_wh={pf.meta.crop_wh}  "
            f"mean={mean:.4f}  std={std:.4f}  ||x||_med={med:.3f}  p95={p95:.3f}"
        )

    print("------------ BATCH STATS -----------")
    B = len(feats)
    T_max = max(lengths)
    print(f"B={B}  T_max={T_max}  D={D}")
    if PAD_TO_MAX and batch_seq is not None:
        valid_mask = torch.zeros_like(batch_attn, dtype=torch.bool)
        for i, L in enumerate(lengths):
            valid_mask[i, :L] = True
        valid_vecs = batch_seq[valid_mask]
        mean = valid_vecs.float().mean().item()
        std = valid_vecs.float().std().item()
        norms = torch.linalg.vector_norm(valid_vecs.float(), dim=-1)
        med = norms.median().item()
        p95 = norms.quantile(0.95).item()
        print(f"[all valid tokens] mean={mean:.4f} std={std:.4f} ||x||_med={med:.3f} p95={p95:.3f}")
    print("------------------------------------")

    # 4) 前向检查（带“零图像占位”以短路视觉注入）
    if DO_FORWARD_CHECK and PAD_TO_MAX and batch_seq is not None:
        print("[forward] quick logits check (prefill, with zero-image stubs) …")
        dummy_ids = torch.zeros((B, T_max), dtype=torch.long, device=DEVICE)

        images_stub, images_spatial_crop = make_zero_image_stubs(
            batch_size=B,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            device=DEVICE,
            dtype=OUT_DTYPE,
        )

        # autocast 上下文
        if DEVICE == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=OUT_DTYPE)
        else:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            autocast_ctx = _NullCtx()

        with torch.no_grad(), autocast_ctx:
            out = model(
                input_ids=dummy_ids,           # 仅占位
                inputs_embeds=batch_seq,       # (B, T, D)
                attention_mask=batch_attn,     # (B, T)
                use_cache=False,
                return_dict=True,
                # 关键：传零图像占位，让视觉分支短路
                images=images_stub,            # List[(zeros_crop, zeros_ori)] * B
                images_seq_mask=None,
                images_spatial_crop=images_spatial_crop,  # (B, 2) zeros
            )
            logits = out.logits
            print(f"logits shape: {tuple(logits.shape)}  (expect ~ (B, T, vocab))")

    print("[done] sanity check passed.")


if __name__ == "__main__":
    main()