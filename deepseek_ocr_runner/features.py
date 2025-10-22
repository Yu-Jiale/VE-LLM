# features.py — minimal continuous encoder

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image, ImageOps

# DeepSeek-OCR modified components (must exist in your repo)
from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    load_image,
    dynamic_preprocess,
    BasicImageTransform,
)


@dataclass
class PageMeta:
    # shapes for local/global raster grids
    local_hw: Tuple[int, int]   # (h_local, w_local) after regrid; (0,0) if no local
    global_hw: Tuple[int, int]  # (h_global, w_global)
    crop_wh: Tuple[int, int]    # (width_crop_num, height_crop_num)

    # slices for debugging/inspection (within seq)
    slice_local: slice
    slice_global: slice
    slice_viewsep: slice

    base_size: int
    image_size: int


@dataclass
class PageFeatures:
    seq: torch.Tensor                     # (T, D), continuous visual token sequence (bf16/fp16)
    meta: PageMeta
    specials: Dict[str, torch.Tensor]     # {"row_nl": (D,), "view_sep": (D,)}
    dtype: torch.dtype
    device: torch.device


TensorableImage = Union[str, Image.Image]


@torch.no_grad()
def encode_page_to_sequence(
    model,
    image: TensorableImage,
    *,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    device: str = "cuda",
    out_dtype: torch.dtype = torch.bfloat16,
) -> PageFeatures:
    """
    Encode a single rendered page into a *1D continuous* visual-embedding sequence, in the exact
    order used by DeepSeek-OCR's visual concatenation pipeline:
        local (row-major with row-newline) -> global (row-major with row-newline) -> view_separator

    Returns PageFeatures(seq, meta, specials).
    """
    model.eval()

    # ---- 1) load & normalize ----
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = load_image(image).convert("RGB")

    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    if crop_mode:
        if pil.size[0] <= 640 and pil.size[1] <= 640:
            crop_ratio = (1, 1)
            images_crop_raw: List[Image.Image] = []
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess(pil)  # (list[PIL], (Wc, Hc))
    else:
        raise NotImplementedError("crop_mode=False is intentionally unsupported for stability.")

    # Global (padded) view -> (1,3,base,base)
    global_view = ImageOps.pad(
        pil,
        (base_size, base_size),
        color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5)),
    )
    images_ori = image_transform(global_view).to(out_dtype).unsqueeze(0).to(device)

    # Local crops -> (Nc,3,640,640) or a zero placeholder (forces local=empty path)
    Wc, Hc = crop_ratio
    if (Wc > 1) or (Hc > 1):
        crops = [image_transform(im).to(out_dtype) for im in images_crop_raw]
        images_crop = torch.stack(crops, dim=0).to(device)
    else:
        images_crop = torch.zeros((1, 3, base_size, base_size), dtype=out_dtype, device=device)

    # ---- 2) pull model internals ----
    sam_model     = model.model.sam_model
    vision_model  = model.model.vision_model
    projector     = model.model.projector
    row_nl_vec    = model.model.image_newline.squeeze(0)  # (D,)
    view_sep_vec  = model.model.view_seperator            # (D,)

    # ---- 3) forward to get local/global tokens ----
    with torch.autocast(device_type="cuda", dtype=out_dtype):
        # Global branch
        g_f1 = sam_model(images_ori)
        g_f2 = vision_model(images_ori, g_f1)
        glob = torch.cat((g_f2[:, 1:], g_f1.flatten(2).permute(0, 2, 1)), dim=-1)
        glob = projector(glob).squeeze(0)                  # (Hg*Wg, D)
        HgWg, D = glob.shape
        Hg = Wg = int(HgWg ** 0.5)
        assert Hg * Wg == HgWg, f"Global grid is not square: {HgWg}"
        glob = glob.view(Hg, Wg, D)
        g_row_nl = row_nl_vec.expand(Hg, 1, D)
        glob_1d = torch.cat([glob, g_row_nl], dim=1).reshape(-1, D)  # (Hg*(Wg+1), D)

        # Local branch (may be empty)
        if (images_crop.numel() > 0) and (images_crop.abs().sum().item() != 0):
            l_f1 = sam_model(images_crop)
            l_f2 = vision_model(images_crop, l_f1)
            local = torch.cat((l_f2[:, 1:], l_f1.flatten(2).permute(0, 2, 1)), dim=-1)
            local = projector(local)                         # (Nc, hw2, D)

            hw2 = local.shape[1]
            h2 = w2 = int(hw2 ** 0.5)
            assert h2 * w2 == hw2, f"Local feature not square: {hw2}"

            # Regrid to (Hc*h2, Wc*w2, D) with row-major order
            local = (
                local.view(Hc, Wc, h2, w2, D)
                     .permute(0, 2, 1, 3, 4)
                     .reshape(Hc * h2, Wc * w2, D)
            )
            l_row_nl = row_nl_vec.expand(Hc * h2, 1, D)
            local_1d = torch.cat([local, l_row_nl], dim=1).reshape(-1, D)
            local_hw = (Hc * h2, Wc * w2)
        else:
            local_1d = torch.zeros((0, glob.shape[-1]), dtype=out_dtype, device=device)
            local_hw = (0, 0)

        view_sep = view_sep_vec.unsqueeze(0)  # (1, D)

    # ---- 4) unified reading path: local -> global -> view_sep ----
    T_local = local_1d.shape[0]
    T_glob  = glob_1d.shape[0]
    seq = torch.cat([local_1d, glob_1d, view_sep], dim=0)    # (T, D)

    meta = PageMeta(
        local_hw=local_hw,
        global_hw=(Hg, Wg),
        crop_wh=(Wc, Hc),
        slice_local=slice(0, T_local),
        slice_global=slice(T_local, T_local + T_glob),
        slice_viewsep=slice(T_local + T_glob, T_local + T_glob + 1),
        base_size=base_size,
        image_size=image_size,
    )

    specials = {
        "row_nl":  row_nl_vec.detach().to(out_dtype).to(device),
        "view_sep":view_sep_vec.detach().to(out_dtype).to(device),
    }

    return PageFeatures(
        seq=seq,
        meta=meta,
        specials=specials,
        dtype=out_dtype,
        device=torch.device(device),
    )


@torch.no_grad()
def encode_pages_to_batch(
    model,
    images: List[TensorableImage],
    *,
    device: str = "cuda",
    out_dtype: torch.dtype = torch.bfloat16,
    pad_to_max: bool = True,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Batch wrapper for training/storage convenience. Returns padded batch (optional) and per-page
    PageFeatures (for meta/specials inspection).
    """
    feats: List[PageFeatures] = [
        encode_page_to_sequence(
            model,
            img,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            device=device,
            out_dtype=out_dtype,
        )
        for img in images
    ]

    lengths = [f.seq.shape[0] for f in feats]
    D = feats[0].seq.shape[1]
    T_max = max(lengths)

    batch_seq = None
    batch_attn = None

    if pad_to_max:
        batch_seq = torch.zeros((len(feats), T_max, D), dtype=out_dtype, device=device)
        batch_attn = torch.zeros((len(feats), T_max), dtype=torch.long, device=device)
        for i, f in enumerate(feats):
            t = f.seq.shape[0]
            batch_seq[i, :t] = f.seq
            batch_attn[i, :t] = 1

    return {
        "features": feats,            # list[PageFeatures]
        "batch_seq": batch_seq,       # (B, T_max, D) or None
        "batch_attn": batch_attn,     # (B, T_max) or None
        "lengths": lengths,           # list[int]
        "D": D,
        "specials": feats[0].specials,
    }


