import math
import torch
from PIL import ImageOps
from typing import Dict, List, Tuple

from deepseek_ocr_runner.third_party.deepseek_ocr_mod.modeling_deepseekocr import (
    format_messages, text_encode, load_image, dynamic_preprocess, BasicImageTransform
)

from ._utils import log, pack_batch_tensors, DEBUG

# ===== 内部：与 DeepseekOCRModel.forward 的视觉拼接保持一致 =====
@torch.no_grad()
def _build_vision_tokens_for_one_sample(model, images_crop, image_ori,
                                        width_crop_num: int, height_crop_num: int):
    sam_model = model.model.sam_model
    vision_model = model.model.vision_model
    projector = model.model.projector
    image_newline = model.model.image_newline
    view_sep     = model.model.view_seperator

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        patches = images_crop
        image_g = image_ori

        if (patches.numel() > 0) and (patches.abs().sum().item() != 0):
            local_f1 = sam_model(patches)
            local_f2 = vision_model(patches, local_f1)
            local = torch.cat((local_f2[:, 1:], local_f1.flatten(2).permute(0, 2, 1)), dim=-1)
            local = projector(local)

            global_f1 = sam_model(image_g)
            global_f2 = vision_model(image_g, global_f1)
            glob = torch.cat((global_f2[:, 1:], global_f1.flatten(2).permute(0, 2, 1)), dim=-1)
            glob = projector(glob)

            _, hw, dim = glob.shape
            h = w = int(hw ** 0.5)
            _, hw2, dim2 = local.shape
            h2 = w2 = int(hw2 ** 0.5)

            glob = glob.view(h, w, dim)
            glob = torch.cat([glob, image_newline[None, None, :].expand(h, 1, dim)], dim=1)
            glob = glob.view(-1, dim)

            local = local.view(height_crop_num, width_crop_num, h2, w2, dim2)\
                         .permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, dim2)
            local = torch.cat([local, image_newline[None, None, :].expand(height_crop_num*h2, 1, dim2)], dim=1)
            local = local.view(-1, dim2)

            vision_tokens = torch.cat([local, glob, view_sep[None, :]], dim=0)
        else:
            global_f1 = sam_model(image_g)
            global_f2 = vision_model(image_g, global_f1)
            glob = torch.cat((global_f2[:, 1:], global_f1.flatten(2).permute(0, 2, 1)), dim=-1)
            glob = projector(glob)
            _, hw, dim = glob.shape
            h = w = int(hw ** 0.5)
            glob = glob.view(h, w, dim)
            glob = torch.cat([glob, image_newline[None, None, :].expand(h, 1, dim)], dim=1)
            glob = glob.view(-1, dim)
            vision_tokens = torch.cat([glob, view_sep[None, :]], dim=0)

    return vision_tokens


@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    conversations: List[List[Dict]],     # 每样本 1 张图（对齐原实现）
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    device: str = "cuda",
    image_token: str = "<image>",
    image_token_id: int = 128815,        # 如需兼容别的 tokenizer，可暴露为可配置参数
    patch_size: int = 16,
    downsample_ratio: int = 4,
    bos_id: int = 0,
) -> Dict:
    """
    返回:
      {
        "inputs_embeds": (B, T_pad, hidden),
        "attention_mask": (B, T_pad),
        "prefill_lens": List[int],
        "decode_images_stub": List[Tuple[Tensor, Tensor]],  # [(zeros_crop, zeros_ori)] * B
        "decode_images_spatial_stub": Tensor,               # (B, 2) 全 0
        "base_size": int,
        "image_size": int,
      }
    """
    model.eval()
    B = len(conversations)

    inputs_embeds_list  = []
    attention_mask_list = []
    prefill_lens        = []
    decode_images_stub  = []
    decode_images_spatial = []

    for b in range(B):
        conv = conversations[b]
        prompt = format_messages(conversations=conv, sft_format='plain', system_prompt='')

        pil_images = []
        for msg in conv:
            if "images" in msg:
                for p in msg["images"]:
                    img = load_image(p).convert("RGB")
                    pil_images.append(img)

        has_image = len(pil_images) > 0
        if has_image:
            pil_images = [pil_images[0]]
            if image_token not in prompt:
                prompt = image_token + prompt
            text_splits = prompt.split(image_token)
            if len(text_splits) > 2:
                text_splits = [text_splits[0], ''.join(text_splits[1:])]
        else:
            text_splits = [prompt]

        token_ids = []
        img_mask  = []

        # 文本前半段
        ids = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
        token_ids += ids
        img_mask  += [False] * len(ids)

        images_list, images_crop_list = [], []
        width_crop_num = height_crop_num = 1

        if has_image:
            image = pil_images[0]
            image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

            if crop_mode:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                    images_crop_raw = []
                else:
                    images_crop_raw, crop_ratio = dynamic_preprocess(image)

                global_view = ImageOps.pad(
                    image, (base_size, base_size),
                    color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5))
                )
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                width_crop_num, height_crop_num = crop_ratio
                if width_crop_num > 1 or height_crop_num > 1:
                    for im in images_crop_raw:
                        images_crop_list.append(image_transform(im).to(torch.bfloat16))

                num_queries      = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                        num_queries * height_crop_num
                    )
                token_ids += tokenized_image
                img_mask  += [True] * len(tokenized_image)
            else:
                raise NotImplementedError("建议 crop_mode=True 与 infer() 保持一致。")

        # 文本后半段
        tail = text_splits[-1] if len(text_splits) > 1 else ""
        ids = text_encode(tokenizer, tail, bos=False, eos=False)
        token_ids += ids
        img_mask  += [False] * len(ids)

        # BOS
        token_ids = [bos_id] + token_ids
        img_mask  = [False] + img_mask

        # to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        img_mask  = torch.tensor(img_mask,  dtype=torch.bool, device=device)
        prefill_len = input_ids.shape[0]

        # 文本嵌入
        inputs_embeds = model.get_input_embeddings()(input_ids.unsqueeze(0))

        # 视觉注入
        if has_image:
            if images_list:
                images_ori  = torch.stack(images_list, dim=0).to(device)
                if images_crop_list:
                    images_crop = torch.stack(images_crop_list, dim=0).to(device)
                else:
                    images_crop = torch.zeros((1, 3, base_size, base_size),
                                              dtype=images_ori.dtype, device=device)

                vision_tokens = _build_vision_tokens_for_one_sample(
                    model,
                    images_crop=images_crop,
                    image_ori  =images_ori,
                    width_crop_num=width_crop_num,
                    height_crop_num=height_crop_num,
                )
                true_cnt = img_mask.sum().item()
                log(f"[ENC] sample {b}: img_mask True={true_cnt}, vision_tokens={tuple(vision_tokens.shape)}")
                assert true_cnt == vision_tokens.shape[0], \
                    f"[ENC] sample {b}: mask True({true_cnt}) != vision_tokens({vision_tokens.shape[0]})"

                inputs_embeds[0].masked_scatter_(img_mask.unsqueeze(-1), vision_tokens.to(inputs_embeds.dtype))

        attention_mask = torch.ones((1, prefill_len), dtype=torch.long, device=device)

        inputs_embeds_list.append(inputs_embeds[0])
        attention_mask_list.append(attention_mask[0])
        prefill_lens.append(prefill_len)

        # 为 decode 阶段准备占位图像（强制视觉分支短路）
        zeros_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16, device=device)
        zeros_ori  = torch.zeros((1, 3, image_size, image_size), dtype=torch.bfloat16, device=device)
        decode_images_stub.append((zeros_crop, zeros_ori))
        decode_images_spatial.append(torch.tensor([0, 0], dtype=torch.long, device=device))

        log(f"[ENC] sample {b}: inputs_embeds={tuple(inputs_embeds.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}, prefill_len={prefill_len}")

    hidden = inputs_embeds_list[0].shape[-1]
    inputs_embeds_pad, _  = pack_batch_tensors(inputs_embeds_list, pad_value=0.0, dtype=inputs_embeds_list[0].dtype)
    attention_mask_pad, _ = pack_batch_tensors(attention_mask_list, pad_value=0,   dtype=attention_mask_list[0].dtype)
    decode_images_spatial_stub = torch.stack(decode_images_spatial, dim=0)  # (B, 2)

    log(f"[ENC] batch: inputs_embeds_pad={tuple(inputs_embeds_pad.shape)}, "
        f"attention_mask_pad={tuple(attention_mask_pad.shape)}")

    return {
        "inputs_embeds": inputs_embeds_pad,
        "attention_mask": attention_mask_pad,
        "prefill_lens": prefill_lens,
        "decode_images_stub": decode_images_stub,
        "decode_images_spatial_stub": decode_images_spatial_stub,
        "base_size": base_size,
        "image_size": image_size,
        "hidden_size": hidden,
    }