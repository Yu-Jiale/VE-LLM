import torch
from typing import Dict, List
from ._utils import log, DEBUG

@torch.no_grad()
def decode_generate(
    model,
    tokenizer,
    enc_outputs: Dict,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    no_repeat_ngram_size: int = 20,  # 预留参数（未启用）
    top_k: int = 0,
    top_p: float = 1.0,
) -> List[str]:
    """
    手写解码，绕过 HF generate 在 inputs_embeds 首步的 RoPE 0-length 问题。
    步骤：
      1) prefill: 整段 inputs_embeds 前向一次，拿 past_key_values 与最后一步 logits；
      2) loop: 逐 token 续写（每步 input_ids 长度=1），不再传 images。
    enc_outputs 来自 encoder.encode_batch()。
    """
    inputs_embeds_all  = enc_outputs["inputs_embeds"]          # (B, T_pad, H)
    attention_mask_all = enc_outputs["attention_mask"]         # (B, T_pad)
    prefill_lens       = enc_outputs["prefill_lens"]           # List[int]

    images_stub        = enc_outputs["decode_images_stub"]     # List[(zero_crop, zero_ori)]
    spatial_stub_all   = enc_outputs["decode_images_spatial_stub"]  # (B, 2)

    B, T_pad, H = inputs_embeds_all.shape
    outs = []

    log(f"[PREFILL] per-sample manual decode: B={B}, T_pad={T_pad}, H={H}")

    for i in range(B):
        prefill = prefill_lens[i]
        # 切到有效长度
        inputs_embeds  = inputs_embeds_all[i, :prefill].unsqueeze(0).contiguous()
        attention_mask = attention_mask_all[i, :prefill].unsqueeze(0).contiguous()

        # 提供与长度匹配的占位 input_ids（不影响语义）
        dummy_input_ids = torch.zeros((1, prefill), dtype=torch.long, device=inputs_embeds.device)

        # 占位图像（sum==0 使得视觉注入分支整体跳过）
        zeros_crop, zeros_ori = images_stub[i]
        spatial_stub = spatial_stub_all[i].unsqueeze(0)  # (1,2)

        log(f"[PREFILL] sample {i}: prefill_len={prefill}, "
            f"inputs_embeds={tuple(inputs_embeds.shape)}, attn_mask={tuple(attention_mask.shape)}")

        # ---------- 1) PREFILL ----------
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                images=[(zeros_crop, zeros_ori)],
                images_seq_mask=None,
                images_spatial_crop=spatial_stub,
            )

        logits = out.logits
        past_kv = out.past_key_values

        # 首 token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated = [next_token.item()]
        log(f"[PREFILL] sample {i}: first token id={generated[-1]}")

        # ---------- 2) LOOP ----------
        cur_len = 1
        eos_id = getattr(tokenizer, "eos_token_id", None)
        attn_mask_step = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        while cur_len < max_new_tokens:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=next_token.view(1, 1),
                    attention_mask=attn_mask_step,
                    use_cache=True,
                    past_key_values=past_kv,
                    return_dict=True,
                )

            logits = out.logits
            past_kv = out.past_key_values

            if top_k > 0 or top_p < 1.0 or temperature > 0.0:
                probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
                if top_k > 0:
                    topk = torch.topk(probs, top_k)
                    mask = torch.full_like(probs, float('-inf'))
                    mask.scatter_(1, topk.indices, torch.log(topk.values + 1e-12))
                    logit_step = mask
                else:
                    logit_step = torch.log(probs + 1e-12)
                next_token = torch.argmax(logit_step, dim=-1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

            token_id = next_token.item()
            generated.append(token_id)
            cur_len += 1

            if DEBUG and (cur_len % 50 == 0):
                log(f"[STEP] sample {i}: generated {cur_len} tokens, last={token_id}")

            if eos_id is not None and token_id == eos_id:
                log(f"[STEP] sample {i}: hit EOS at step={cur_len}")
                break

        # 解码新增
        new_ids = torch.tensor(generated, dtype=torch.long, device=inputs_embeds.device).view(1, -1)
        text = tokenizer.decode(new_ids[0], skip_special_tokens=False)
        stop_str = '<｜end▁of▁sentence｜>'
        if text.endswith(stop_str):
            text = text[:-len(stop_str)]
        outs.append(text.strip())

        log(f"[DONE] sample {i}: total_new={len(generated)} tokens, out_len={len(outs[-1])} chars")

    return outs