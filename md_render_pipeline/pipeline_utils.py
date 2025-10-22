import torch
from typing import Tuple


@torch.no_grad()
def make_zero_image_stubs(
    batch_size: int,
    base_size: int,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[list, torch.Tensor]:
    """
    Construct the zero-image placeholders expected by DeepSeek-OCR so that the
    visual branch short-circuits during decoding.
    """
    zeros_crop = torch.zeros((1, 3, base_size, base_size), dtype=dtype, device=device)
    zeros_ori = torch.zeros((1, 3, image_size, image_size), dtype=dtype, device=device)

    images = [(zeros_crop, zeros_ori) for _ in range(batch_size)]
    images_spatial_crop = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
    return images, images_spatial_crop


@torch.no_grad()
def text_to_embeds(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """
    Tokenize text without adding special tokens and project into the model's
    embedding space.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return torch.zeros((0, model.config.hidden_size), dtype=model.dtype, device=device)
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return model.get_input_embeddings()(ids_tensor)[0]


@torch.no_grad()
def build_inputs_with_image_seq(
    model,
    tokenizer,
    image_seq: torch.Tensor,
    text_before: str,
    text_after: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Concatenate [BOS] + text_before + image_seq + text_after as inputs_embeds and
    return the tensor, attention mask, and effective prefill length.
    """
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    bos = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    bos_embed = model.get_input_embeddings()(bos)[0]

    emb_before = text_to_embeds(model, tokenizer, text_before, device)
    emb_after = text_to_embeds(model, tokenizer, text_after, device)

    tgt_dtype = image_seq.dtype
    seq = torch.cat(
        [
            bos_embed.to(tgt_dtype),
            emb_before.to(tgt_dtype),
            image_seq,
            emb_after.to(tgt_dtype),
        ],
        dim=0,
    )
    attn = torch.ones((1, seq.shape[0]), dtype=torch.long, device=device)
    return seq.unsqueeze(0), attn, seq.shape[0]

