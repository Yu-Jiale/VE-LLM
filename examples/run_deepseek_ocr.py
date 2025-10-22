from transformers import AutoModel, AutoTokenizer
import os, torch
import torch
from deepseek_ocr_runner import encode_batch, decode_generate, utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "deepseek-ai/DeepSeek-OCR"
cache_dir = "./models/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
).cuda().eval()

utils.DEBUG = True

conversations = [
    [  # 样本 1
        {"role": "<|User|>", "content": "<image>\nConvert everything to Markdown.", "images": ["./test_document.png"]},
        {"role": "<|Assistant|>", "content": ""},
    ],
    [  # 样本 2
        {"role": "<|User|>", "content": "<image>\nParse the figure.", "images": ["./test_document.png"]},
        {"role": "<|Assistant|>", "content": ""},
    ],
]

enc = encode_batch(
    model, tokenizer, conversations,
    base_size=1024, image_size=640, crop_mode=True, device="cuda"
)

texts = decode_generate(model, tokenizer, enc, max_new_tokens=2048, temperature=0.0)
for i, t in enumerate(texts):
    print(f"=== Sample {i} ===")
    print(t)
    