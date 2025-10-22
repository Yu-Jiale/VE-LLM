# deepseek_ocr_runner/__init__.py
from .encoder import encode_batch
from .decoder import decode_generate
from .features import encode_page_to_sequence, encode_pages_to_batch
from . import _utils as utils

__all__ = [
    "encode_batch",             # 原 batch encoder（文本混排）
    "decode_generate",          # 手写解码
    "encode_page_to_sequence",  # 新：单页 → 连续视觉嵌入序列
    "encode_pages_to_batch",    # 新：批量包装
    "utils",
]