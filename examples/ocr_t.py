from pathlib import Path
import os
import torch
from transformers import AutoModel, AutoTokenizer

CACHE_DIR = Path(r"E:\model_cache")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR)

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=str(CACHE_DIR),
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    model_name,
    use_safetensors=True,
    trust_remote_code=True,
    cache_dir=str(CACHE_DIR),
    local_files_only=True
)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to bullet point markdown. "
image_file = 'input.png'
output_path = 'output.png'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path = output_path,
    base_size = 1024,
    image_size = 640,
    crop_mode=True,
    save_results = True,
    test_compress = True
)
