import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_url
print(hf_hub_url("gpt2", "config.json"))
# 应输出：https://hf-mirror.com/gpt2/resolve/main/config.json

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("✅ 成功加载模型！")