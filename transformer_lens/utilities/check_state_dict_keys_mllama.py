from datetime import date

import torch
from transformers import AutoProcessor

from transformer_lens import MllamaForConditionalGeneration

# Get today's date
date_string: str = date.today().strftime("%d %b %Y")

# Model and processor
model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

model_state_dict = model.state_dict()
model_state_dict_keys = list(model_state_dict.keys())

# Display the first 100 keys for an overview
print(model_state_dict_keys)
print(dir(model))