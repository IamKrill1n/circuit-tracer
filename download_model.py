# import torch
# from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

# transcoder, config = load_transcoder_from_hub("mntss/clt-gemma-2-2b-2.5M")

import os

from circuit_tracer.replacement_model import ReplacementModel
import torch

# Avoid flaky Xet transfer path in some network/SSL environments.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

model_name = "qwen3-4b"
transcoder_name = "mwhanna/qwen3-4b-transcoders"
backend = 'transformerlens'  # change to 'nnsight' for the nnsight backend!
model = ReplacementModel.from_pretrained(
    model_name, transcoder_name, dtype=torch.bfloat16, backend=backend
)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "google/gemma-2-2b" # google/gemma-scope-2-4b-it crosscoder/layer_9_17_22_29_width_65k_l0_medium
# transcoder_name = "mntss/clt-gemma-2-2b-2.5M"
# backend = 'transformerlens'  # change to 'nnsight' for the nnsight backend!
# model = ReplacementModel.from_pretrained(
#     model_name, transcoder_name, dtype=torch.bfloat16, backend=backend
# )
# model_name = 'google/gemma-3-4b-it' # google/gemma-scope-2-4b-it crosscoder/layer_9_17_22_29_width_65k_l0_medium
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# device = torch.device("cuda")
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# backend = 'transformerlens'  # change to 'nnsight' for the nnsight backend!
# backend = 'nnsight'
# model = ReplacementModel.from_pretrained("google/gemma-2-2b", "mntss/clt-gemma-2-2b-2.5M", dtype=torch.bfloat16, backend=backend)