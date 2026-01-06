import torch
from circuit_tracer import ReplacementModel, attribute

model_name = 'meta-llama/Llama-3.2-1B'
transcoder_name = "mntss/clt-llama-3.2-1b-524k" #"gemma" mntss/clt-gemma-2-2b-426k
model = ReplacementModel.from_pretrained(model_name, transcoder_name, dtype=torch.bfloat16)