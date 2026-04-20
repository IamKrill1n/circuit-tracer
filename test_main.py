import os
from summarization.prune import prune_graph_pipeline
from api import generate_graph, get_feature, save_subgraph
from pathlib import Path
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.create_graph_files import create_graph_files  
from circuit_tracer.graph import Graph, prune_graph, compute_graph_scores
from transformers import AutoModelForCausalLM, AutoTokenizer
import shap
from scipy.special import softmax

model_name = os.getenv("HF_MODEL_NAME", "google/gemma-2-2b-it")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype="auto")
# if torch.cuda.is_available():
#     model = model.cuda()
# # set model decoder to true
# model.config.is_decoder = True
# # ensure task_specific_params is initialized (avoid NoneType assignment error)
# if model.config.task_specific_params is None:
#     model.config.task_specific_params = {}
# # set text-generation params under task_specific_params
# model.config.task_specific_params["text-generation"] = {
#     "do_sample": False, # set to False for deterministic output
#     "max_new_tokens": 1, # set to 1 for single-token generation
#     "temperature": 0, # set to 0 for deterministic output
#     # "no_repeat_ngram_siz e": 2, 
# }
# explainer = shap.Explainer(model, tokenizer)
# prompts = ['Fact: The capital of the state containing Dallas is']

# shap_values = explainer(prompts)
# print(shap_values.values.squeeze())
# token_weights = softmax(shap_values.values.squeeze())
# print(token_weights)

json_path = "demos/temp_graph_files/austin.json"
source_set = 'clt-hp' #'clt-hp' # gemmascope-transcoder-16k
# token_weights = [0.00198786, 0.03153391, 0.00083086, 0.01473883, 0.22338926, 0.00649094,
#  0.00222269, 0.01996207, 0.0052309, 0.67869559, 0.01491708]
token_weights = [0, 0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0]
kept_ids, pruned_adj, node_inf, node_rel, attr, metadata = prune_graph_pipeline(
    json_path=json_path,
    logit_weights='target',
    token_weights=token_weights,
    node_threshold=0.7,
    edge_threshold=0.9,
    keep_all_tokens_and_logits=False,
)

# print(len(kept_ids))
for i, node_id in enumerate(kept_ids):
    print(node_id, attr[node_id].get("clerp", ""), node_inf[i].item(), node_rel[i].item())

# print(attr['1_89326_9'])

# LLM-based classify using OpenAI
feature_types = classify_features_with_llm(kept_ids, attr, metadata)
print(feature_types)