# We are running evaluation for our methods.


## 2.Modify eval_prune.py to
### Evaluate pruning:
- input: graph path at demos/temp_graph_files/clt-hp, shap values at demos/shap_values.json
- run pruning algorithm with 3 normalization choices (softmax, relu_l1, entmax15) of token weights, keeping edge thresholds at 0.95 and sweep node thresholds with step 0.1 and report the number of nodes and combined graph scores into a csv. You have to write a function to normalize shap values for custom choice of masker_keep_prefix
- save pruned graphs and eval results csv (don't save token_weights)