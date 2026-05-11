# Fork of safety-research/circuit-tracer 
This is a fork of `safety-research/circuit-tracer`, extended with a **summarization pipeline** that automates summarizing the attribution graph. The core library computes attribution graphs (circuits) for transformer language models using MLP transcoders.
## Recap
The summarization pipeline consists of 2 stage:
1. Prunning: prune the attribution graph down to a subgraph containing important nodes and edges. 
- Key idea: introduce **relevance** and use along with **influence** (from original clt paper) to align the subgraph with human rationale better (features activating on useless tokens are pruned).
2. Clustering: cluster functionally similar feature nodes together, further simplify the summarization graph and aid steering.
- Key idea: use spectral clustering on a weighted edge profile similarity matrix, penalize clustering features of long layer span, and force DAG post process.

## Evaluation Methodology

To rigorously assess the quality, interpretability, and structural soundness of our proposed supernode graph generation method, we evaluate it against both trivial and standard graph-theoretic baselines. A high-quality supernode graph for mechanistic interpretability must group functionally similar features together without collapsing sequential causal chains into cyclic or self-amplifying blobs. 

### 1. Baselines

To contextualize the performance of our method, we compare it against three baselines representing naive or standard approaches to feature grouping:

*   **Random Assignment (Null Baseline):** Individual features are randomly assigned to $K$ supernodes. This establishes the absolute lower bound for functional cohesion and causal sequence preservation.
*   **By-Layer / Layer-Collapse (Structural Baseline):** All features within a single transformer layer are grouped into a single supernode. Because information strictly flows forward through layers, this baseline naturally forms a perfect Directed Acyclic Graph (DAG) and ensures high causal independence. However, it represents an extreme loss of functional granularity, as it lumps orthogonal mathematical and semantic features together.
*   **Standard Louvain Community Detection (Graph-Theoretic Baseline):** We apply the Louvain modularity optimization algorithm directly to the un-directed raw feature-to-feature adjacency matrix. While Louvain is a standard for maximizing community cohesion, it is agnostic to the sequential forward-pass of a transformer. It tends to cluster heavily connected sequential nodes together, thereby violating causal independence and generating massive structural cycles.

### 2. Scoring Metrics

Because our baselines produce fundamentally different graph structures (e.g., DAG vs. cyclic), we evaluate all methods using an agnostic evaluation framework on a fixed ground-truth similarity space. We utilize two primary metrics:

*   **Silhouette Score (Role Similarity):** To measure functional cohesion, we compute the Silhouette Score using a precomputed weighted cosine similarity matrix of the features' incoming and outgoing edge profiles. A high score indicates that features within a supernode play highly similar mechanistic roles. 
    *   **Raw Silhouette ($S_{raw}$):** Reported in its standard range of $[-1, 1]$, where $1$ denotes perfect clustering and negative values denote misclassification.
    *   **Normalized Silhouette ($S_{norm}$):** For composite scoring and ease of comparison, we apply an affine transformation to map the score to a strictly positive range $[0, 1]$ via $S_{norm} = \frac{S_{raw} + 1}{2}$.
*   **Internal Independence Score ($S_{ind}$):** To penalize the collapse of sequential causal paths, we calculate the proportion of a supernode’s edge weight that interacts with the *external* graph rather than amplifying itself *internally*. For a set of supernodes $C$, let $W_{in}$ be the sum of entirely internal edge weights, and $W_{out}$ be the sum of external edge weights. We define the score over the domain $[0, 1]$ as:
    $$S_{ind} = \frac{1}{|C|} \sum_{c \in C} \frac{W_{out}^{(c)}}{W_{in}^{(c)} + W_{out}^{(c)}}$$
    A score approaching $1.0$ indicates that supernodes represent distinct functional steps in a causal chain (ideal), whereas a lower score indicates that sequential causal loops have been improperly collapsed into a single supernode.

### 3. Experimental Settings and Hyperparameters

To ensure a fair comparison across algorithms, we strictly control the resolution of the generated graphs. 

*   **Cluster Count Constraint:** For the Random, Louvain, and Proposed methods, the target number of supernodes $K$ is fixed to two structural resolutions: $K = \lfloor n/2 \rfloor$ and $K = \lfloor n/3 \rfloor$, where $n$ is the total number of individual features. The By-Layer baseline is naturally fixed to $K = L$, where $L$ is the number of layers in the network.
*   **Hyperparameter Sweep (Proposed Method):** To identify the optimal configuration for causal extraction, we perform a grid search over the hyperparameters defining our algorithmic constraints:
    *   **Similarity Mode:** We evaluate both *node-based* similarity (using raw feature activations/encodings) and *edge-based* similarity (using the causal edge profiles).
    *   **Mean Method:** For aggregating cross-layer cluster representations, we ablate *arithmetic*, *harmonic*, and *geometric* means to observe the impact of dampening extreme outlier weights.
    *   **Decay Rate ($\gamma$):** We sweep the temporal decay parameter governing layer-span distance penalties from $0.0$ to $1.0$ with a step size of $0.1$ ($\gamma \in \{0.0, 0.1, 0.2, \dots, 1.0\}$). This allows us to quantify the exact trade-off between localized DAG enforcement (high decay) and global conceptual grouping (low decay).

### 4. Ablation study
To validate our pruning methodology, we measure the effect of ablating features in our subgraph on the output logit. 
We measure the KL divergence between the model with ablation and the clean forward pass. The ablation we use is constraint patching on every feature. (to be implemented)

We experiment with steering with our summarization graph on multihop reasoning dataset. (to be implemented)

### 5. Neuronpedia
Neuronpedia api is used to generate attribution graphs, upload supernode graph (neuronpedia calls it subgraph), and steering. The api doc is here https://www.neuronpedia.org/api-doc


