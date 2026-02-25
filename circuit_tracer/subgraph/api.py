import os
import requests
from typing import Tuple, List, Optional
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "https://www.neuronpedia.org"


def _get_api_key() -> str:
    """Retrieve API key from environment."""
    return str(os.getenv("NEURONPEDIA_API_KEY", ""))


def get_feature(modelId: str, layer: str, index: int) -> Tuple[int, str]:
    """Fetch a feature from the Neuronpedia API.
    
    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        layer: Layer identifier (e.g., "10-clt-hp")
        index: Feature index
        
    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/feature/{modelId}/{layer}/{index}"
    headers = {"x-api-key": _get_api_key()}
    
    resp = requests.get(url, headers=headers, timeout=30)
    return resp.status_code, resp.text


def generate_autointerp(
    modelId: str,
    layer: str,
    index: int,
    explanationModelName: str = "gemini-2.5-flash",
    explanationType: str = "oai_token-act-pair"
) -> Tuple[int, str]:
    """Generate auto-interpretation for a feature.
    
    Args:
        modelId: Model identifier
        layer: Layer identifier
        index: Feature index
        explanationModelName: Model to use for explanation generation
        explanationType: Type of explanation to generate
        
    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/explanation/generate"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _get_api_key()
    }
    payload = {
        "modelId": modelId,
        "layer": layer,
        "index": index,
        "explanationType": explanationType,
        "explanationModelName": explanationModelName
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.status_code, resp.text


def generate_graph(
    modelId: str,
    prompt: str,
    slug: str,
    sourceSetName: str,
    desiredLogitProb: float = 0.95,
    edgeThreshold: float = 0.85,
    maxFeatureNodes: int = 5000,
    maxNLogits: int = 10,
    nodeThreshold: float = 0.8
) -> Tuple[int, str]:
    """Generate an attribution graph via the Neuronpedia API.
    
    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        prompt: Input text prompt
        slug: Unique identifier for the graph
        sourceSetName: Source set name (e.g., "clt-hp")
        desiredLogitProb: Desired logit probability threshold
        edgeThreshold: Edge weight threshold
        maxFeatureNodes: Maximum number of feature nodes
        maxNLogits: Maximum number of logits
        nodeThreshold: Node importance threshold
        
    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/graph/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "modelId": modelId,
        "prompt": prompt,
        "slug": slug,
        "sourceSetName": sourceSetName,
        "desiredLogitProb": desiredLogitProb,
        "edgeThreshold": edgeThreshold,
        "maxFeatureNodes": maxFeatureNodes,
        "maxNLogits": maxNLogits,
        "nodeThreshold": nodeThreshold
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    return resp.status_code, resp.text


def save_subgraph(
    modelId: str,
    slug: str,
    displayName: str,
    pinnedIds: List[str],
    supernodes: Optional[List[List[str]]] = None,
    clerps: Optional[List[str]] = None,
    pruningThreshold: float = 0.8,
    densityThreshold: float = 0.99,
    overwriteId: str = "",
) -> Tuple[int, str]:
    """Save a subgraph to Neuronpedia.

    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        slug: Slug of the parent graph
        displayName: Human-readable name for the subgraph
        pinnedIds: List of node IDs to pin in the subgraph
        supernodes: List of supernode groups, each is [label, node_id, ...]
        clerps: List of custom clerp overrides
        pruningThreshold: Pruning threshold used to produce this subgraph
        densityThreshold: Density threshold used to produce this subgraph
        overwriteId: If non-empty, overwrite an existing subgraph with this ID

    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/graph/subgraph/save"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _get_api_key(),
    }
    payload = {
        "modelId": modelId,
        "slug": slug,
        "displayName": displayName,
        "pinnedIds": pinnedIds,
        "supernodes": supernodes if supernodes is not None else [],
        "clerps": clerps if clerps is not None else [],
        "pruningThreshold": pruningThreshold,
        "densityThreshold": densityThreshold,
        "overwriteId": overwriteId,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.status_code, resp.text


if __name__ == "__main__":
    # Example: get feature
    # status, data = get_feature("gemma-2-2b", "10-clt-hp", 512)
    # print(f"Status: {status}")
    # print(data)
    
    # Example: generate graph (uncomment to use)
    # status, data = generate_graph(
    #     modelId="gemma-2-2b",
    #     prompt="If dog is to bark, then cat is to",
    #     slug="meow",
    #     sourceSetName="clt-hp",
    #     desiredLogitProb=0.9,
    #     edgeThreshold=0.8,
    #     maxFeatureNodes=5000,
    #     maxNLogits=10,
    #     nodeThreshold=0.5
    # )
    # print(f"Status: {status}")
    # print(data)
    # print(_get_api_key())
    status, data = save_subgraph(
        modelId="gemma-2-2b",
        slug="my doggo graph",
        displayName="test save subgraph",
        pinnedIds=["2_15681_2", "E_2_0", "4_14735_2"],
        supernodes=[["supernode", "4_14735_2", "19_9180_3"]],
        pruningThreshold=0.8,
        densityThreshold=0.99,
    )
    print(f"Status: {status}")
    print(data)