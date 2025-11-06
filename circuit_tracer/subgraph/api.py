import http.client
import os
import json
conn = http.client.HTTPSConnection("www.neuronpedia.org")


def get_feature(modelId : str, layer : str, index : int):
    headers = { 'x-api-key': str(os.getenv("NEURONPEDIA_API_KEY")) }
    conn.request("GET", f"/api/feature/{modelId}/{layer}/{index}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    return res.status, data.decode("utf-8")

def generate_graph(
    modelId : str, 
    prompt : str, 
    slug: str, 
    sourceSetName: str,
    desiredLogitProb: float = 0.95, 
    edgeThreshold: float = 0.85, 
    maxFeatureNodes: int = 5000,
    maxNLogits: int = 10,
    nodeThreshold: float = 0.8
):
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

    headers = { 'Content-Type': "application/json" }

    conn.request("POST", "/api/graph/generate", json.dumps(payload), headers)

    res = conn.getresponse()
    data = res.read()

    return res.status, data.decode("utf-8")

if __name__ == '__main__':
    status, data = get_feature("gemma-2-2b", "10-clt-hp", 512)
    print(data)
    # status, data = generate_graph(modelId="gemma-2-2b", prompt="If dog is to bark, then cat is to", slug="meow", sourceSetName="clt-hp",
    #                 desiredLogitProb=0.9, edgeThreshold=0.8, maxFeatureNodes=5000, maxNLogits=10, nodeThreshold=0.5)
    # print(status)
    # print(data)