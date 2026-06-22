from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import torch
from tqdm import tqdm
import sys
import math
from typing import List
import matplotlib.pyplot as plt
rootAttr = '__root__'

def generateExplanations(G, model, data, explainer='gnnExplainer', epochs=200, numTopEdges=25):
    print('\n\n-- Generating Explanations --')
    nodes = list(range(G.number_of_nodes()))

    if explainer == 'gnnExplainer':
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs'
            )
        )

        explanationGraphs = []
        explanationRoots = []

        for nodeID in tqdm(nodes, file=sys.stdout):

            # For each node an explanation is generated and processed to be the component of the root  node
            explanation = explainer(data.x, data.edge_index, index=nodeID)
            explanationGraph, explanationRoot = rootedSubgraph(G, data.edge_index, nodeID, explanation, numTopEdges=numTopEdges)

            # The resulting explanation graphs and roots are stored in a list
            explanationGraphs.append(explanationGraph)
            explanationRoots.append(explanationRoot)

    else:
        print("Currently no explainers available besides GNNExplainer.")

    return explanationGraphs, explanationRoots


def rootedSubgraph(G, edgeIndex, nodeID, explanation, numTopEdges=25):
    edgeMask = explanation.edge_mask

    # Only the "numTopEdges" edges with the highest importance are kept and finally stored as (u,v) tuples
    if edgeMask.nonzero(as_tuple=False).size(0) > numTopEdges:
        edgeMask[torch.argsort(edgeMask, descending=True)[numTopEdges:]] = 0.
    edges = edgeIndex[:, edgeMask.nonzero(as_tuple=False).view(-1).tolist()].t().tolist()
    edges = [tuple(edge) for edge in edges]

    # The subgraph component containing the root node of the explanation is created
    explanationGraph = G.edge_subgraph(edges).copy()
    if not nodeID in explanationGraph.nodes:
        explanationGraph.add_node(nodeID)
    explanationGraph = explanationGraph.subgraph(nx.node_connected_component(explanationGraph, nodeID)).copy()

    # Relabel nodes to 0,...,n-1 (for Gaston) but keep original nodeIDs as attributes and get new nodeID for root
    explanationGraph = nx.convert_node_labels_to_integers(explanationGraph, label_attribute='nodeID')
    explanationRoot = [id for id, data in explanationGraph.nodes(data=True) if data['nodeID'] == nodeID][0]
    explanationGraph.graph['root'] = explanationRoot

    return explanationGraph, explanationRoot

def labelGroupedExplanations(explanationGraphs, explanationRoots, modelPredictions, data):

    labelGroupedExplanations = {label: ([], []) for label in set(data.y.tolist())}

    for label, explanationGraph, explanationRoot in zip(modelPredictions.tolist(), explanationGraphs, explanationRoots):
        if len(explanationGraph.edges()) > 0: #If explanation is only one node it is useless
            labelGroupedExplanations[label][0].append(explanationGraph)
            labelGroupedExplanations[label][1].append(explanationRoot)

    return labelGroupedExplanations

def drawExplanations(explanationGraphs):

    def getColours(G, root):
        colors = []
        for n in G.nodes:
            if n == root:
                colors.append('tab:green')
            else:
                colors.append('tab:red')
        return colors


    numGraphs = len(explanationGraphs)
    nCols = min(3, numGraphs)
    nRows = int(math.ceil(numGraphs / nCols))

    fig, ax = plt.subplots(nRows, nCols, sharex=True, sharey=True, figsize=(5 * nCols, 5 * nRows))
    ax = ax.flatten()

    for i, G in enumerate(explanationGraphs):
        rootNode = G.graph['root']
        nx.draw(G, node_color=getColours(G, rootNode), ax=ax[i], pos=nx.spring_layout(G), node_size=100)
        ax[i].set_axis_off()
        ax[i].set_title(f'{i + 1}')

    plt.show()


def printExplanationInfo(explanationsByLabel):
    for label, (explanationGraphs, explanationRoots) in explanationsByLabel.items():

        edges = 0
        graphs = len(explanationGraphs)

        if graphs == 0:
            print(f'Label: {label} has no suitable Explanations!')
            continue

        for explanationGraph in explanationGraphs:
            edges += len(explanationGraph.edges())

        print(f'Label: {label}, Number of Explanations: {graphs}, Average Explanation Size: {edges / graphs}')