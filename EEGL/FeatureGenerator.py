import gc
from SUBTREEGL.GuidanceTree import *
from SubgraphSolver import subgraphIsoTest
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch
import sys


def generateFeatures(G, maximalFrequentSubgraphsByLabel, data, subIsoTest, k, treeParams=None):
    trainMask = data.train_mask
    testMask = data.test_mask
    valMask = data.val_mask

    trainLabels = torch.full_like(data.y, -1)
    trainLabels[trainMask] = data.y[trainMask]

    nodes = list(G.nodes)
    numNodes = len(nodes)
    numLabels = len(maximalFrequentSubgraphsByLabel)

    patterns, patternRadii = buildPatternsList(maximalFrequentSubgraphsByLabel)
    X = np.zeros((numNodes, len(patterns)))

    transactionGraphs = prepareTransactionGraphs(nodes, G, patternRadii)
    trainTransactionGraphs = [transactionGraphs[i] for i, mask in enumerate(trainMask) if mask == 1]
    X = calculateMetrics(patterns, trainTransactionGraphs, trainLabels, subIsoTest, X, treeParams)
    patterns.sort(key=lambda p: p.graph['metrics']['F1-Score'], reverse=True)

    # Select top k patterns
    topKPatterns, topKPatternIndices = selectTopK(patterns, k=k)

    # Calculate feature vector entries for all test and validation nodes
    testValTransactionGraphs = [graph for i, graph in enumerate(trainTransactionGraphs) if testMask[i] == 1 or valMask[i] == 1]
    X = calcFeatureVectorEntries(topKPatterns, testValTransactionGraphs, subIsoTest, X, treeParams)

    # Create new feature vectors
    featureVectors = X[:, topKPatternIndices]
    return featureVectors, topKPatterns


def buildPatternsList(maximalFrequentSubgraphsByLabel):
    patterns = []
    patternDict = {}
    key = 0
    metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'F1-Score': None}
    patternRadiiSet = set()  # To store unique pattern sizes

    for label, subgraphs in maximalFrequentSubgraphsByLabel.items():
        for pattern in subgraphs:

            if 'key' in pattern.graph:  # for miners being able to identify subgraphs across labels, e.g: ftm
                key = pattern.graph.get('key')
                if key in patternDict:
                    patternDict[key][1].append(label)
                    patternDict[key][2].append(metrics)
                else:
                    patternDict[key] = (pattern, [label], [metrics])

            else:  # for miners that cannot, e.g: gaston
                key += 1
                patternDict[key] = (pattern, [label], [metrics])

    # patterns: holds patterns by indexing each and adding the nx graph and labels
    for key, (pattern, labels, metrics) in patternDict.items():
        pattern.graph['index'] = len(patterns)
        pattern.graph['label'] = labels
        pattern.graph['metrics'] = metrics
        pattern.graph['radius'] = nx.eccentricity(G=pattern, v=pattern.graph['root'])
        patternRadiiSet.add(pattern.graph['radius'])
        patterns.append(pattern)

    patternRadii = list(patternRadiiSet)
    return patterns, patternRadii


def prepareTransactionGraphs(nodes, G, patternSizes):
    transactionGraphs = Parallel(n_jobs=-1)(
        delayed(prepareTransactionGraphForNode)(node, G, patternSizes) for node in
        nodes
    )
    return transactionGraphs

def prepareTransactionGraphForNode(node, G, patternRadii):
    transactionGraph = {}
    for size in patternRadii:
        tg = nx.ego_graph(G, node, radius=size, undirected=True)
        tg.graph['root'] = node
        transactionGraph[size] = tg

    return transactionGraph


def calculateMetrics(patterns, transactionGraphs, trainLabels, subIsoTest, X, treeParams):
    print(f'\n-- Calculating F1-Scores for {len(patterns)} Patterns, checking {len(transactionGraphs)} Train Nodes for each --')
    print(f'Parallel: Total of {len(patterns) * len(transactionGraphs)} jobs to be processed.')

    # Initialize metricsAccumulated for all patterns and labels
    metricsAccumulated = {p.graph['index']: [{'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for _ in p.graph['label']] for p in patterns}

    # Process each pattern-node pair in parallel
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(processPatternTrainNodePair)
        (pattern, transactionGraph, trainLabels, subIsoTest, treeParams)
        for pattern in patterns for transactionGraph in transactionGraphs
    )

    # Accumulate the results in metricsAccumulated
    for pattern, nodeIndex, labelMetrics, patternMatches in results:
        if patternMatches:
            X[nodeIndex][pattern.graph['index']] = 1.0

        for i, metrics in enumerate(labelMetrics):
            # Accumulate metrics
            metricsAccumulated[pattern.graph['index']][i]['TP'] += metrics['TP']
            metricsAccumulated[pattern.graph['index']][i]['FP'] += metrics['FP']
            metricsAccumulated[pattern.graph['index']][i]['TN'] += metrics['TN']
            metricsAccumulated[pattern.graph['index']][i]['FN'] += metrics['FN']

    # Apply accumulated metrics to patterns and update X matrix
    for pattern in patterns:
        bestLabelIndex = -1
        bestF1Score = -1

        for i, metrics in enumerate(metricsAccumulated[pattern.graph['index']]):
            # Calculate F1-Score for the current label
            f1Score = calculateF1Score(metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN'])
            metricsAccumulated[pattern.graph['index']][i]['F1-Score'] = f1Score

            # Find the label with the best F1-Score
            if f1Score > bestF1Score:
                bestF1Score = f1Score
                bestLabelIndex = i

        # Set the best metrics and F1-Score to the pattern
        bestMetrics = metricsAccumulated[pattern.graph['index']][bestLabelIndex]
        pattern.graph['label'] = pattern.graph['label'][bestLabelIndex]
        pattern.graph['metrics'] = bestMetrics

    return X



def processPatternTrainNodePair(pattern, transactionGraph, trainLabels, subIsoTest, treeParams=None):

    radius = pattern.graph['radius']
    tg = transactionGraph[radius]
    if treeParams is not None:
        k = treeParams['k']
    patternMatches = subgraphIsoTest(targetGraph=tg, root=tg.graph['root'], patternGraph=pattern, method=subIsoTest, k=k)

    labelMetrics = []
    for label in pattern.graph['label']:
        metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        trueLabel = trainLabels[tg.graph['root']]
        labelMatches = (trueLabel == label)

        if patternMatches:
            if labelMatches:
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        else:
            if labelMatches:
                metrics['FN'] += 1
            else:
                metrics['TN'] += 1

        labelMetrics.append(metrics)

    return pattern, tg.graph['root'], labelMetrics, patternMatches

def calculateF1Score(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1Score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1Score

def selectTopK(patterns, k=10):
    # Check if the number of patterns is smaller than k
    if len(patterns) <= k:
        print(f"\n-- Fewer than {k} patterns available, selecting all {len(patterns)} patterns --")
        topKpatterns = patterns
        topKindices = [pattern.graph['index'] for pattern in patterns]
        for i, pattern in enumerate(patterns, start=1):
            print(f"Pattern {i}: Index:{pattern.graph['index']}       Label {pattern.graph['label']}      F1-Score: {pattern.graph['metrics']['F1-Score']}")
        return topKpatterns, topKindices

    print(f"\n-- Top {k} patterns --")
    topKpatterns = []
    topKindices = []
    labelSet = set()
    indiceSet = set()

    while len(topKpatterns) < k:
        # Update the foundLabels to reflect only those patterns not yet in topKpatterns
        foundLabels = set(pattern.graph['label'] for pattern in patterns if pattern.graph['index'] not in indiceSet)
        numLabels = len(foundLabels)

        for pattern in patterns:
            if len(topKpatterns) >= k:
                break

            if pattern.graph['index'] in indiceSet or pattern.graph['label'] in labelSet:
                continue

            pattern.graph['size'] = len(pattern.edges())
            topKpatterns.append(pattern)
            topKindices.append(pattern.graph['index'])
            labelSet.add(pattern.graph['label'])
            indiceSet.add(pattern.graph['index'])
            print(f"Pattern {len(topKpatterns)}: "
                  f"Label {pattern.graph['label']}      "
                  f"Size {pattern.graph['size']}      "
                  f"F1-Score: {pattern.graph['metrics']['F1-Score']}")

            if len(labelSet) >= numLabels:
                labelSet.clear()
                break

    return topKpatterns, topKindices



def calcFeatureVectorEntries(topKPatterns, transactionGraphs, subIsoTest, X, treeParams=None):
    print(f"\n--- Calculating Feature Vector Entries for Test and Validation Nodes ---")
    print(f'Parallel: Total of {len(topKPatterns) * len(transactionGraphs)} jobs to be processed.')
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(processPatternTestValNodePair)
        (pattern, transactionGraph, subIsoTest, treeParams)
        for pattern in topKPatterns for transactionGraph in transactionGraphs
    )
    for pattern, nodeIndex, matchesNode in results:
        if matchesNode:
            X[nodeIndex][pattern.graph['index']] = 1.0

    return X

def processPatternTestValNodePair(pattern, transactionGraph, subIsoTest, treeParams=None):
    patternGraph = pattern
    matchesNode = False
    radius = pattern.graph['radius']
    tg = transactionGraph[radius]
    k = treeParams['k']
    if subgraphIsoTest(targetGraph=tg, root=tg.graph['root'], patternGraph=patternGraph, method=subIsoTest, k=k):
        matchesNode = True

    return pattern, tg.graph['root'], matchesNode


