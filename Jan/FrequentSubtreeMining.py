from Jan.Visualisation import *
from Jan.Glasgow import glasgowIsoTest
import networkx as nx
import time
import pickle as pkl


from joblib import Parallel, delayed

def extractFrequentSubtrees(explanationsByLabel, labelSupportDict, treeParams, totalCores=24):

    patternsByLabel = {}
    tasks = []  # To store all delayed tasks
    labelsWithMoreTasksThanCores = []  # Labels that have more explanationGraphs than totalCores
    coresByLabel = {}  # To store cores assigned to each label for sorting

    # Filter out labels with no explanationGraphs and initialize their pattern to None
    for label in explanationsByLabel:
        if len(explanationsByLabel[label][0]) == 0:
            patternsByLabel[label] = []

    # Remove labels with no tasks
    labelsToProcess = [label for label in explanationsByLabel if len(explanationsByLabel[label][0]) > 0]

    # Find the labels that have more explanationGraphs than cores
    for label in labelsToProcess:
        transactionGraphs = explanationsByLabel[label][0]

        if len(transactionGraphs) > totalCores:
            labelsWithMoreTasksThanCores.append(label)

    if labelsWithMoreTasksThanCores:
        coresForLargeLabels = totalCores // len(labelsWithMoreTasksThanCores)  # Floor division to distribute evenly
    else:
        coresForLargeLabels = 0

    # Calculate the number of cores for each label and store in coresByLabel
    for label in labelsToProcess:
        transactionGraphs = explanationsByLabel[label][0]

        if len(transactionGraphs) > totalCores:
            coresForLabel = coresForLargeLabels
        else:
            coresForLabel = len(transactionGraphs) - labelSupportDict[label] + 1

        coresForLabel = min(coresForLabel, totalCores)
        coresByLabel[label] = coresForLabel  # Store cores for sorting

    # Sort labels by cores in descending order
    sortedLabels = sorted(coresByLabel.keys(), key=lambda label: coresByLabel[label], reverse=True)

    # Append tasks in the order of labels sorted by cores
    for label in sortedLabels:
        tasks.append(delayed(processLabel)(label, explanationsByLabel, labelSupportDict, treeParams, coresByLabel[label]))

    # Run all tasks in parallel
    results = Parallel(n_jobs=totalCores)(tasks)

    # Update patternsByLabel with the results
    for label, patterns in results:
        patternsByLabel[label] = patterns

    return patternsByLabel


def processLabel(label, explanationsByLabel, labelSupportDict, treeParams, cores):
    explanationGraphs, explanationRoots = explanationsByLabel[label]
    support = min(len(explanationGraphs), labelSupportDict[label])
    if support == 0:
        support = 1

    ftmTime = time.time()
    maximalFrequentSubgraphs = frequentSubtreeMining(explanationGraphs, explanationRoots, support, label, treeParams, cores)
    ftmTime = time.time() - ftmTime
    print(f"Extracted {len(maximalFrequentSubgraphs)} Patterns for label {label} using {cores} cores with support at least: {support} in {ftmTime} seconds.")

    return label, maximalFrequentSubgraphs

def frequentSubtreeMining(transactionGraphs, roots, t, label, treeParams, cores):

    k = treeParams['k']
    patternSize = treeParams['patternSize']
    subtreesLevelGraph = getFTMdata(nodes=patternSize)
    frequentSubtreeDict = {}
    visited = set()

    for node in list(subtreesLevelGraph.nodes):

        if node in visited: continue
        pattern = subtreesLevelGraph.nodes[node].get('pattern')
        if pattern is None: continue

        pattern = preparePattern(pattern, node, transactionGraphs, subtreesLevelGraph)
        remainingGraphs = len(transactionGraphs)
        supportCount = 0

        for batchStart in range(0, len(transactionGraphs), cores):
            batchEnd = min(batchStart + cores, len(transactionGraphs))
            batchResult = Parallel(n_jobs=min(cores, len(transactionGraphs)))(
                delayed(checkSupport)(i, transactionGraphs[i], roots[i], pattern, k)
                for i in range(batchStart, batchEnd))

            for (isoTestResult, i, characteristics) in batchResult:
                supportCount += isoTestResult

                if characteristics is not None:
                    pattern.graph['characteristics'][i] = pattern.graph['characteristics'][i].union(characteristics)

            # Early stopping condition
            remainingGraphs -= (batchEnd - batchStart)
            if supportCount + remainingGraphs < t:
                break

        if supportCount >= t: # Saving frequent patterns
            frequentSubtreeDict[node] = pattern
        else: # Pruning infrequent patterns
            pruned = {node} | nx.descendants(subtreesLevelGraph, node)
            subtreesLevelGraph.remove_nodes_from(pruned)
            visited.update(pruned)

    # Before returning, clear the characteristics from each pattern
    for pattern in frequentSubtreeDict.values():
        pattern.graph.pop('characteristics', None)

    #visualiseTree(subtreesLevelGraph, title=f'Frequent Subtrees for Label {label}')
    maximalFrequentSubtrees = [pattern for node, pattern in frequentSubtreeDict.items()
                               if subtreesLevelGraph.out_degree(node) == 0]

    return maximalFrequentSubtrees



def preparePattern(pattern, node, transactionGraphs, subtreesLevelGraph):
    pattern.graph['root'] = 0
    pattern.graph['key'] = node
    pattern.graph['characteristics'] = [set() for _ in range(len(transactionGraphs))]
    for predecessor in subtreesLevelGraph.predecessors(node):
        precedingPattern = subtreesLevelGraph.nodes[predecessor].get('pattern')
        if precedingPattern is None: continue
        for i in range(len(transactionGraphs)): pattern.graph['characteristics'][i] \
            = (pattern.graph['characteristics'][i].union(precedingPattern.graph['characteristics'][i]))

    return pattern

def checkSupport(i, transactionGraph, root, pattern, k):
    characteristics = pattern.graph['characteristics'][i]
    if transactionGraph.number_of_nodes() <= 1:
        return 0
    isoTestResult = glasgowIsoTest(targetGraph=transactionGraph, patternGraph=pattern, root=root)
    if isoTestResult is True:
        return 1, i, None
    else:
        return 0, i, None

def getFTMdata(nodes):
    with open(f'../EEGL/data/levelGraphs/rootedTreesLevelGraph{nodes}.pkl', 'rb') as file:
        set = pkl.load(file)
        subtreesLevelGraph = set.pop()
    return subtreesLevelGraph

if __name__ == '__main__':

    levelGraph = getFTMdata(12)
    print(levelGraph.edges())
    for node in levelGraph.nodes():
        pattern = levelGraph.nodes[node].get('pattern')
        print(pattern.edges())
    print(f"Nodes: {len(levelGraph.nodes())}")

    #visualiseTree(levelGraph)



