from SUBTREEGL.FrequentSubtreeMining import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from SUBTREEGL.Visualisation import visualiseTree
import Seba.OuterplanarPatternMiner as Opm
from SubgraphSolver import subgraphIsoTest
from joblib import Parallel, delayed
from collections import defaultdict
from Gaston import runGaston
from tqdm.auto import tqdm
import numpy as np
import time


def printFrequentSubgraphs(frequentSubgraphsByLabel, detailed=False):
    print(f'\n\n-- Maximal Frequent Subgraphs by Label --')

    for label, subgraphs in frequentSubgraphsByLabel.items():
        if detailed: print(f"\nLabel: {label}")


        edges = 0
        if subgraphs is None:
            continue

        for i, subgraph in enumerate(subgraphs):
            edges += len(subgraph.edges())
            if detailed: print(f"{i}. {subgraph.edges()}")
        if len(subgraphs) == 0:
            print(f"Label {label}   Total number of frequent subgraphs: {len(subgraphs)}")
        else:
            print(f"Label {label}   Total number of frequent subgraphs: {len(subgraphs)}    Average frequent subgraph size: {edges/len(subgraphs)}")

def transformDict(maximalFrequentSubgraphsByLabel):
    transformedDict = {}

    for label, innerDict in maximalFrequentSubgraphsByLabel.items():
        transformedDict[label] = list(innerDict.values())

    return transformedDict

def maximalTestGaston(patternGraph, targetGraph):

    if (patternGraph.graph['isMaximal'] is False  # If patternGraph is not maximal, no need for consideration
    or patternGraph.graph['support'] < targetGraph.graph['support']  # if patternGraph less frequent too
    or patternGraph == targetGraph): # if patternGraph == targetGraph as well
        return False

    # Returns true if it finds isomorphism
    if subgraphIsoTest(targetGraph=targetGraph, patternGraph=patternGraph, method='glasgow'):
        return True # returns true if it finds isomorphism

    return False


def maximalFrequentSubgraphsGaston(frequentSubgraphsByLabel):
    """
    Extracts maximal frequent subgraphs from a dictionary of frequent subgraphs.

    Args:
        frequentSubgraphsByLabel (dict): A dictionary where keys are labels and values are lists of frequent subgraphs.

    Returns:
        dict: A dictionary where keys are labels and values are lists of maximal frequent subgraphs.
    """
    print('\n\n-- Extracting Maximal Frequent Subgraphs --')
    maximalFrequentSubgraphsByLabel = {}

    for label, subgraphs in frequentSubgraphsByLabel.items():

        # Sort subgraphs in order of their support, set isMaximal=True initially
        subgraphsSorted = sorted(subgraphs, key=lambda subgraph: subgraph.graph['support'])
        for subgraph in subgraphsSorted: subgraph.graph['isMaximal'] = True

        with tqdm(total=len(subgraphsSorted), desc=f"Processing Subgraphs for Label {label}") as pbar:
            for targetGraph in subgraphsSorted:

                # We don't check non-maximal graphs as target graphs
                if targetGraph.graph['isMaximal'] is False:
                    pbar.update(1)
                    continue

                results = Parallel(n_jobs=-1)(
                    delayed(maximalTestGaston)(patternGraph=patternGraph, targetGraph=targetGraph)
                    for patternGraph in subgraphsSorted)

                # Update the isMaximal attribute based on the results
                excluded = 0
                for patternGraph, result in zip(subgraphsSorted, results):
                    if result:
                        patternGraph.graph['isMaximal'] = False
                        excluded += 1

                pbar.update(1)

        # Take only those remaining maximal
        maximalFrequentSubgraphsByLabel[label] = [subgraph for subgraph in subgraphsSorted
                                                  if subgraph.graph.get('isMaximal') == True]
        pbar.write(f"{len(maximalFrequentSubgraphsByLabel[label])} maximal subgraphs extracted out of {len(frequentSubgraphsByLabel[label])}.")
        pbar.close()

    return maximalFrequentSubgraphsByLabel


def extractFrequentSubgraphs(explanationsByLabel, labelSupportDict, subgraphMiner='gaston', treeParams=None):
    patternsByLabel = {}
    patternSize = treeParams['patternSize']

    if subgraphMiner == 'gaston':
        frequentSubgraphsByLabel = runGaston(explanationsByLabel, labelSupportDict,  patternSize)
        patternsByLabel = maximalFrequentSubgraphsGaston(frequentSubgraphsByLabel)

    elif subgraphMiner == 'opm':
        # Use joblib for parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(processLabel)(label, explanationsByLabel, labelSupportDict)
            for label in explanationsByLabel.keys())

        for label, patterns in results:
            if patterns is not None:
                patternsByLabel[label] = patterns

    elif subgraphMiner == 'ftm':
        if treeParams == None:
            print('\n\n!! Expected Tree Params !!')
        else:
            patternsByLabel = extractFrequentSubtrees(explanationsByLabel, labelSupportDict, treeParams, totalCores=24)

    else:
        print("Error: Unsupported subgraph miner!")
        return 0
    return patternsByLabel


def processLabel(label, explanationsByLabel, labelSupportDict):
    explanationGraphs, explanationRoots = explanationsByLabel[label]
    support = min(len(explanationGraphs), labelSupportDict[label])
    if support == 0:  return label, None

    opmTime = time.time()
    maximalFrequentSubgraph = Opm.frequent_outerplanar_graphs(explanationGraphs, explanationRoots, support)
    opmTime = time.time() - opmTime
    print(f"Extracted Frequent Outerplanar Subgraphs for Label: {label} with Support at least: {support} in {opmTime} seconds.")

    return label, maximalFrequentSubgraph
