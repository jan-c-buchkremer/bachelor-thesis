from Framework import *
from sklearn.model_selection import StratifiedKFold
import pickle as pkl
import numpy as np
import copy
import time
import re
import sys

def kFoldCV(G, eeglParams, fileName, nSplits=10, eeglIterations=2, storePickle=True, gridSearchHyperParams=None, treeParams=None, str='WTF'):
    """
    Performs k-fold cross-validation on the provided graph using the EEGL framework.

    Args:
        G: The input graph.
        eeglParams: A dictionary of parameters to initialize the EEGL framework.
        fileName: The base name for storing results.
        nSplits: Number of splits for cross-validation.
        eeglIterations: Number of iterations for the workflow.
        storePickle: Whether to store the results as pickle files.
        gridSearchHyperParams: Hyperparameters for grid search.

    Returns:
        None
    """
    runDict = {}
    cv = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)
    Xfake = np.zeros((G.number_of_nodes(), eeglParams['numFeatures']))  # Placeholder for X in cv.split
    y = np.array([G.nodes[node]['y'] for node in G.nodes])  # Target labels
    times = []
    if storePickle: print('Storing results with pickle.')
    for i, (trainIndex, testIndex) in enumerate(cv.split(Xfake, y)):
        print(f'\n\n\n\n Fold {i + 1}/{nSplits}:\n\n')
        kFoldTime = time.time()

        trainValTestData = {
            'trainIndex': trainIndex,
            'valIndex': testIndex,
            'testIndex': testIndex
        }

        eegl = EEGL(
            G=G,
            trainIndex=trainValTestData['trainIndex'],
            valIndex=trainValTestData['valIndex'],
            testIndex=trainValTestData['testIndex'],
            explainer=eeglParams['explainer'],
            subgraphMiner=eeglParams['subgraphMiner'],
            subIsoTest=eeglParams['subIsoTest'],
            numFeatures=eeglParams['numFeatures'],
            initFeatures=eeglParams['initFeatures'],
            frequency=eeglParams['frequency'],
            gnnParameters=eeglParams['gnnParameters'],
            treeParams=treeParams
        )

        foldResultDict = runEEGL(eegl,
                                 fileName + f'fold{i}_',
                                 eeglIterations=eeglIterations,
                                 storePickle=storePickle,
                                gridSearchHyperParams=gridSearchHyperParams)

        runDict[f'Fold{i}'] = foldResultDict

        kFoldTime = time.time() - kFoldTime
        times.append(kFoldTime)
        predictedTimeLeft = (nSplits - i - 1) * np.mean(times)
        print(f'Fold {i + 1} took {time.strftime("%H:%M:%S", time.gmtime(kFoldTime))}')
        print(f'Predicted time left: {time.strftime("%H:%M:%S", time.gmtime(predictedTimeLeft))}')

    return runDict



def runEEGL(eegl, fileName, eeglIterations=2, storePickle=True, gridSearchHyperParams=None):
    """
    Executes the iterative workflow for EEGL, including training, evaluation, and feature extraction.

    Args:
        eegl: An instance of the EEGL class.
        fileName: The base name for storing results.
        eeglIterations: Number of iterations for the workflow.
        storePickle: Whether to store the results as pickle files.
        gridSearchHyperParams: Hyperparameters for grid search.

    Returns:
        None
    """
    results = {}
    eeglStates = {}

    if gridSearchHyperParams is not None:
        print("\n\n\n Performing grid search on hyperparameters.")
        gridSearchResults = eegl.gridSearch(hyperparameters=gridSearchHyperParams)

    for i in range(eeglIterations):
        print(f'\n\n\n##### EEGL ITERATION {i + 1}/{eeglIterations} #####')
        IterationResults = eegl.runEEGLIteration()


        results[i] = IterationResults
        eeglStates[i] = copy.deepcopy(eegl)

        if storePickle:
            with open(fileName + 'results.pkl', 'wb') as f:
                pkl.dump(results, f)

            with open(fileName + 'eegl.pkl', 'wb') as f:
                pkl.dump(eeglStates, f)

    return results

def runTask(graphName, patternSize, k=None, subgraphMiner='ftm', savePickle=False):

    treeParams = {}
    treeParams['k'] = k
    treeParams['patternSize'] = patternSize

    initFeatures = 'random'

    if initFeatures not in ['random', 'vanilla', 'optimal']:
        raise ValueError(f'arg2 must be "vanilla", "random", or "optimal", got {initFeatures}')

    G = loadGraphFromPickle(graphName)

    # Hyperparameter choice based on graph graphName
    if graphName == 'G180':
        numFeatures = 56
        frequency = 0.8
        nSplits = 5
        eeglIterations = 5
    elif re.match(r'^m', graphName, re.IGNORECASE):
        numFeatures = 10
        frequency = 0.8
        nSplits = 5
        eeglIterations = 5
    elif re.match(r'^C', graphName, re.IGNORECASE):
        numFeatures = 18
        frequency = 0.2
        nSplits = 5
        eeglIterations = 5
    else:
        raise ValueError(f'No match for graphName {graphName}.')

    eeglParams = {
        'explainer': 'gnnExplainer',
        'subgraphMiner': subgraphMiner,
        'subIsoTest': 'glasgow',
        'numFeatures': numFeatures,
        'initFeatures': initFeatures,
        'frequency': frequency,
        'gnnParameters': {
            'hiddenChannels': 16,
            'dropout': 0.5,
            'learningRate': 0.01,
            'weightDecay': 5e-4,
            'epochs': 200
        }
    }

    gsHyperParams = {
        'epochs': [200, 100],
        'hiddenChannels': [16, 32],
        'weightDecay': [5e-4, 1e-4],
        'learningRate': [0.1, 0.01, 0.001],
        'dropout': [0.5, 0.2, 0.0]
    }

    str = 'WTF'
    if subgraphMiner == 'ftm':
        str = 'FTM'
    elif subgraphMiner == 'gaston':
        str = 'GASTON'

    fileName = f'data/eeglRuns/{graphName}_{str}_S{patternSize}_F{frequency:.2f}_'

    print(f'Running {nSplits}-fold cross-validation on hyperparameters: ' f'initFeatures: {initFeatures}, 'f'numFeatures: {numFeatures}, ' f'frequency: {frequency}')
    runDict = kFoldCV(G, eeglParams=eeglParams, fileName=fileName, nSplits=nSplits, eeglIterations=eeglIterations, gridSearchHyperParams=gsHyperParams, treeParams=treeParams, storePickle=savePickle, str=str)

    runDictName = f'{str}_{graphName}_S{patternSize}.pkl'
    with open(f'data/eeglRuns/runDicts/{runDictName}', 'wb') as f:
        pkl.dump(runDict, f)

if __name__ == '__main__':

    pass
