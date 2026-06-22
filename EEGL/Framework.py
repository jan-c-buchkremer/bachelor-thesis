from Explainer import generateExplanations, labelGroupedExplanations, drawExplanations, printExplanationInfo
from PatternExtractor import extractFrequentSubgraphs, printFrequentSubgraphs
from sklearn.model_selection import ParameterGrid
from FeatureGenerator import generateFeatures
from contextlib import redirect_stdout
from GNN import trainingStep, evaluate
from tqdm import tqdm
import pickle as pkl
import numpy as np
import cProfile
import pstats
import torch
import time
import sys
import os


class EEGL:
    def __init__(self, G, trainIndex=None, valIndex=None, testIndex=None, explainer='gnnExplainer', subgraphMiner='gaston', subIsoTest='networkX', numFeatures=10, initFeatures='random', frequency=0.8, gnnParameters='standardParameters', treeParams=None):
        self.G = G
        self.trainIndex = trainIndex
        self.valIndex = valIndex
        self.testIndex = testIndex
        self.explainer = explainer
        self.subgraphMiner = subgraphMiner
        self.subIsoTest = subIsoTest
        self.numFeatures = numFeatures
        self.initFeatures = initFeatures
        self.frequency = frequency
        self.gnnParameters = gnnParameters
        self.treeParams = treeParams

        # Applying Features to Nodes
        if initFeatures == 'vanilla':
            self.initFeatures = torch.ones((len(self.G), numFeatures), dtype=torch.float)
        elif initFeatures == 'random':
            self.initFeatures = torch.rand(len(self.G), numFeatures, dtype=torch.float)
        elif initFeatures == 'optimal':
            self.initFeatures = torch.tensor([numFeatures * [G.nodes[i]['y']] for i in G.nodes], dtype=torch.float)
        else:
            raise ValueError(f'init_features must be either "vanilla" or "random", but was {initFeatures}')

        self.numFeatures = self.initFeatures.shape[1]
        self.features = self.initFeatures

        applyFeatures(self.G, np.array(self.features))

        # Initialization of other attributes
        self.explanationGraphs = None
        self.explanationRoots = None
        self.model = None
        self.data = None
        self.modelPredictions = None
        self.modelAccuracies = None
        self.features = None
        self.topK = None
        self.topKpatterns = None
        self.maximalFrequentSubgraphsByLabel = None
        self.times = None

        if self.gnnParameters == 'standardParameters':
            self.gnnParameters = {
                'hiddenChannels': 16,
                'dropout': 0.5,
                'learningRate': 0.01,
                'weightDecay': 5e-4,
                'epochs': 200
            }


    def runEEGLIteration(self):

        self.times = {}

        # '''

        classificationTime = time.time()
        print("GNN Parameters", self.gnnParameters)

        # Prediction and Evaluation
        self.model, self.data = trainingStep(self.G,
                                             hiddenChannels=self.gnnParameters['hiddenChannels'],
                                             dropout=self.gnnParameters['dropout'],
                                             learningRate=self.gnnParameters['learningRate'],
                                             weightDecay=self.gnnParameters['weightDecay'],
                                             epochs=self.gnnParameters['epochs'])

        self.modelAccuracies, self.modelPredictions = evaluate(self.model, self.data)
        print(f'Training Accuracy: {self.modelAccuracies['train']}, Validation Accuracy: {self.modelAccuracies['val']}, Test Accuracy: {self.modelAccuracies['test']}')

        classificationTime = time.time() - classificationTime
        self.times['gnn'] = classificationTime

        # Explanation Generation
        explanationTime = time.time()
        self.explanationGraphs, self.explanationRoots = generateExplanations(G=self.G, model=self.model, data=self.data, explainer=self.explainer)

        explanationTime = time.time() - explanationTime
        self.times['explainer'] = explanationTime
        print("Explanation Time: ", explanationTime)

        print('\n\n-- Labels and their Explanation Graphs Edge Count --')
        #drawExplanations(self.explanationGraphs[:min(len(self.explanationGraphs), 12)])
        explanationsByLabel = labelGroupedExplanations(self.explanationGraphs, self.explanationRoots, self.modelPredictions, self.data)
        printExplanationInfo(explanationsByLabel)

        # Creating Label Support Dict
        labelSupportDict = {}
        for label, (explanationGraphs, explanationRoots) in explanationsByLabel.items():
            labelSupportDict[label] = np.floor(len(explanationGraphs) * self.frequency).astype(int)


        # Pattern Extractor
        extractorTime = time.time()
        print(f'\n\n-- Extracting Explanation Subgraph Patterns with {self.subgraphMiner} --')
        self.maximalFrequentSubgraphsByLabel = extractFrequentSubgraphs(explanationsByLabel, labelSupportDict, self.subgraphMiner, self.treeParams)
        printFrequentSubgraphs(self.maximalFrequentSubgraphsByLabel, detailed=False) # set detailed=True for print of all subgraphs

        extractorTime = time.time() - extractorTime
        self.times['extractor'] = extractorTime
        print("Extraction Time: ", extractorTime)

        # Top-k Pattern Extraction
        annotationTime = time.time()
        print(f'\n\n-- Generating Features with {self.subIsoTest} --')
        self.features, self.topKpatterns = generateFeatures(self.G, self.maximalFrequentSubgraphsByLabel, self.data, self.subIsoTest, self.numFeatures, self.treeParams)

        # Applying Features
        applyFeatures(self.G, self.features)
        print(f"New Feature Vectors of {len(self.features)} Features applied to {len(self.G.nodes())} nodes.")

        annotationTime = time.time() - annotationTime
        self.times['annotator'] = annotationTime
        print("Feature Annotation Time: ", annotationTime)

        print(f"\n Iteration Times:")
        print(self.times)

        results = {}
        results['times'] = self.times
        results['accuracies'] = self.modelAccuracies
        results['patterns'] = self.topKpatterns
        return results


    def gridSearch(self, hyperparameters='standardHyperparameters'):

        if hyperparameters == 'standardHyperparameters':
            hyperparameters = {
                'epochs': [200, 100],
                'hiddenChannels': [16, 32],
                'weightDecay': [5e-4, 1e-4],
                'learningRate': [0.1, 0.01, 0.001],
                'dropout': [0.5, 0.2, 0.0]
            }

        gridsearchTime = time.time()
        hyperparameterCombinations = list(ParameterGrid(hyperparameters))

        results = {'parameters': [], 'trainingAccuracy': [], 'validationAccuracy': []}
        defaultParameters = {'hiddenChannels': 16, 'dropout': 0.5, 'learningRate': 0.01, 'weightDecay': 5e-4, 'epochs': self.gnnParameters['epochs']}
        for param in defaultParameters.keys():
            if param not in hyperparameters.keys():
                hyperparameters[param] = [defaultParameters[param]]

        print(f'\n\n-- Performing grid search on {len(hyperparameterCombinations)} hyperparameter combinations --')

        # Iterate over hyperparameter combinations
        for hyperparameters in tqdm(hyperparameterCombinations, file=sys.stdout):
            hiddenChannels = hyperparameters['hiddenChannels']
            dropout = hyperparameters['dropout']
            learningRate = hyperparameters['learningRate']
            weightDecay = hyperparameters['weightDecay']
            epochs = hyperparameters['epochs']

            # Train GNN while suppressing its console output
            with redirect_stdout(open(os.devnull, 'w')):
                model, data = trainingStep(self.G,
                                           learningRate=learningRate,
                                           weightDecay=weightDecay,
                                           epochs=epochs,
                                           dropout=dropout,
                                           hiddenChannels=hiddenChannels)

            # Evaluate the model
            self.modelAccuracies, self.modelPredictions = evaluate(model, data)
            trainingAccuracy = self.modelAccuracies['train']
            validationAccuracy = self.modelAccuracies['val']
            results['parameters'].append(hyperparameters)
            results['trainingAccuracy'].append(trainingAccuracy)
            results['validationAccuracy'].append(validationAccuracy)

        # Set GNN parameters to the best hyperparameters
        bestHyperparameters = np.argmax(results['validationAccuracy'])
        self.gnnParameters = results['parameters'][bestHyperparameters]
        print(f'Best Hyperparameters: {self.gnnParameters} with validation accuracy: {results["validationAccuracy"][bestHyperparameters]}\n')

        gridsearchTime = time.time() - gridsearchTime
        results['time'] = gridsearchTime

        return results

    def resetFeatures(self):

        self.features = self.initFeatures

    def saveState(self, filename):
        try:
            with open(filename, 'wb') as f:
                pkl.dump(self, f)
                print("Run saved.")
        except (OSError, IOError) as e:
            print(f"Error saving file: {e}")
        except pkl.PickleError as e:
            print(f"Pickling error: {e}")

    @staticmethod
    def loadState(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)


def applyFeatures(G, features): # Half ass function, never used

    if len(G) != len(features):
        raise ValueError('Graph Nodes and Feature Vector of unequal length!')

    for i in range(len(G)):
        G.nodes[i]['x'] = torch.tensor(features[i].flatten(), dtype=torch.float)


def loadGraphFromPickle(graphName):

    # Define the directory where pickle files are stored
    path = 'data/graphs/'
    ending = '.pkl'

    # Map graphName to the corresponding file name
    graphFiles = {
        'G180': 'G_180',
        'm1': 'm1',
        'm1p': 'm1p',
        'm1pp': 'm1pp',
        'm2': 'm2',
        'm2p': 'm2p',
        'm2pp': 'm2pp',
        'C20': 'C20',
        'C24': 'C24',
        'C26': 'C26',
        'C60': 'C60',
        'C70': 'C70'

    }

    # Check if the provided graph_name is valid
    if graphName not in graphFiles:
        raise ValueError(f'Invalid graph name: {graphName}')

    # Construct the file path for the pickle file
    filePath = path + graphFiles[graphName] + ending

    print(f'Loading Graph from {filePath}\n')

    # Load the graph from the pickle file
    with open(filePath, 'rb') as f:
        G = pkl.load(f)

    return G


if __name__ == "__main__":
    treeParams = {}
    treeParams['k'] = 10
    treeParams['patternSize'] = 8
    G = loadGraphFromPickle('G180')
    eegl = EEGL(G=G, subgraphMiner='ftm', subIsoTest='subtree', treeParams=treeParams)
    eegl.gridSearch()
    eegl.runEEGLIteration()
