import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import pprint
import networkx as nx
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from SUBTREEGL.Visualisation import *
import matplotlib.cm as cm


def loadPickle(filePath):
    try:
        with open(filePath, 'rb') as f:
            return pkl.load(f)
    except (EOFError, pkl.UnpicklingError) as e:
        print(f"Error loading {filePath}: {e}")
        return None


def buildNestedDict(rootFolder, firstLevel, secondLevel):
    def add_to_dict(d, path_parts, file_name=None, file_data=None):
        for part in path_parts:
            if part not in d:
                d[part] = {}
            d = d[part]
        if file_name:
            d[file_name] = file_data

    nestedDict = {}

    for folderPath, _, files in os.walk(rootFolder):
        # Split folder path into individual folders
        folderKeys = folderPath.split(os.sep)[1:]  # Skip the root folder
        if not folderKeys:  # Skip if the path only consists of the root folder
            continue

        currentLevel = nestedDict
        add_to_dict(currentLevel, folderKeys)

        for file in files:
            if file.endswith('.pkl'):
                filePath = os.path.join(folderPath, file)
                file_data = loadPickle(filePath)
                add_to_dict(nestedDict, folderKeys, file, file_data)

    nestedDict = nestedDict['eeglRuns']
    resultDict = {}
    eeglStatesDict = {}

    for value in firstLevel:
        oldDict = nestedDict.get(value, None)
        firstValueDict = {}

        for secondValue in secondLevel:

            oldKDict = oldDict.get(secondValue, None)
            secondValueDict = {}

            # Loop through the secondValueDict and filter out only the result subdictionaries
            for key, subDict in oldKDict.items():
                if key.endswith('_results.pkl'):
                    foldNumer = key.split('_fold')[-1].split('_')[0]
                    foldKey = f'Fold{foldNumer}'
                    secondValueDict[foldKey] = subDict

                if key.endswith('_eeglStates.pkl'):
                    foldNumer = key.split('_fold')[-1].split('_')[0]
                    foldKey = f'Fold{foldNumer}'
                    secondValueDict[foldKey] = subDict

            firstValueDict[secondValue] = secondValueDict

        resultDict[value] = firstValueDict

    return resultDict


def singleRunAccuracies(runDict, plot=False, title=None):
    # Initialize a dictionary to store the sum of accuracies for each iteration
    iterationData = {}
    numFolds = len(runDict)  # Total number of folds

    # Loop through each fold and gather accuracies by iteration
    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            if iteration not in iterationData:
                iterationData[iteration] = {'test': 0, 'train': 0, 'val': 0, 'count': 0}
            accuracyDict = iterationResults['accuracies']

            # Sum up the test, train, and val accuracies
            iterationData[iteration]['test'] += accuracyDict['test']
            iterationData[iteration]['train'] += accuracyDict['train']
            iterationData[iteration]['val'] += accuracyDict['val']
            iterationData[iteration]['count'] += 1  # Count iterations over all folds

    # Now calculate the average for each iteration and print the results
    print(f"Number of folds: {numFolds}")
    for iteration, accuracyData in iterationData.items():
        avgTest = accuracyData['test'] / accuracyData['count']
        avgTrain = accuracyData['train'] / accuracyData['count']
        avgVal = accuracyData['val'] / accuracyData['count']

        print(f"Iteration {iteration}: Average test accuracy: {avgTest:.4f}, "
              f"Average train accuracy: {avgTrain:.4f}, "
              f"Average val accuracy: {avgVal:.4f}")

    if plot == True:
        avgTest = []
        avgTrain = []
        avgVal = []
        iterationLabels = []

        # Calculate the average for each iteration and store it for plotting
        for iteration in iterationData:
            acc_data = iterationData[iteration]
            avg_test = acc_data['test'] / acc_data['count']
            avg_train = acc_data['train'] / acc_data['count']
            avg_val = acc_data['val'] / acc_data['count']

            avgTest.append(avg_test)
            avgTrain.append(avg_train)
            avgVal.append(avg_val)
            iterationLabels.append(iteration)

        # Plot the accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(iterationLabels, avgTest, label='Test Accuracy', marker='o', color='blue')
        plt.plot(iterationLabels, avgTrain, label='Train Accuracy', marker='o', color='green')
        plt.plot(iterationLabels, avgVal, label='Validation Accuracy', marker='o', color='red')

        # Add titles and labels
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.show()


import matplotlib.pyplot as plt


def analyseAndPlotRunAccuracies(runDicts, runNames):
    """
    Analyse and plot only test accuracies across multiple runs, print them, and plot them.

    Args:
        runDicts (list): A list of run dictionaries containing accuracies.
        runNames (list): A list of names for each run to be used as labels in the plot.
    """

    # Define a mapping of run names to colors and linestyles
    color_style_map = {
        '$\\sigma = 5, k = 5$': ('#fde725', '-'),  # solid
        '$\\sigma = 5, k = 10$': ('#fde725', '--'),  # dashed
        '$\\sigma = 5, k = 20$': ('#fde725', ':'),  # dotted
        '$\\sigma = 6, k = 5$': ('#72ce55', '-'),
        '$\\sigma = 6, k = 10$': ('#72ce55', '--'),
        '$\\sigma = 6, k = 20$': ('#72ce55', ':'),
        '$\\sigma = 7, k = 5$': ('#23988a', '-'),
        '$\\sigma = 7, k = 10$': ('#23988a', '--'),
        '$\\sigma = 7, k = 20$': ('#23988a', ':'),
        '$\\sigma = 8, k = 5$': ('#33638d', '-'),
        '$\\sigma = 8, k = 10$': ('#33638d', '--'),
        '$\\sigma = 9, k = 5$': ('#450e60', '-'),
        '$\\sigma = 9, k = 10$': ('#450e60', '--'),
    }

    # Define 1 subplot for test accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Loop through each runDict and corresponding name to gather and plot test accuracies
    for runIdx, runDict in enumerate(runDicts):
        runName = runNames[runIdx]
        iterationData = analyseRunAccuracies(runDict)

        avgTest = []
        stdTest = []
        iterationLabels = []

        # Extract the average test accuracies and standard deviations for each iteration
        for iteration, accuracyData in iterationData.items():
            avgTestAcc = accuracyData['test']['mean']
            stdTestAcc = accuracyData['test']['std']

            avgTest.append(avgTestAcc)
            stdTest.append(stdTestAcc)
            iterationLabels.append(iteration)

            # Print test accuracy details for this iteration
            print(f"Run: {runName} | Iteration: {iteration} | "
                  f"Test Accuracy: {avgTestAcc:.4f} ± {stdTestAcc:.4f}")

        # Get the color and linestyle for the current run
        color, linestyle = color_style_map.get(runName, ('#000000', '-'))  # default to black solid if not found

        # Plot test accuracy with error bars for each run
        ax.errorbar(iterationLabels, avgTest, label=runName,
                    marker='o', color=color, linestyle=linestyle, capsize=3)

    # Customize the plot with title, labels, and legend outside the plot
    ax.set_title('Test Accuracy')

    # Set the number of ticks according to the number of iterations you expect (0 to 4)
    num_iterations = len(iterationLabels)
    ax.set_xticks(range(num_iterations))  # Set x-ticks based on the number of iterations
    ax.set_xticklabels([f'R{i}' for i in range(num_iterations)])  # Custom x-tick labels

    ax.set_ylabel('Accuracy')
    ax.grid(True)

    # Place the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside on the right

    plt.tight_layout()
    plt.show()


def analyseRunAccuracies(runDict):
    """
    Analyse the accuracies from a runDict and calculate the average accuracies for each iteration.

    Args:
        runDict (dict): A dictionary containing accuracies for different iterations.

    Returns:
        iterationData (dict): A dictionary with iteration-wise averaged accuracies and standard deviations.
    """
    # Initialize a dictionary to store the sum of accuracies and square sums for each iteration
    iterationData = {}

    # Loop through each fold and gather accuracies by iteration
    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            if iteration not in iterationData:
                # Initialize for this iteration
                iterationData[iteration] = {
                    'test': [],  # Collect accuracies for std dev calculation
                    'train': [],
                    'val': []
                }
            accuracyDict = iterationResults['accuracies']

            # Append the test, train, and val accuracies for this fold
            iterationData[iteration]['test'].append(accuracyDict['test'])
            iterationData[iteration]['train'].append(accuracyDict['train'])
            iterationData[iteration]['val'].append(accuracyDict['val'])

    # Now compute average and std deviation for each iteration
    for iteration, accuracyLists in iterationData.items():
        for key in ['test', 'train', 'val']:
            accuracies = np.array(accuracyLists[key])
            iterationData[iteration][key] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            }

    return iterationData

def generateAccLatexTable(runDicts, runNames):
    """
    Generate a LaTeX-formatted table showing the test accuracy with standard deviation
    for each run across iterations.

    Args:
        runDicts (list): A list of run dictionaries containing accuracies.
        runNames (list): A list of names for each run.

    Returns:
        latexTable (str): A string containing the LaTeX table format.
    """
    iterationLabels = None
    latexTable = "\\begin{tabular}{l" + "c" * len(runDicts[0]) + "}\n\\toprule\n"

    # First row will have the iteration labels (R0, R1, R2, ...)
    latexTable += "Run & " + " & ".join([f"R{iteration}" for iteration in sorted(runDicts[0]['Fold0'].keys())]) + " \\\\\n"
    latexTable += "\\midrule\n"

    # Process each run's accuracies
    for runIdx, runDict in enumerate(runDicts):
        # Get accuracy data for this run
        iterationData = analyseRunAccuracies(runDict)

        # Create a row for this run
        runRow = runNames[runIdx]

        # Append accuracy and std deviation for each iteration in this run
        for iteration in sorted(iterationData.keys()):
            meanAcc = iterationData[iteration]['test']['mean']
            stdAcc = iterationData[iteration]['test']['std']
            runRow += f" & {meanAcc:.4f} $\\pm$ {stdAcc:.4f}"

        runRow += " \\\\\n"
        latexTable += runRow

    latexTable += "\\bottomrule\n\\end{tabular}"

    print(latexTable)
    return latexTable


def analyseRunTimes(runDict):
    # Initialize variables to store the sum of extraction and annotation times
    totalExtractionTime = 0
    totalAnnotationTime = 0
    totalTime = 0
    totalIterations = 0
    numFolds = len(runDict)  # Total number of folds

    # Loop through each fold and each iteration to gather times
    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            # Extract the times for this iteration
            timesDict = iterationResults['times']
            gnnTime = timesDict['gnn']
            explainerTime = timesDict['explainer']
            extractionTime = timesDict['extractor']
            annotationTime = timesDict['annotator']
            runTime = gnnTime + explainerTime + extractionTime + annotationTime

            # Sum up the extraction and annotation times
            totalExtractionTime += extractionTime
            totalAnnotationTime += annotationTime
            totalTime += runTime
            totalIterations += 1

    # Calculate the average extraction and annotation times
    avgExtractionTime = totalExtractionTime / totalIterations if totalIterations > 0 else 0
    avgAnnotationTime = totalAnnotationTime / totalIterations if totalIterations > 0 else 0
    avgTotalTime = totalTime / totalIterations if totalIterations > 0 else 0
    avgRemainingTime = avgTotalTime - avgExtractionTime - avgAnnotationTime

    # Print the results
    print(f"Average Extraction Time: {avgExtractionTime:.4f}")
    print(f"Average Annotation Time: {avgAnnotationTime:.4f}")
    print(f"Average Total Runtime: {avgTotalTime:.4f}\n")

    # Return the computed times
    return avgExtractionTime, avgAnnotationTime, avgRemainingTime, avgTotalTime


def analyseAndPlotRunTimes(runDictList, runNames=None, title=None):
    # Initialize lists to store extracted times for plotting
    extractionTimes = []
    annotationTimes = []
    remainingTimes = []
    totalTimes = []

    # Iterate through the list of runDicts
    for idx, runDict in enumerate(runDictList):
        runName = runNames[idx]
        print(runName)
        # Use the modified analyseRunTimes function to get the average times
        avgExtractionTime, avgAnnotationTime, avgRemainingTime, avgTotalTime = analyseRunTimes(runDict)

        # Store the results for plotting
        extractionTimes.append(avgExtractionTime)
        annotationTimes.append(avgAnnotationTime)
        remainingTimes.append(avgRemainingTime)
        totalTimes.append(avgTotalTime)

    # Number of runs
    numRuns = len(runDictList)

    # Set up the figure and bar positions
    barWidth = 0.5
    r = range(numRuns)

    # Set default run names if not provided
    if runNames is None:
        runNames = [f'Run {i + 1}' for i in r]

    # Create the stacked bar chart
    plt.figure(figsize=(10, 6))

    # Plot the bars with annotation, extraction, and remaining time stacked
    bars1 = plt.bar(r, annotationTimes, color='#015aaa', edgecolor='white', width=barWidth, label='Annotation Time')
    bars2 = plt.bar(r, extractionTimes, bottom=annotationTimes, color='#fcb815', edgecolor='white', width=barWidth,
                    label='Extraction Time')
    bars3 = plt.bar(r, remainingTimes, bottom=[i + j for i, j in zip(annotationTimes, extractionTimes)],
                    color='#9fa192', edgecolor='white', width=barWidth, label='Remaining Time (GNN + Explainer)')

    # Add labels and title
    plt.xlabel('Runs', fontweight='bold')
    plt.ylabel('Average Time (s)', fontweight='bold')
    plt.title(title)

    plt.xticks(r, runNames)  # Use runNames for x-tick labels
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def labelDistrobution(runDict, numLabels, title=None):
    labelDistribution = [0] * numLabels  # Fixing typo to "Distribution"

    # Calculate label distribution
    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            patternsList = iterationResults['patterns']
            for pattern in patternsList:
                patternLabel = pattern.graph['label']
                labelDistribution[patternLabel] += 1

                # Plotting the label distribution using a bar chart
    labels = list(range(numLabels))  # Labels from 0 to numLabels-1
    plt.bar(labels, labelDistribution)

    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(title if title else 'Label Distribution')

    plt.xticks(labels)  # Show labels at x-axis

    # Show the plot
    plt.show()

def plotAveragePatternSizes(runDictList, runNames=None, title=None):

    colors = plt.cm.get_cmap('viridis_r', len(runDictList))  # Use 'tab10' colormap for distinct colors
    plt.figure(figsize=(10, 6))

    for runDict, runName in zip(runDictList, runNames):
        avgPatternSize, totalPatternCount, treePercentage = averagePatternSize(runDict)
        print(f'Run {runName}: Average Pattern Size: {avgPatternSize}, '
              f'Total Pattern Count: {totalPatternCount}, '
              f'Percentage of Trees: {treePercentage:.2f}%')


def averagePatternSize(runDict):

    totalPatternSize = 0
    totalPatternCount = 0
    treeCount = 0

    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            patternsList = iterationResults['patterns']
            for pattern in patternsList:
                totalPatternSize += len(pattern.edges())
                totalPatternCount += 1
                if nx.is_tree(pattern):  # Check if the pattern is a tree
                    treeCount += 1

    averagePatternSize = totalPatternSize / totalPatternCount if totalPatternCount > 0 else 0
    treePercentage = (treeCount / totalPatternCount * 100) if totalPatternCount > 0 else 0

    return averagePatternSize, totalPatternCount, treePercentage

def analyseAndPlotF1Scores(runDicts, runNames, title=None):
    """
    Analyse F1 scores across multiple runs and plot the average F1 scores over iterations for each run.

    Args:
        runDicts (list): A list of run dictionaries containing F1 scores.
        runNames (list): A list of names for each run to be used as labels in the plot.
        title (str, optional): Title of the plot. Defaults to None.
    """
    # Define the specific colors for each combination of mining method and sigma value
    color_map = {
        'FTM, $\\sigma = 8$': '#fde725',
        'FTM, $\\sigma = 10$': '#a8db34',
        'FTM, $\\sigma = 12$': '#5cc863',
        'FSM, $\\sigma = 8$': '#3b518b',
        'FSM, $\\sigma = 10$': '#472c7a',
        'FSM, $\\sigma = 12$': '#440154'
    }

    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get current axis

    # Initialize a variable to track the maximum number of iterations across all runs
    max_iterations = 0

    # Loop through each runDict and corresponding name to gather and plot F1 scores
    for runIdx, runDict in enumerate(runDicts):
        iterationData = analyseF1Scores(runDict)
        print(f"\n{runNames[runIdx]}")
        avgF1Scores = []
        iterationLabels = []

        # Extract the average F1 scores for each iteration
        for iteration, f1Data in iterationData.items():
            avgF1Scores.append(f1Data['average_f1'])
            iterationLabels.append(iteration)
            print(iteration, f1Data['average_f1'])

        # Track the maximum number of iterations across all runs
        max_iterations = max(max_iterations, len(iterationLabels))

        # Plot the F1 scores for this run, using the predefined colors
        color = color_map.get(runNames[runIdx], 'black')  # Default to black if runName not in color_map
        plt.plot(iterationLabels, avgF1Scores, label=runNames[runIdx], marker='o', color=color)

    # Set x-ticks and custom x-tick labels as 'R1', 'R2', etc.
    num_iterations = max_iterations
    ax.set_xticks(range(num_iterations))  # Set x-ticks based on the number of iterations
    ax.set_xticklabels([f'R{i+1}' for i in range(num_iterations)])  # Custom x-tick labels

    # Customize the plot with titles, labels, and legends
    plt.title(title)
    plt.ylabel('Average top 10 pattern F1-Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyseF1Scores(runDict):
    """
    Analyse the F1 scores from a runDict and calculate the average F1 score for each iteration.

    Args:
        runDict (dict): A dictionary containing F1 scores for different iterations.

    Returns:
        iterationData (dict): A dictionary with iteration-wise averaged F1 scores.
    """
    iterationData = {}

    # Loop through each fold and iteration to gather F1 scores
    for foldKey, foldData in runDict.items():
        for iteration, iterationResults in foldData.items():
            if iteration not in iterationData:
                iterationData[iteration] = {'total_f1': 0, 'count': 0}

            # Extract the patterns list for this iteration
            patternsList = iterationResults['patterns']

            # Analyze F1-Scores for this iteration
            if patternsList is None or len(patternsList) == 0:
                average_f1 = 0
            else:
                average_f1, _, _, _ = analyseF1ScoresforPatterns(patternsList)

            # Sum up F1-Scores for this iteration
            iterationData[iteration]['total_f1'] += average_f1
            iterationData[iteration]['count'] += 1

    # Calculate the average F1 score for each iteration
    for iteration in iterationData:
        iterationData[iteration]['average_f1'] = iterationData[iteration]['total_f1'] / iterationData[iteration][
            'count']

    return iterationData


def analyseF1ScoresforPatterns(topKpatterns):
    """
    Analyse F1 scores for a list of patterns and return the average, max, and min F1 scores.

    Args:
        topKpatterns (list): A list of patterns with F1 scores.

    Returns:
        averageF1Score (float): The average F1 score across all patterns.
        maxF1Score (float): The maximum F1 score among the patterns.
        minF1Score (float): The minimum F1 score among the patterns.
        averageLabelF1Scores (dict): The average F1 score for each label.
    """
    # Initialize variables
    totalF1Score = 0
    F1Scores = []
    labelF1Scores = {}
    labelCounts = {}

    # Iterate through each pattern
    for pattern in topKpatterns:
        f1_score = pattern.graph['metrics']['F1-Score']
        label = pattern.graph['label']

        # Collect F1-Scores
        F1Scores.append(f1_score)
        totalF1Score += f1_score

        # Collect label-wise F1-Scores
        if label not in labelF1Scores:
            labelF1Scores[label] = 0
            labelCounts[label] = 0
        labelF1Scores[label] += f1_score
        labelCounts[label] += 1

    # Calculate statistics
    averageF1Score = totalF1Score / len(topKpatterns)
    maxF1Score = max(F1Scores)
    minF1Score = min(F1Scores)

    # Calculate average F1-Score for each label
    averageLabelF1Scores = {label: labelF1Scores[label] / labelCounts[label]
                            for label in labelF1Scores}

    return averageF1Score, maxF1Score, minF1Score, averageLabelF1Scores


def classificationVisualisation(firstKey, secondKey, fold):
    filePattern = f"fold{fold}_eegl.pk"

    # List all files in the given folder
    files = os.listdir(f'data/eeglRuns/{firstKey}/{secondKey}')

    # Find the file that contains the fold number in its name
    fileName = next((f for f in files if filePattern in f), None)

    # Raise an error if no such file is found
    if fileName is None:
        print("Did not work, file not found.")
        return

    # Construct the full path to the file
    filePath = os.path.join(f'data/eeglRuns/{firstKey}/{secondKey}', fileName)

    # Open and load the model using pickle
    with open(filePath, 'rb') as file:
        modelDict = pkl.load(file)

    # Set up the plot with one subplot for each iteration
    num_iterations = len(modelDict.keys())
    fig, axes = plt.subplots(1, num_iterations, figsize=(15, 5))  # Adjust the figure size as needed

    # Ensure axes is iterable even if there is only one iteration
    if num_iterations == 1:
        axes = [axes]

    # Loop through each iteration and plot the graph
    for idx, iteration in enumerate(modelDict.keys()):
        model_predictions = modelDict[iteration].modelPredictions.cpu().numpy()  # Convert to numpy array
        graph = modelDict[iteration].G  # Assuming this is a networkx-compatible graph

        # Normalize the predictions for color mapping
        cmap = cm.get_cmap('rainbow', 28)  # 28 distinct colors
        color_map = {cls: cmap(i) for i, cls in enumerate(np.unique(model_predictions))}

        # Map node colors according to model predictions
        node_colors = [color_map[model_predictions[node]] for node in graph.nodes]
        pos = nx.kamada_kawai_layout(graph)
        # Draw the graph for this iteration on the corresponding axis
        ax = axes[idx]
        nx.draw(graph, pos, ax=ax, node_color=node_colors, with_labels=False, node_size=50, cmap=plt.cm.Set1)
        # Add the 'R{idx}' label below the graph in bold and larger font size
        ax.text(0.5, -0.02, f'R{idx}', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=20, fontweight='normal', fontfamily='serif')

    plt.tight_layout()
    plt.show()

def G180vis(firstKey, secondKey, fold):
    filePattern = f"fold{fold}_eegl.pk"

    # List all files in the given folder
    files = os.listdir(f'data/eeglRuns/{firstKey}/{secondKey}')

    # Find the file that contains the fold number in its name
    fileName = next((f for f in files if filePattern in f), None)

    # Raise an error if no such file is found
    if fileName is None:
        print("Did not work, file not found.")
        return

    # Construct the full path to the file
    filePath = os.path.join(f'data/eeglRuns/{firstKey}/{secondKey}', fileName)

    # Open and load the model using pickle
    with open(filePath, 'rb') as file:
        modelDict = pkl.load(file)

    # Assuming there's only one key (iteration) in modelDict
    iteration = list(modelDict.keys())[0]  # Get the first (and only) iteration

    model_data = modelDict[iteration]
    true_labels = model_data.data.y.cpu().numpy()  # True labels tensor
    graph = model_data.G  # Assuming this is a networkx-compatible graph

    # Normalize the true labels for color mapping
    cmap = cm.get_cmap('rainbow', 28)  # 28 distinct colors
    color_map = {cls: cmap(i) for i, cls in enumerate(np.unique(true_labels))}

    # Map node colors according to true labels
    node_colors = [color_map[true_labels[node]] for node in graph.nodes]

    # Set up the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    pos = nx.kamada_kawai_layout(graph)  # Define layout

    # Draw the graph with node colors corresponding to true labels
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, font_size=8, node_size=200)
    plt.title("Graph with True Label Coloring")
    plt.show()


def confusionmatrices(firstKey, secondKey, fold):
    filePattern = f"fold{fold}_eegl.pk"

    # List all files in the given folder
    files = os.listdir(f'data/eeglRuns/{firstKey}/{secondKey}')

    # Find the file that contains the fold number in its name
    fileName = next((f for f in files if filePattern in f), None)

    # Raise an error if no such file is found
    if fileName is None:
        print("Did not work, file not found.")
        return

    # Construct the full path to the file
    filePath = os.path.join(f'data/eeglRuns/{firstKey}/{secondKey}', fileName)

    # Open and load the model using pickle
    with open(filePath, 'rb') as file:
        modelDict = pkl.load(file)

    # Set up the figure for subplots (plot matrices side by side)
    fig, axes = plt.subplots(1, len(modelDict), figsize=(5 * len(modelDict), 5))

    for idx, iteration in enumerate(modelDict.keys()):
        true_labels = modelDict[iteration].data.y  # True labels tensor
        model_predictions = modelDict[iteration].modelPredictions

        # 2. Convert tensors to numpy arrays for use with sklearn
        y_true = true_labels.cpu().numpy()
        y_pred = model_predictions.cpu().numpy()

        # 3. Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # 4. Plot the confusion matrix using viridis colormap
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        ax = axes[idx]  # Assign to the correct subplot
        disp.plot(cmap='viridis_r', ax=ax, colorbar=False)

        # Remove the axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Set the title as 'R{iteration}'
        ax.set_title(f"Round: R{iteration}", fontsize=16)

        # Set tick labels visibility
        ax.tick_params(labelleft=False, labelbottom=True)

    # Adjust layout so plots don't overlap
    plt.tight_layout(pad=2.0)

    # Optionally, display the confusion matrix plot
    plt.show()


def topKpatterns(firstKey, secondKey, fold):
    filePattern = f"fold{fold}_eegl.pk"

    # List all files in the given folder
    files = os.listdir(f'data/eeglRuns/{firstKey}/{secondKey}')

    # Find the file that contains the fold number in its name
    fileName = next((f for f in files if filePattern in f), None)

    # Raise an error if no such file is found
    if fileName is None:
        print("Did not work, file not found.")
        return

    # Construct the full path to the file
    filePath = os.path.join(f'data/eeglRuns/{firstKey}/{secondKey}', fileName)

    # Open and load the model using pickle
    with open(filePath, 'rb') as file:
        modelDict = pkl.load(file)

    # Set up the figure for subplots (plot matrices side by side)
    fig, axes = plt.subplots(1, len(modelDict), figsize=(5 * len(modelDict), 5))

    for idx, iteration in enumerate(modelDict.keys()):
        topKpatterns = modelDict[iteration].topKpatterns
        visualiseTopPatterns(topKpatterns)

def getRunDict(mainDict, firstFolderKey, secondFolderKey, runDictName):
    resultDict = mainDict[firstFolderKey][secondFolderKey]

    with open('data/eeglRuns/runDicts/' + runDictName, 'wb') as f:
        pkl.dump(resultDict, f)


def loadAndPlot(dataset, title=None):
    """
    Function to load the run dictionaries and plot the accuracies.

    :param dataset: List containing [runDict names, run display names]
    :param title: The title to use in the plot
    """
    dict_names = dataset[0]
    run_names = dataset[1]
    dicts = []

    # Load the runDicts
    for dict_name in dict_names:
        with open(f'data/eeglRuns/runDicts/{dict_name}.pkl', 'rb') as file:
            run_dict = pkl.load(file)
            dicts.append(run_dict)

    # Call the function to analyse and plot
    analyseAndPlotRunAccuracies(dicts, run_names)
    analyseAndPlotRunTimes(dicts, run_names)
    generateAccLatexTable(dicts, run_names)
    analyseAndPlotF1Scores(dicts, run_names)
    plotAveragePatternSizes(dicts, run_names)


probftmG180 = [
    ['PROBFTM_G180_S5_K5', 'PROBFTM_G180_S5_K10', 'PROBFTM_G180_S5_K20',
     'PROBFTM_G180_S6_K5', 'PROBFTM_G180_S6_K10', 'PROBFTM_G180_S6_K20',
     'PROBFTM_G180_S7_K5', 'PROBFTM_G180_S7_K10', 'PROBFTM_G180_S7_K20',
     'PROBFTM_G180_S8_K5', 'PROBFTM_G180_S8_K10',
     'PROBFTM_G180_S9_K5', 'PROBFTM_G180_S9_K10'],
    [
        'σ = 5\nk = 5', 'σ = 5\nk = 10', 'σ = 5\nk = 20',
        'σ = 6\nk = 5', 'σ = 6\nk = 10', 'σ = 6\nk = 20',
        'σ = 7\nk = 5', 'σ = 7\nk = 10', 'σ = 7\nk = 20',
        'σ = 8\nk = 5', 'σ = 8\nk = 10',
        'σ = 9\nk = 5', 'σ = 9\nk = 10'
    ]
]



m1all = [
    ['GASTON_m1_S8', 'FTM_m1_S8', 'GASTON_m1_S10', 'FTM_m1_S10', 'GASTON_m1_S12', 'FTM_m1_S12'],
    ['FSM, $\\sigma = 8$', 'FTM, $\\sigma = 8$', 'FSM, $\\sigma = 10$', 'FTM, $\\sigma = 10$', 'FSM, $\\sigma = 12$', 'FTM, $\\sigma = 12$']]


m1pall = [
    ['GASTON_m1p_S8', 'FTM_m1p_S8', 'GASTON_m1p_S10', 'FTM_m1p_S10', 'GASTON_m1p_S12', 'FTM_m1p_S12'],
    ['FSM, $\\sigma = 8$', 'FTM, $\\sigma = 8$', 'FSM, $\\sigma = 10$', 'FTM, $\\sigma = 10$', 'FSM, $\\sigma = 12$', 'FTM, $\\sigma = 12$']]

m2all = [
    ['GASTON_m2_S8', 'FTM_m2_S8', 'GASTON_m2_S10', 'FTM_m2_S10', 'GASTON_m2_S12', 'FTM_m2_S12'],
    ['FSM, $\\sigma = 8$', 'FTM, $\\sigma = 8$', 'FSM, $\\sigma = 10$', 'FTM, $\\sigma = 10$', 'FSM, $\\sigma = 12$', 'FTM, $\\sigma = 12$']]

m2pall = [
    ['GASTON_m2p_S8', 'FTM_m2p_S8', 'GASTON_m2p_S10', 'FTM_m2p_S10', 'GASTON_m2p_S12', 'FTM_m2p_S12'],
    ['FSM, $\\sigma = 8$', 'FTM, $\\sigma = 8$', 'FSM, $\\sigma = 10$', 'FTM, $\\sigma = 10$', 'FSM, $\\sigma = 12$', 'FTM, $\\sigma = 12$']]

G180all = [['GASTON_G180_S8', 'FTM_G180_S8', 'GASTON_G180_S10', 'FTM_G180_S10', 'GASTON_G180_S12', 'FTM_G180_S12'],
            ['FSM, $\\sigma = 8$', 'FTM, $\\sigma = 8$', 'FSM, $\\sigma = 10$', 'FTM, $\\sigma = 10$', 'FSM, $\\sigma = 12$', 'FTM, $\\sigma = 12$']]

def runPlanner():
    minerKeys = ['m2GASTON', 'm2FTM']
    miners = ['GASTON', 'FTM']
    patternSizes = ['PatternSize12']
    sigmas = [12]

    nestedDict = buildNestedDict('data/eeglRuns/', minerKeys, patternSizes)
    for minerKey, miner in zip(minerKeys, miners):
        for patternSize,  sigma in zip(patternSizes, sigmas):
            getRunDict(nestedDict, minerKey, patternSize, f'{miner}_M2_S{sigma}.pkl')



if __name__ == '__main__':

    confusionmatrices('m2FTM', 'PatternSize12', 1)
    confusionmatrices('m2GASTON', 'PatternSize12', 1)

    confusionmatrices('m2FTM', 'PatternSize12', 2)
    confusionmatrices('m2GASTON', 'PatternSize12', 2)

    confusionmatrices('m2FTM', 'PatternSize12', 3)
    confusionmatrices('m2GASTON', 'PatternSize12', 3)

    confusionmatrices('m2FTM', 'PatternSize12', 4)
    confusionmatrices('m2GASTON', 'PatternSize12', 4)


    '''
    loadAndPlot(G180all)
    loadAndPlot(m1all)
    loadAndPlot(m1pall)
    loadAndPlot(m2all)
    loadAndPlot(m2pall)

    
    
    classificationVisualisation('G180FTM', 'PatternSize12', 0)
    classificationVisualisation('G180FTM', 'PatternSize12', 1)
    classificationVisualisation('G180FTM', 'PatternSize12', 2)
    classificationVisualisation('G180FTM', 'PatternSize12', 3)
    classificationVisualisation('G180FTM', 'PatternSize12', 4)

    
    topKpatterns('m1FTM', 'PatternSize12', 0)
    topKpatterns('m1FTM', 'PatternSize12', 1)
    topKpatterns('m1FTM', 'PatternSize12', 2)
    topKpatterns('m1FTM', 'PatternSize12', 3)
    topKpatterns('m1FTM', 'PatternSize12', 4)
   
    '''

