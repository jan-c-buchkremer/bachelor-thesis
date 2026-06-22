from tqdm.auto import tqdm
import networkx as nx
import subprocess
import os


extractorTempsDirectory = "data/temps/PatternExtractor"

inputDirectory = os.path.join(extractorTempsDirectory, "input")
os.makedirs(inputDirectory, exist_ok=True)

outputDirectory = os.path.join(extractorTempsDirectory, "output")
os.makedirs(outputDirectory, exist_ok=True)

def networkXtoFile(labelDict):

    for label, (explanationGraphs, explanationRoots) in labelDict.items():

        filePath = os.path.join(inputDirectory, f"label_{label}_graph")

        with open(filePath, "w") as file:

            for idx, (graph, root) in enumerate(zip(explanationGraphs, explanationRoots)):

                file.write(f"t # {idx}\n")
                nodes = list(graph.nodes())

                for node in nodes:
                    '''
                    Root node is distinguished by label 1 in gaston file graph. 
                    Other nodes get label 0
                    '''
                    if node == root: file.write(f"v {node} 1\n")  # Root node
                    else: file.write(f"v {node} 0\n")  # Other nodes

                for edge in graph.edges(): file.write(f"e {edge[0]} {edge[1]} 0\n")  # Edge with label 0


def gaston(filename, support=None, patternSize=None):

    outputFilename = filename + ".out"
    projectDirectory = os.path.dirname(os.path.abspath(__file__))

    gastonPath = os.path.join(projectDirectory, "external/gaston-1.1")
    inputFile = os.path.join(projectDirectory, inputDirectory, filename)
    outputFile = os.path.join(projectDirectory, outputDirectory, outputFilename)

    support = str(support)
    patternSize = str(patternSize)

    # Ensure the paths are correct and accessible
    assert os.path.isfile(inputFile), f"Input file {inputFile} does not exist."
    assert os.path.isdir(gastonPath), f"Gaston directory {gastonPath} does not exist."

    currentDirectory = os.getcwd()
    os.chdir(gastonPath)

    cmd = ["./gaston", "-m", patternSize, support, inputFile, outputFile]

    try:

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stderr:
            pass

    except subprocess.CalledProcessError as e:

        print(f"Error running Gaston: {e}")
        print(e.stdout)
        print(e.stderr)

    # Change back to the original directory
    os.chdir(currentDirectory)


def gastonToNetworkX(outputFilename):

    frequentSubgraphs = []
    currentGraph = None
    outputFile = os.path.join(outputDirectory, outputFilename)

    with open(outputFile, 'r') as file:

        for line in file:
            line = line.strip()

            if line.startswith('#'):
                '''
                # signifies support count
                Current graphs are added when the start signified for a new graph appears and the root node has been 
                found in its nodeset.
                '''
                if currentGraph is not None and currentGraph.graph['root'] is not None: # Only add graphs with root
                    frequentSubgraphs.append(currentGraph)

                support = int(line.split()[1])
                currentGraph = nx.Graph(root=None, support=support)

            elif line.startswith('t'):

                # t signifies graph ID
                continue

            elif line.startswith('v'):

                parts = line.split()
                node = int(parts[1])
                nodeLabel = int(parts[2])

                # Signify a root node in the graph
                if nodeLabel == 1: currentGraph.graph['root'] = node
                currentGraph.add_node(node)

            elif line.startswith('e'):

                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                edgeLabel = int(parts[3])
                currentGraph.add_edge(u, v)

        if currentGraph is not None and currentGraph.graph['root'] is not None:  # Only add graphs with root
            frequentSubgraphs.append(currentGraph)

    return frequentSubgraphs # list of frequent subgraphs, each being saved as nx graph with 'root' signifying root node

def runGaston(explanationsByLabel, labelSupportDict,  patternSize=None):


    frequentSubgraphsByLabel = {}
    networkXtoFile(explanationsByLabel)
    inputFiles = [f for f in os.listdir(inputDirectory)]

    with tqdm(total=len(inputFiles), desc=f"Gaston FSM for {len(explanationsByLabel)} labels with patterns up to size {patternSize}") as pbar:
        for inputFile in inputFiles:

            label = int(inputFile.split('_')[1])  # Convert label to int
            support = labelSupportDict[label]
            explanationGraphs, explanationRoots = explanationsByLabel[label]

            if len(explanationGraphs) == 0:
                frequentSubgraphsByLabel[label] = []

            elif len(explanationGraphs) == 1:
                frequentSubgraphsByLabel[label] = explanationGraphs
                for graph, root in zip(explanationGraphs, explanationRoots):
                    graph.graph['support'] = 1
                    graph.graph['root'] = root

            else:
                gaston(inputFile, support,  patternSize)
                outputFilename = inputFile + ".out"

                nxLabelExplanations = gastonToNetworkX(outputFilename)
                frequentSubgraphsByLabel[label] = nxLabelExplanations

            pbar.update(1)

    # Cleanup: Remove all files in input and output directories using rm -r *
    try:
        subprocess.run(f"rm -r {inputDirectory}/*", shell=True, check=True)
        subprocess.run(f"rm -r {outputDirectory}/*", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during cleanup: {e}")


    return frequentSubgraphsByLabel # outputs dictionary of labels with lists of frequent subgraphs for each
