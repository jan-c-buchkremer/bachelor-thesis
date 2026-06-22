from SUBTREEGL.SubTreeIsoTester import SubTreeIsoTester
from networkx.algorithms import isomorphism
import networkx as nx
import subprocess
import tempfile
import os

projectDirectory = os.path.dirname(os.path.abspath(__file__))
solverTempsDirectory = "data/temps/SubgraphSolver"

def nxToCSV(graph, csvFile,  root=None):
    with open(csvFile, 'w') as file:
        for node in graph.nodes:
            if node == root:
                file.write(f"{node},,1\n")
            else:
                file.write(f"{node},,0\n")

        for edge in graph.edges():
            file.write(f"{edge[0]},{edge[1]}\n")



def csvToNX(csvFile):
    graph = nx.Graph()

    with open(csvFile, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[1] == '': # node with label
                node = parts[0]
                label = parts[2]

                if label != '':
                    graph.add_node(node, label=label)
                else:
                    graph.add_node(node)

            elif len(parts) >= 2: #edge
                u = parts[0]
                v = parts[1]

                if len(parts) == 3: #with label
                    label = parts[2]
                    if label != '':
                        graph.add_edge(u, v, label=label)
                    else:
                        graph.add_edge(u, v)
                else:
                    graph.add_edge(u, v)

    return graph


def subgraphIsoTest(targetGraph, patternGraph, method='glasgow', root=None, guidanceTree=None, k=None):
    """
    Tests if the patternGraph is isomorphic to a subgraph of the targetGraph.

    Args:
        targetGraph (networkx.Graph): The graph in which to search for the subgraph.
        patternGraph (networkx.Graph): The graph pattern to search for within the targetGraph.
        method (str, optional): The subgraph isomorphism method to use ('glasgow' or 'subtree').
                                Defaults to 'glasgow'.
        root (node, optional): The root node for subtree isomorphism testing.
                               Used only when method is 'subtree'. Defaults to None.

    Returns:
        bool: True if the patternGraph is isomorphic to a subgraph of the targetGraph, otherwise False.

    Raises:
        ValueError: If the specified method is not supported.
    """

    if method == 'glasgow':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as patternGraphFile, \
                tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as targetGraphFile:

            if root is not None:
                nxToCSV(patternGraph, patternGraphFile.name, root=patternGraph.graph['root'])
                nxToCSV(targetGraph, targetGraphFile.name, root=root)

            else:
                nxToCSV(patternGraph, patternGraphFile.name, root=None)
                nxToCSV(targetGraph, targetGraphFile.name, root=None)

            result = glasgowSubgraphSolver(patternGraphFile.name, targetGraphFile.name)

            # Clean up the temporary files
            os.remove(patternGraphFile.name)
            os.remove(targetGraphFile.name)

            return result

    elif method == 'subtree':

        sti = SubTreeIsoTester(G=targetGraph, H=patternGraph, k=k, rG=root)
        (isoTestResult, _) = sti.getSubTreeIso()
        return isoTestResult

    elif method == 'nx':
        GM = isomorphism.GraphMatcher(G1=targetGraph, G2=patternGraph)
        return GM.subgraph_is_isomorphic()
    else:
        raise ValueError(f'Subgraph Isomorphism Tester {method} not supported')


def glasgowSubgraphSolver(patternFile, targetFile):
    """
        Executes the Glasgow subgraph solver to determine if a pattern graph is
        isomorphic to a subgraph of a target graph.

        Args:
            patternFile (str): Path to the CSV file representing the pattern graph.
            targetFile (str): Path to the CSV file representing the target graph.

        Returns:
            bool: True if the pattern graph is isomorphic to a subgraph of the target graph, otherwise False.
        """
    # Absolute path to the Glasgow subgraph solver
    glasgowPath = os.path.join(projectDirectory,
                               "../EEGL/external/glasgow-subgraph-solver-master/build/glasgow_subgraph_solver")
    glasgowPath = os.path.abspath(glasgowPath)  # Convert to absolute path

    # Prepare the command by providing full paths to the solver, pattern file, and target file
    command = f"{glasgowPath} --parallel {patternFile} {targetFile}"

    # Execute the command using subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate()

    # Decode the output and errors
    decodedOutput = output.decode('utf-8')
    decodedErrors = errors.decode('utf-8')

    # Check if 'status = true' exists in any line of the output
    return 'status = true' in decodedOutput.lower()

