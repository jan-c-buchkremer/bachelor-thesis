from SUBTREEGL.FrequentSubtreeMining import getFTMdata
from SUBTREEGL.Visualisation import *
import pickle as pkl
from EEGL.SubgraphSolver import subgraphIsoTest

def loadGraphFromPickle(graphName):

    # Define the directory where pickle files are stored
    path = '../EEGL/data/graphs/'
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


def extractG180motifs():
    G180 = loadGraphFromPickle('G180')
    motifNodes = [0, 15, 30, 45]
    motifNodesNeighbors = [[165, 15], [0, 30], [15, 45], [30, 60]]
    motifs = []

    for motifNode, neighbors in zip(motifNodes, motifNodesNeighbors):
        motif = G180.copy()
        motif.remove_nodes_from(neighbors)
        motif = motif.subgraph(nx.node_connected_component(motif, motifNode))
        motif.graph['root'] = motifNode
        motifs.append(motif)
        #visualiseKamadaGraph(motif, highlightNode=motifNode)

    return motifs


def motifSpanningTrees(motifs, levelGraph):
    motifPatterns = []  # To hold the dictionaries of motif patterns

    for motif in motifs:
        visualiseGraph(motif, highlightNode=motif.graph['root'])
        visited = set()
        motifTrees = {}

        for node in list(levelGraph.nodes):

            if node in visited:
                continue

            pattern = levelGraph.nodes[node].get('pattern')
            if pattern is None:
                continue


            pattern.graph['key'] = node

            subIsoTest = subgraphIsoTest(targetGraph=motif, patternGraph=pattern, root=motif.graph['root'])
            if subIsoTest:
                motifTrees[node] = pattern
            else:
                pruned = {node} | nx.descendants(levelGraph, node)
                levelGraph.remove_nodes_from(pruned)
                visited.update(pruned)

        motifPatterns.append(motifTrees)

    # Now perform the comparisons
    for i, motifA in enumerate(motifPatterns):
        print(f"Motif {i}:")


        # Set to store all patterns in motif A
        patternsA = set(motifA.values())

        # (a) Count patterns unique to motifA compared to each other motif
        for j, motifB in enumerate(motifPatterns):
            if i == j:
                continue  # Skip comparison with itself

            patternsB = set(motifB.values())
            uniqueToMotifA = patternsA - patternsB  # Patterns in A but not in B
            print(f"    Trees unique to Motif {i} compared to Motif {j}: {len(uniqueToMotifA)}")

        # (b) Count patterns in motifA that are not in any other motifs
        otherMotifsCombined = set()
        for j, motifB in enumerate(motifPatterns):
            if i == j:
                continue  # Skip itself
            otherMotifsCombined.update(motifB.values())

        uniqueToMotifAInAll = patternsA - otherMotifsCombined
        #for tree in uniqueToMotifAInAll:
            #visualiseGraph(tree, highlightNode=0, title=f'{len(tree.edges())}')
        print(f"    Trees unique to Motif {i} (in none of the other motifs): {len(uniqueToMotifAInAll)}")

    return None
def constructLowDensityMotifs():
    houseMotif = nx.Graph()
    houseMotif.add_edges_from([(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)])

    houseMotifP = nx.Graph()
    houseMotifP.add_edges_from(houseMotif.edges)
    houseMotifP.add_edge(0, 5)

    M2A = nx.Graph()
    M2A.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)])

    M2B = nx.Graph()
    M2B.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                        (1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)])
    M2pA = nx.Graph()
    M2pA.add_edges_from([(0, 1), (0, 2), (0, 7),
                         (1, 3), (1, 5), (2, 4), (2, 6), (3, 4), (5, 6)])

    M2pB = nx.Graph()
    M2pB.add_edges_from([(0, 1), (0, 2), (0, 7),
                         (1, 3), (1, 4), (3, 4), (2, 5), (2, 6), (5, 6)])

    return houseMotif, houseMotifP, M2A, M2B, M2pA, M2pB


def M2Analysis():
    levelGraph = getFTMdata(12)
    houseMotif, houseMotifP, M2A, M2B, M2pA, M2pB = constructLowDensityMotifs()

    M2Arooted = M2A.copy()
    M2Arooted.graph['root'] = 0
    visualiseGraph(M2Arooted, 'M2 a)', 0)
    M2Brooted = M2B.copy()
    M2Brooted.graph['root'] = 0
    visualiseGraph(M2Brooted, 'M2 b)', 0)
    M2motifs = [M2Arooted, M2Brooted]
    motifSpanningTrees(M2motifs, levelGraph)


def M2pAnalysis():
    levelGraph = getFTMdata(12)
    houseMotif, houseMotifP, M2A, M2B, M2pA, M2pB = constructLowDensityMotifs()
    M2pArooted = M2pA.copy()
    M2pArooted.graph['root'] = 0
    visualiseGraph(M2pArooted, 'M2p a)', 0)

    M2pBrooted = M2pB.copy()
    M2pBrooted.graph['root'] = 0
    visualiseGraph(M2pBrooted, 'M2p b)', 0)

    M2pmotifs = [M2pArooted, M2pBrooted]
    motifSpanningTrees(M2pmotifs, levelGraph)


def M2Vis():
    _, _, M2A, _, _, _ = constructLowDensityMotifs()  # Assuming this function is defined elsewhere

    # Define the trees
    tree1 = nx.Graph()
    tree1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    tree2 = nx.Graph()
    tree2.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 6)])

    # Create a combined plot for M2A with highlighted edges from both trees
    plt.figure(figsize=(24, 12))  # Wider figure to accommodate two plots side by side

    # Visualize with tree1
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    combinedGraph1 = nx.compose(M2A, tree1)
    nodeColours1 = ['firebrick' if node in tree1.nodes else 'navy' for node in combinedGraph1.nodes]
    edgeWidths1 = [10 if edge in tree1.edges else 3 for edge in combinedGraph1.edges]

    nx.draw_networkx(combinedGraph1, with_labels=False, node_color=nodeColours1, node_size=1500, font_size=12,
                     font_weight='bold', font_color="whitesmoke", width=edgeWidths1)

    # Visualize with tree2
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    combinedGraph2 = nx.compose(M2A, tree2)
    nodeColours2 = ['firebrick' if node in tree2.nodes else 'navy' for node in combinedGraph2.nodes]
    edgeWidths2 = [10 if edge in tree2.edges else 3 for edge in combinedGraph2.edges]

    nx.draw_networkx(combinedGraph2, with_labels=False, node_color=nodeColours2, node_size=1500, font_size=12,
                     font_weight='bold', font_color="whitesmoke", width=edgeWidths2)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


if __name__ == '__main__':
    motifs = extractG180motifs()
    levelGraph = getFTMdata(12)
    motifSpanningTrees(motifs, levelGraph)

