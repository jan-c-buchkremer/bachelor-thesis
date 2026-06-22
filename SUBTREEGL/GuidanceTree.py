from collections import deque
import networkx as nx
import random


def calculateBlocks(G, root):
    articulation_points = list(nx.articulation_points(G))
    biconnected_components = list(nx.biconnected_component_edges(G))
    return biconnected_components


def createNodeBlocksDict(blocks):
    nodeBlocksDict = {}
    for block in blocks:
        for edge in block:
            for node in edge:
                nodeBlocksDict.setdefault(node, set()).add(frozenset(block))
    return {node: list(blocks) for node, blocks in nodeBlocksDict.items()}


def SkeletonTree(G, nodeBlocksDict, r):

    T = nx.DiGraph()
    visited = set()
    queue = deque([r])
    visited.add(r)
    finishedBlocks = []
    rootBlocksDict = {}
    latesAnestors = {r: None}

    while queue:
        isBlockRoot = False
        currentNode = queue.popleft()
        if currentNode in nodeBlocksDict:
            blocksForCurrentNode = nodeBlocksDict[currentNode]
        else:
            print("Was ist denn hier los?")

        for block in blocksForCurrentNode:
            if block not in finishedBlocks:

                T.add_node(currentNode)

                if latesAnestors[currentNode] is not None:
                    T.add_edge(latesAnestors[currentNode], currentNode)

                if currentNode not in rootBlocksDict.keys():
                    rootBlocksDict[currentNode] = set()

                rootBlocksDict[currentNode] = rootBlocksDict[currentNode].union(block)

                finishedBlocks.append(block)
                isBlockRoot = True

        for neighbor in G.neighbors(currentNode):
            if neighbor not in visited:
                if isBlockRoot:
                    latesAnestors[neighbor] = currentNode
                else:
                    latesAnestors[neighbor] = latesAnestors.get(currentNode)
                visited.add(neighbor)
                queue.append(neighbor)

    for node, blocks in rootBlocksDict.items():
        rootBlocksDict[node] = [block for block in blocks]

    return T, rootBlocksDict


def spanningTreeIter(blockGraph: nx.Graph, k, source):
    """An iterator that yields up to k random directed spanning trees of the given blockGraph rooted at the source node.

    Args:
        blockGraph (nx.Graph): The input graph from which to generate spanning trees.
        k (int): The number of random spanning trees to generate.
        source (int): The source node to orient the trees.

    Yields:
        nx.DiGraph: A directed random spanning tree of the blockGraph rooted at the source node.
    """

    # Calculate the number of spanning trees in the graph
    num_spanning_trees = round(nx.number_of_spanning_trees(blockGraph))


    # Ensure we only generate up to the maximum number of spanning trees
    currentK = min(k, num_spanning_trees)

    for _ in range(currentK):
        # Generate a random spanning tree as an undirected graph
        spanning_tree_edges = nx.random_spanning_tree(blockGraph)
        spanning_tree = nx.Graph(spanning_tree_edges)

        # Convert the spanning tree to a directed graph, oriented from the source node
        directed_tree = nx.bfs_tree(spanning_tree, source=source)

        yield directed_tree


def getGuidanceTree(G, r, k):

    if G.number_of_nodes() == 1:
        T = nx.DiGraph()
        for node in G.nodes():
            T.add_node(node)
            return T, {node: [G]}

    blockEdges = calculateBlocks(G, r)
    nodeBlocksDict = createNodeBlocksDict(blockEdges)
    T, rootsBlocksDict = SkeletonTree(G, nodeBlocksDict, r)

    spanningTreesDict = {}
    for root, blockEdges in rootsBlocksDict.items():
        blockGraph = nx.Graph()
        blockGraph.add_edges_from(blockEdges)

        trees = spanningTreeIter(blockGraph, k=k, source=root)
        spanningTreesDict[root] = list(trees)

    return T, spanningTreesDict
