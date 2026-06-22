from Jan.SubTreeIsoTester import SubTreeIsoTester
from Jan.Visualisation import visualiseTree
from Jan.Glasgow import glasgowIsoTest
from Jan.TreeGenerator import TreeIterator
from joblib import Parallel, delayed
import networkx as nx
from time import time
import pickle as pkl
import time
import string
from collections import deque

def checkIsomorphisms(nodes, i, levelTree, iPlusOne, j, levelAboveTree):
    isoTestResult = glasgowIsoTest(targetGraph=levelAboveTree, patternGraph=levelTree, root=0)
    if isoTestResult:
        return (f'{nodes}_{i}', f'{iPlusOne}_{j}'), levelAboveTree
    return None


def rootedTreeLevelGraph(maxNodes):
    nodes = 1
    levelSubtreesDict = {}
    subtreesLevelGraph = nx.DiGraph()
    visualiseTree(subtreesLevelGraph)

    while nodes < maxNodes:
        levelTrees = list(levelRootedTrees(nodes))
        levelAboveTrees = list(levelRootedTrees(nodes + 1))
        print(
            f"Currently checking {len(levelTrees) * len(levelAboveTrees)} Isomorphisms for rooted Trees of {nodes} nodes into rooted Trees of {nodes + 1} nodes.")

        # Add nodes for current level
        for i, levelTree in enumerate(levelTrees):
            subtreesLevelGraph.add_node(f'{nodes}_{i}')
            levelSubtreesDict[f'{nodes}_{i}'] = levelTree

        # Use Parallel to check isomorphisms
        results = Parallel(n_jobs=-1, verbose=2)(
            delayed(checkIsomorphisms)(nodes, i, levelTree, nodes + 1, j, levelAboveTree)
            for i, levelTree in enumerate(levelTrees)
            for j, levelAboveTree in enumerate(levelAboveTrees)
        )

        # Add nodes and edges found by parallel processing
        for result in results:
            if result is not None:
                edge = result[0]
                levelAboveTree = result[1]
                subtreesLevelGraph.add_edge(*edge)
                levelSubtreesDict[edge[1]] = levelAboveTree
                subtreesLevelGraph.add_node(edge[1])

        nodes += 1

    for key, tree in levelSubtreesDict.items():
        tree = relabelTree(tree, 0)
        levelSubtreesDict[key] = tree
        print(len(tree.nodes()), tree.edges())

    # Add the pattern from levelSubtreeDict as an attribute to each node in the graph
    for node in subtreesLevelGraph.nodes:
        if node in levelSubtreesDict:
            nx.set_node_attributes(subtreesLevelGraph, {node: levelSubtreesDict[node]}, 'pattern')
        else:
            print(f"Node {node} not found in levelSubtreeDict. This is unexpected.")

    return subtreesLevelGraph

def dfsRelabel(tree, root):
    """
    Relabel the nodes of the tree using DFS starting from the root.
    """
    mapping = {}
    newLabel = 0
    stack = [root]
    visited = set(stack)

    while stack:
        current = stack.pop()
        if current not in mapping:
            mapping[current] = newLabel
            newLabel += 1
        neighbors = [n for n in tree.neighbors(current) if n not in visited]
        visited.update(neighbors)
        stack.extend(neighbors)

    tree = nx.relabel_nodes(tree, mapping)
    tree.graph['root'] = 0
    return tree


def calculate_subtree_size(tree, node, parent=None, sizes=None):
    """
    Recursively calculate the size of the subtree for each node.
    """
    if sizes is None:
        sizes = {}

    size = 1  # Count the node itself
    for neighbor in tree.neighbors(node):
        if neighbor != parent:  # Avoid going back to the parent
            size += calculate_subtree_size(tree, neighbor, node, sizes)

    sizes[node] = size
    return size


def relabelTree(tree, root):
    """
    Relabel the nodes of the tree using BFS starting from the root.
    The root node is always labeled '0'. For each level starting from depth 1,
    nodes are prefixed alphabetically, and the numbering starts anew for each level,
    based on the size of the subtree (largest subtree gets '0', second largest gets '1', etc.).
    """
    # Step 1: Calculate subtree sizes for all nodes
    subtree_sizes = {}
    calculate_subtree_size(tree, root, sizes=subtree_sizes)

    # Step 2: BFS relabeling with prefixes and sorted by subtree size
    mapping = {root: 0}  # Root node is always labeled '0'
    queue = deque([(root, 0)])  # Queue stores tuples (node, depth)
    visited = set([root])
    alphabet = string.ascii_lowercase  # a, b, c, ...

    while queue:
        # Collect all nodes at the current depth
        current_level = {}
        current_depth = queue[0][1]  # Get the depth of the first node in the queue

        # Process all nodes at the same depth
        while queue and queue[0][1] == current_depth:
            current, depth = queue.popleft()
            if depth != 0:  # Skip relabeling for root
                current_level[current] = subtree_sizes[current]

            # Add neighbors to the queue
            neighbors = [n for n in tree.neighbors(current) if n not in visited]
            visited.update(neighbors)
            queue.extend([(n, depth + 1) for n in neighbors])

        # Step 3: Sort nodes by their subtree sizes in descending order
        sorted_nodes = sorted(current_level.keys(), key=lambda x: subtree_sizes[x], reverse=True)

        # Step 4: Assign labels based on the sorted order
        for idx, node in enumerate(sorted_nodes):
            prefix = alphabet[current_depth - 1] if current_depth - 1 < len(alphabet) else ''
            mapping[node] = f"{prefix}{idx}"

    return nx.relabel_nodes(tree, mapping)


def isNotIsomorphic(reRootedTree, existingTrees):
    """
    Check if the re-rooted tree is not isomorphic to any of the existing trees.
    """
    for tree in existingTrees:
        tree.graph['root'] = 0
        isoTestResult = glasgowIsoTest(targetGraph=reRootedTree, patternGraph=tree, root=0)
        if isoTestResult:
            return False
    return True

def levelRootedTrees(numNodes):
    """
    Generate all distinct rooted trees for a given number of nodes.
    """
    levelRootedTrees = list(TreeIterator(numNodes))
    newTrees = []

    for tree in levelRootedTrees:
        roots = list(tree.nodes)
        tree.graph['root'] = 0

        reRootedTrees = Parallel(n_jobs=-1, verbose=2)(
            delayed(dfsRelabel)(tree, root) for root in roots
        )

        # Initialize the keep array to True for each rerooted tree
        keep = [True] * len(reRootedTrees)  # keep array corresponds to reRootedTrees indices

        # Compare trees for subtree isomorphism
        for i, hTree in enumerate(reRootedTrees):
            for j, gTree in enumerate(reRootedTrees):
                if i == j or keep[j] == False:  # Skip if it's the same tree or gTree is not kept
                    continue
                isoTestResult = glasgowIsoTest(targetGraph=gTree, patternGraph=hTree, root=0)
                if isoTestResult:
                    keep[i] = False  # Correct assignment of False to mark hTree as duplicate

        reRootedTrees = [tree for i, tree in enumerate(reRootedTrees) if keep[i]]

        # Parallelize the isomorphism checking for each re-rooted tree
        notIsomorphicTrees = Parallel(n_jobs=-1, verbose=2)(
            delayed(isNotIsomorphic)(reRootedTree, levelRootedTrees + newTrees)
            for reRootedTree in reRootedTrees
        )

        # Add the non-isomorphic trees to the list
        for i, notIsomorphic in enumerate(notIsomorphicTrees):
            if notIsomorphic:
                newTrees.append(reRootedTrees[i])


    levelRootedTrees.extend(newTrees)
    return levelRootedTrees


if __name__ == '__main__':

    nodes = 13
    subtreesLevelGraph = rootedTreeLevelGraph(nodes)
    print("Done")
    with open(f'../EEGL/data/levelGraphs/rootedTreesLevelGraph{nodes}.pkl', 'wb') as file:
        pkl.dump({subtreesLevelGraph}, file)

    visualiseTree(subtreesLevelGraph)
    






