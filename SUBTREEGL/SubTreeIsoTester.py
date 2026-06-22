from SUBTREEGL.Visualisation import visualiseTree, visualiseBipartiteGraph
from SUBTREEGL.GuidanceTree import *
import networkx as nx
import time

class SubTreeIsoTester:
    def __init__(self, G, H, k, rG=None, guidanceTree=None, characteristics=None, debug=False, print=False, visualise=False, time=False):

        self.G = G
        self.H = H
        self.k = k
        self.rG = rG
        self.debug = debug
        self.print = print
        self.visualise = visualise
        self.time = time

        if rG == None: self.rG = next(iter(G.nodes)) # choose any node as root

        if guidanceTree is not None: self.T, self.spanningTreesDict = guidanceTree
        else: self.T, self.spanningTreesDict = getGuidanceTree(self.G, self.rG, self.k)

        if characteristics is not None: self.characteristics = characteristics
        else: self.characteristics = set()

        if self.debug:
            self.print = True
            self.visualise = True

        self.matchingCache = {}
        self.numIsoTriples = 0

    def getSubTreeIso(self):
        subIsoTime = time.time()
        if self.print: print("G :", self.G.edges())
        if self.print: print("H :", self.H.edges())
        if len(self.H.nodes()) == 1 and len(self.G.nodes()) >= 1: return True, self.characteristics

        self.T, self.spanningTreesDict = getGuidanceTree(self.G, self.rG, self.k)

        if self.visualise: visualiseTree(self.T, title="Skeleton Tree")

        for v in list(nx.dfs_postorder_nodes(self.T, source=self.rG)):  # Postorder Iteration

            if self.print: print("Block Root v: ", v)
            if self.visualise: visualiseTree(self.T, highlightNode=v, title="Skeleton Tree with current v")

            for localTreeV in self.spanningTreesDict[v]:

                if self.visualise: visualiseTree(localTreeV, highlightNode=v, title="Blocktree with root")

                for w in list(nx.dfs_postorder_nodes(localTreeV, source=v)):  # Postorder Iteration

                    if self.visualise: visualiseTree(localTreeV, highlightNode=w, title="Blocktree with current w")
                    if self.print: print("  Block Node w: ", w)

                    if w not in self.spanningTreesDict.keys(): localTreesW = set()
                    else: localTreesW = self.spanningTreesDict[w]

                    self.getCharacteristics(v, localTreeV, w, localTreesW)

                    finalIsoTriple = ((0, 0), tuple(localTreeV.edges()), self.rG)

                    if finalIsoTriple in self.characteristics:
                        subIsoTime = time.time() - subIsoTime
                        if self.print or self.time:
                            print(f"{finalIsoTriple} found in {subIsoTime} seconds.")
                        return True, self.characteristics  # Ensure returning a tuple (True, characteristics)

        subIsoTime = time.time() - subIsoTime
        if self.print or self.time: print(f"No Subgraph Isomorphism found in {subIsoTime} seconds.")
        return False, None

    def getCharacteristics(self, v, localTreeV, w, localTreesW):

        gluedTrees = set()
        childrenInTreeV = set(localTreeV.successors(w))

        if v != w:
            for localTreeW in localTreesW:

                gluedTree = (self.glueLocalSpanningTrees(localTreeV, localTreeW), localTreeW)  # Saving as Tuple
                gluedTrees.add(gluedTree)

        if len(gluedTrees) == 0:
            localTreeW = nx.DiGraph()
            gluedTrees.add((localTreeV, localTreeW))

        for gluedTree in gluedTrees:

            if gluedTree[1] is not None:
                localTreeW = gluedTree[1]
                childrenInTreeW = set(localTreeW.successors(w)) if localTreeW.has_node(w) else set()
                childrenInBothTrees = childrenInTreeV.union(childrenInTreeW)

            else:
                childrenInTreeW = set()
                childrenInBothTrees = childrenInTreeV

            if 0 not in self.H.nodes():
                print("Zero not found: ",self.H.edges())
            for u in list(nx.dfs_postorder_nodes(self.H, source=0)):

                if self.visualise: visualiseTree(self.H, highlightNode=u, title="Pattern with current u")
                if self.print: print("      Pattern Node u:", u)
                neighbors = set(nx.all_neighbors(self.H, u))

                neighborsList = list(neighbors)
                childrenList = list(childrenInBothTrees)

                edges = []
                for c in childrenInBothTrees:
                    for neighbor in neighborsList:
                        isoTripleV = ((u, neighbor), tuple(localTreeV.edges()), c)
                        isoTripleW = ((u, neighbor), tuple(localTreeW.edges()), c)

                        if c in childrenInTreeV and isoTripleV in self.characteristics:
                            edges.append((neighbor, c))

                        elif c in childrenInTreeW and isoTripleW in self.characteristics:
                            edges.append((neighbor, c))

                currentIsoTriple = ((u, u), tuple(localTreeV.edges()), w)
                if currentIsoTriple not in self.characteristics:
                    if self.neighborCoveringMatching(neighborsList, childrenList, edges):
                        self.numIsoTriples += 1
                        if self.print: print(f"          {self.numIsoTriples}. {((u, u), localTreeV.edges(), w)}")
                        self.characteristics.add(currentIsoTriple)

                for neighbor in neighbors:
                    currentIsoTriple = ((neighbor, u), tuple(localTreeV.edges()), w)
                    if currentIsoTriple not in self.characteristics:
                        neighborsWithoutY = neighbors - {neighbor}
                        neighborsWithoutYList = list(neighborsWithoutY)

                        if self.neighborCoveringMatching(neighborsWithoutYList, childrenList, edges):
                            self.numIsoTriples += 1
                            if self.print: print(f"          {self.numIsoTriples}. {((neighbor, u), localTreeV.edges(), w)}")
                            self.characteristics.add(currentIsoTriple)



    def glueLocalSpanningTrees(self, localTreeV, localTreeW):

        if localTreeW.edges() == localTreeV.edges():
            return localTreeV

        gluedTree = nx.Graph()

        # Add all nodes and edges from the first tree
        gluedTree.add_nodes_from(localTreeV.nodes(data=True))
        gluedTree.add_edges_from(localTreeV.edges(data=True))

        # Add all nodes and edges from the second tree
        gluedTree.add_nodes_from(localTreeW.nodes(data=True))
        gluedTree.add_edges_from(localTreeW.edges(data=True))

        return gluedTree

    def neighborCoveringMatching(self, neighbors, children, edges):

        # Create adjacency list for the bipartite graph
        graph = {neighbor: [] for neighbor in neighbors}
        for neighbor, child in edges:
            if neighbor in graph:
                graph[neighbor].append(child)

        matchedChildren = {b: None for b in children}

        def dfs(neighbor, visited):
            for child in graph[neighbor]:
                if child not in visited:
                    visited.add(child)
                    # Check if 'child' is neighbor key in matchedChildren before accessing it
                    if child not in matchedChildren or matchedChildren[child] is None or dfs(matchedChildren[child], visited):
                        matchedChildren[child] = neighbor
                        return True
            return False

        # Try to find neighbor matching for each element in neighbors
        for neighbor in neighbors:
            visited = set()
            if not dfs(neighbor, visited):
                return False

        return True


