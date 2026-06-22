from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualiseBipartiteGraph(G, title=None):

    nodesFromG = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}

    color_map = []
    for node in G:
        if node in nodesFromG:
            color_map.append('green')
        else:
            color_map.append('gold')


    plt.figure(figsize=(12, 12))
    if title: plt.title(title, fontsize=15, fontweight='bold')
    pos = nx.drawing.layout.bipartite_layout(G, nodesFromG)
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=color_map, node_size=1500, font_size=12,
                     font_weight='bold', font_color="whitesmoke")

    green_patch = mpatches.Patch(color='green', label='Nodes from G')
    gold_patch = mpatches.Patch(color='gold', label='Nodes from H')
    plt.legend(handles=[green_patch, gold_patch])

    plt.show()


def visualiseTinG(T, G, title=None):
    combinedGraph = nx.compose(G, T)

    nodeColours = ['firebrick' if node in T.nodes else 'navy' for node in combinedGraph.nodes]

    edgeWidths = [5 if edge in T.edges else 1 for edge in combinedGraph.edges]

    plt.figure(figsize=(12, 12))
    if title: plt.title(title, fontsize=15, fontweight='bold')
    nx.draw_networkx(combinedGraph, with_labels=True, node_color=nodeColours, node_size=1500, font_size=12,
                     font_weight='bold', font_color="whitesmoke", width=edgeWidths)

    plt.show()


def visualiseGraph(G, title=None, highlightNode=None):

    nodeColors = []
    for node in G.nodes():
        if node == highlightNode:
            nodeColors.append('firebrick')
        else:
            nodeColors.append('navy')

    plt.figure(figsize=(12, 12))
    if title: plt.title(title, fontsize=15, fontweight='bold')
    nx.draw_networkx(G, with_labels=True, node_color=nodeColors, node_size=1500, font_size=12,
                     font_weight='bold', font_color="whitesmoke")

    plt.show()

def visualiseKamadaGraph(G, title=None, highlightNode=None, withLabels=False, colorByLabel=False):

    # Define the colormap from viridis
    cmap = cm.get_cmap('viridis')

    # Get node labels if 'colorByLabel' is True
    if colorByLabel:
        labels = [G.nodes[node]['y'] for node in G.nodes()]
        num_classes = len(set(labels))

        # Normalize labels to be in range [0, 1] for colormap
        norm = mcolors.Normalize(vmin=min(labels), vmax=max(labels))
        nodeColors = [cmap(norm(label)) for label in labels]
    else:
        # Default coloring, navy for non-highlighted and firebrick for highlighted node
        nodeColors = []
        for node in G.nodes():
            if node == highlightNode:
                nodeColors.append('firebrick')
            else:
                nodeColors.append('navy')

    # Set figure size
    plt.figure(figsize=(12, 12))

    # Add title if provided
    if title:
        plt.title(title, fontsize=15, fontweight='bold')

    # Kamada-Kawai layout for the graph
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph with the Kamada-Kawai layout
    nx.draw_networkx(G, pos, with_labels=withLabels, node_color=nodeColors, node_size=50, font_size=12,
                     font_weight='bold', font_color="whitesmoke")

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def visualiseTree(T, highlightNode=None, title=None):
    # Check if graph is empty
    if len(T.nodes()) == 0:
        print("Graph is empty. No nodes to visualize.")
        return

    # Define node colors
    nodeColors = ['firebrick' if node == highlightNode else 'navy' for node in T.nodes()]

    try:
        # Generate Graphviz layout
        pos = graphviz_layout(T, prog="dot")
    except Exception as e:
        print(f"Error in Graphviz layout generation: {e}")
        return

    # Plotting the graph
    plt.figure(figsize=(12, 12))
    if title:
        plt.title(title, fontsize=15, fontweight='bold')

    nx.draw(T, pos, node_color=nodeColors, with_labels=True, font_weight='bold',
            node_size=1500, font_size=12, font_color="whitesmoke")
    plt.show()

    return


def visualiseTopPatterns(graphs):
    plt.figure(figsize=(20, 10))  # Create a figure with enough space for 10 subplots

    for i, G in enumerate(graphs):
        plt.subplot(2, 5, i + 1)  # Create a 2x5 grid of subplots
        highlightNode = G.graph['root']  # Assume each graph has a 'root' attribute
        title = f"Label {G.graph['label']}"  # Assume each graph has a 'label' attribute

        # Set node colors
        nodeColors = []
        for node in G.nodes():
            if node == highlightNode:
                nodeColors.append('firebrick')  # Color for the root node
            else:
                nodeColors.append('navy')  # Color for other nodes
        pos = nx.kamada_kawai_layout(G)
        # Draw the graph
        nx.draw_networkx(G, pos, with_labels=False, node_color=nodeColors, node_size=1500,
                         font_size=12, font_weight='bold', font_color="whitesmoke")

        # Set the title for each subplot
        plt.title(title, fontsize=15, fontweight='bold')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


if __name__ == '__main__':
    # Example tree creation
    T = nx.DiGraph()  # Directed graph for the tree structure

    # Adding nodes and edges to the tree
    T.add_edges_from([(1, 2), (1, 3),  # Root node 1 with children 2 and 3
                      (2, 4), (2, 5),  # Node 2 with children 4 and 5
                      (3, 6), (3, 7)])  # Node 3 with children 6 and 7

    # Now we can call your visualiseTree function to visualize the tree
    visualiseTree(T, highlightNode=3, title="Binary Tree Example")
