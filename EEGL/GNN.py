from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import torch
import sys


def loadDataset(G: nx.Graph, trainRatio: float= 0.8, valRatio: float=0.1):
    dataset = from_networkx(G)
    dataset.num_nodes = len(G.nodes)
    dataset.num_classes = len(set([G.nodes[node]['y'] for node in G.nodes]))
    dataset.num_node_features = len(G.nodes[0]['x'])

    # Shuffle and split the dataset into training, validation and test sets
    permutatedDataset = torch.randperm(dataset.num_nodes)
    trainingSetSize = int(dataset.num_nodes * trainRatio)
    validationSetSize = int(dataset.num_nodes * valRatio)

    dataset.train_mask = torch.full_like(dataset.y, False, dtype=bool)
    dataset.train_mask[permutatedDataset[:trainingSetSize]] = True
    dataset.val_mask = torch.full_like(dataset.y, False, dtype=torch.bool)
    dataset.val_mask[permutatedDataset[trainingSetSize:trainingSetSize + validationSetSize]] = True
    dataset.test_mask = torch.full_like(dataset.y, False, dtype=torch.bool)
    dataset.test_mask[permutatedDataset[trainingSetSize + validationSetSize:]] = True

    # Get the IDs of training, validation and test nodes
    dataset.train_index = dataset.train_mask.nonzero(as_tuple=False).view(-1)
    dataset.val_index = dataset.val_mask.nonzero(as_tuple=False).view(-1)
    dataset.test_index = dataset.test_mask.nonzero(as_tuple=False).view(-1)

    return dataset


def trainingStep(G, learningRate=0.01, weightDecay=5e-4, epochs=200, dropout=0.5, hiddenChannels=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = loadDataset(G)
    data = dataset.to(device)
    model = GCN(dataset, hiddenChannels=hiddenChannels, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)

    print(f'\n\n-- Training GNN Model --')
    print(f'With features {dataset.x.cpu()[:8, :5]}')

    model.train()
    for epoch in tqdm(range(epochs), file=sys.stdout):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model, data


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    modelPredictions = model(data.x, data.edge_index).argmax(dim=1)
    modelAccuracies = {}

    masks = {
        'train': data.train_mask,
        'val': data.val_mask,
        'test': data.test_mask
    }

    for key, mask in masks.items():
        modelAccuracies[key] = int((modelPredictions[mask] == data.y[mask]).sum()) / int(mask.sum())

    return modelAccuracies, modelPredictions


def printDetailedClassification(G, data, modelPredictions):

    # Print out each node and its corresponding true label, predicted label, and features
    for node, trueLabel, predictedLabel in zip(range(len(G)), data.y, modelPredictions):
        print(f"Node {node} - True Label: {trueLabel.item()}, Predicted Label: {predictedLabel.item()}")


class GCN(torch.nn.Module):
    def __init__(self, dataset, hiddenChannels=16, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GraphConv(dataset.num_node_features, hiddenChannels)
        self.conv2 = GraphConv(hiddenChannels, 2 * hiddenChannels)
        self.conv3 = GraphConv(2 * hiddenChannels, dataset.num_classes)

    def forward(self, x, edgeIndex):
        x = F.relu(self.conv1(x, edgeIndex))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edgeIndex))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edgeIndex)

        return F.log_softmax(x, dim=1)
