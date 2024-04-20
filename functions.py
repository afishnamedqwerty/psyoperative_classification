import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx
from datetime import datetime
from random import choice

# Load entity data
def load_entities(entity_files):
    entities = {}
    for file in entity_files:
        with open(file, 'r') as f:
            entities[file] = json.load(f)
    return entities

# Feature extraction for users
def extract_user_features(users):
    features = []
    for user in users.itertuples():
        account_age_days = (datetime.now() - datetime.strptime(user.created_at, '%Y-%m-%d')).days
        tweet_count = max(user.tweet_count, 1)
        features.append([
            user.tweet_count / account_age_days,
            user.retweet_count / tweet_count,
            user.mention_count / tweet_count,
            user.hashtag_count / tweet_count,
            user.url_count / tweet_count,
            user.sensitive_tweet_count / tweet_count,
            account_age_days,
            int(user.verified),
            int(user.description != '') + int(user.location != '') + int(user.profile_image_url != ''),
            user.followers_count / max(user.following_count, 1)
        ])
    return features

# Dynamic Subgraph Sampling
def dynamic_subgraph_sampling(graph, root_node, walk_length=30, num_walks=20):
    subgraphs = []
    for _ in range(num_walks):
        subgraph_nodes = set([root_node])
        current_node = root_node
        for _ in range(walk_length):
            neighbors = list(graph.neighbors(current_node))
            if neighbors:
                next_node = choice(neighbors)
                subgraph_nodes.add(next_node)
                current_node = next_node
            else:
                break
        induced_subgraph = graph.subgraph(subgraph_nodes).copy()
        subgraphs.append(induced_subgraph)
    return subgraphs

# Model Definition
class EnhancedGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GATConv(2 * out_channels, out_channels, heads=3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)

class GraphBotDetector(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(GraphBotDetector, self).__init__()
        self.feature_extractor = EnhancedGCN(feature_dim, 128)
        self.pool = global_mean_pool
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, batch_data):
        node_features = self.feature_extractor(batch_data.x, batch_data.edge_index)
        graph_features = self.pool(node_features, batch_data.batch)
        return self.fc(graph_features)

# Training and Evaluation
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Implement the visualization of results as specified
