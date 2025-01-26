#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib
import ast
import numpy as np
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from pandas.api.types import CategoricalDtype

# xgboost (unused here, but left for reference)
# from xgboost import XGBRegressor

# PyTorch Geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

###############################################################################
#                           Custom Transformer
###############################################################################
class TopKCategories(TransformerMixin, BaseEstimator):
    """
    Custom transformer to keep only the top K categories for each categorical feature.
    Categories outside the top K are replaced with 'Other'.
    """
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.top_categories_ = []
    
    def fit(self, X, y=None):
        self.top_categories_ = []
        for i in range(X.shape[1]):
            # Convert everything to str
            col = X[:, i].astype(str)
            unique, counts = np.unique(col, return_counts=True)
            # Keep top_k categories by frequency
            top_k_cats = unique[np.argsort(counts)[-self.top_k:]]
            self.top_categories_.append(set(top_k_cats))
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            col_as_str = X_transformed[:, i].astype(str)
            # Replace anything not in top_k with 'Other'
            X_transformed[:, i] = np.where(
                np.isin(col_as_str, list(self.top_categories_[i])),
                col_as_str,
                'Other'
            )
        return X_transformed


###############################################################################
#                           Neo4j Queries
###############################################################################
def connect_to_neo4j(uri=None, username=None, password=None):
    # Update these credentials or read from environment or config
    HARDCODED_URI = "neo4j+s://d3f09fae.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "eYDPD1Qd_nn9HZYTKY7bKDHgRgbKF5YBgasio3TeaNs"
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def _get_nodes(tx):
    # Use ID(n) as 'id' instead of n.tmdbId
    query = """
    MATCH (n)
    RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties
    """
    result = tx.run(query)
    nodes = []
    for record in result:
        node_data = record["properties"]  # dictionary of n's properties
        node_data["id"] = record["id"]
        node_data["labels"] = record["labels"]
        nodes.append(node_data)
    return nodes

def get_nodes(driver):
    with driver.session() as session:
        return session.read_transaction(_get_nodes)

def _get_relationships(tx):
    # Similarly, use ID(r) for the relationship ID, and ID(n), ID(m) for start/end
    query = """
    MATCH (n)-[r]->(m)
    RETURN ID(r) AS id, ID(n) AS start_id, ID(m) AS end_id, type(r) AS type, properties(r) AS properties
    """
    result = tx.run(query)
    relationships = []
    for record in result:
        rel_data = record["properties"]  # dictionary of r's properties
        rel_data["id"] = record["id"]
        rel_data["start_id"] = record["start_id"]
        rel_data["end_id"] = record["end_id"]
        rel_data["type"] = record["type"]
        relationships.append(rel_data)
    return relationships

def get_relationships(driver):
    with driver.session() as session:
        return session.read_transaction(_get_relationships)

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    nodes_df = pd.DataFrame(nodes)
    relationships_df = pd.DataFrame(relationships)

    print("Nodes DataFrame columns:", nodes_df.columns)
    print("First few rows of nodes_df:")
    print(nodes_df.head())

    # Drop 'labels' and 'properties' if you don't need them as columns
    if 'labels' in nodes_df.columns:
        nodes_df.drop(columns='labels', inplace=True)
    if 'properties' in nodes_df.columns:
        nodes_df.drop(columns='properties', inplace=True)
    if 'properties' in relationships_df.columns:
        relationships_df.drop(columns='properties', inplace=True)

    return nodes_df, relationships_df


###############################################################################
#                           Data Cleaning and Preprocessing
###############################################################################
def detect_feature_type(series):
    """
    Detect the type of a feature (numerical, categorical, or other).
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    elif isinstance(series.dtype, CategoricalDtype) or pd.api.types.is_string_dtype(series):
        return "categorical"
    else:
        return "other"

def preprocess_data(nodes_df, relationships_df):
    """
    Preprocess nodes and relationships data.
    - Fill missing values
    - Convert categorical features to one-hot
    - Scale numerical columns
    """
    # 1. Handle missing values in nodes
    for col in nodes_df.columns:
        if col == "id":
            continue  # don't alter the ID column
        ftype = detect_feature_type(nodes_df[col])
        if ftype == "numerical":
            mean_val = nodes_df[col].mean()
            nodes_df[col] = nodes_df[col].fillna(mean_val)
        elif ftype == "categorical":
            if not nodes_df[col].mode().empty:
                mode_val = nodes_df[col].mode()[0]
                nodes_df[col] = nodes_df[col].fillna(mode_val)
            else:
                nodes_df[col] = nodes_df[col].fillna("unknown")
        else:
            # Convert to string, fill with 'unknown'
            nodes_df[col] = nodes_df[col].astype(str)
            nodes_df[col] = nodes_df[col].fillna("unknown")

    # 2. Handle missing values in relationships
    for col in relationships_df.columns:
        if col in ["id", "start_id", "end_id"]:
            continue
        ftype = detect_feature_type(relationships_df[col])
        if ftype == "numerical":
            mean_val = relationships_df[col].mean()
            relationships_df[col] = relationships_df[col].fillna(mean_val)
        elif ftype == "categorical":
            if not relationships_df[col].mode().empty:
                mode_val = relationships_df[col].mode()[0]
                relationships_df[col] = relationships_df[col].fillna(mode_val)
            else:
                relationships_df[col] = relationships_df[col].fillna("unknown")
        else:
            relationships_df[col] = relationships_df[col].astype(str)
            relationships_df[col] = relationships_df[col].fillna("unknown")

    # 3. Convert categorical data to numeric
    # For nodes
    categorical_cols_nodes = [
        c for c in nodes_df.columns
        if c != "id" and detect_feature_type(nodes_df[c]) == "categorical"
    ]
    numerical_cols_nodes = [
        c for c in nodes_df.columns
        if c != "id" and detect_feature_type(nodes_df[c]) == "numerical"
    ]

    if categorical_cols_nodes:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(nodes_df[categorical_cols_nodes])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(categorical_cols_nodes),
            index=nodes_df.index
        )
        # Drop original cat columns and attach encoded
        nodes_df = pd.concat([nodes_df.drop(categorical_cols_nodes, axis=1), encoded_df], axis=1)

    if numerical_cols_nodes:
        scaler = StandardScaler()
        nodes_df[numerical_cols_nodes] = scaler.fit_transform(nodes_df[numerical_cols_nodes])

    # For relationships
    categorical_cols_rel = [
        c for c in relationships_df.columns
        if c not in ["id", "start_id", "end_id"] and detect_feature_type(relationships_df[c]) == "categorical"
    ]
    numerical_cols_rel = [
        c for c in relationships_df.columns
        if c not in ["id", "start_id", "end_id"] and detect_feature_type(relationships_df[c]) == "numerical"
    ]

    if categorical_cols_rel:
        ohe_rel = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_rel = ohe_rel.fit_transform(relationships_df[categorical_cols_rel])
        encoded_rel_df = pd.DataFrame(
            encoded_rel,
            columns=ohe_rel.get_feature_names_out(categorical_cols_rel),
            index=relationships_df.index
        )
        relationships_df = pd.concat([relationships_df.drop(categorical_cols_rel, axis=1), encoded_rel_df], axis=1)

    if numerical_cols_rel:
        scaler_rel = StandardScaler()
        relationships_df[numerical_cols_rel] = scaler_rel.fit_transform(relationships_df[numerical_cols_rel])

    return nodes_df, relationships_df


###############################################################################
#                           Identify Features and Target
###############################################################################
def identify_features_and_target(nodes_df):
    """
    Interactive selection of target and features.
    """
    print("\n=== Step 4: Identify Features and Target ===")

    def group_columns_by_type(columns):
        grouped = {}
        for col in columns:
            if "_" in col:
                prefix = col.split("_")[0]
                grouped.setdefault(prefix, []).append(col)
            else:
                grouped.setdefault(col, []).append(col)
        return grouped

    grouped_cols = group_columns_by_type(nodes_df.columns)

    print("\nAvailable attribute types:")
    for attr_type, cols in grouped_cols.items():
        print(f" - {attr_type} ({len(cols)} columns)")

    # Prompt user for which attribute group to infer
    while True:
        attr_type = input("\nEnter the type of attribute to infer (e.g. 'occupation'): ").strip()
        if attr_type in grouped_cols:
            break
        else:
            print(f"Error: '{attr_type}' not in grouped columns.")

    print(f"\nColumns of type '{attr_type}':")
    for col in grouped_cols[attr_type]:
        print(f" - {col}")

    while True:
        target = input(f"\nEnter the specific column from '{attr_type}' to predict (e.g. '{attr_type}_unknown'): ").strip()
        if target in grouped_cols[attr_type]:
            break
        else:
            print(f"Error: '{target}' is not valid under '{attr_type}' group.")

    # Prompt user to select features
    while True:
        features_input = input("\nEnter features to use (comma-separated) or 'all': ").strip()
        if features_input.lower() == "all":
            features = list(nodes_df.columns)
            break
        else:
            features = [f.strip() for f in features_input.split(",")]
            invalid = [f for f in features if f not in nodes_df.columns]
            if invalid:
                print(f"Invalid columns: {invalid}")
            else:
                break

    print(f"\n[INFO] Target attribute: {target}")
    print(f"[INFO] Features for training: {features}")
    return target, features


###############################################################################
#                           Create Graph Representation
###############################################################################
def create_graph_representation(nodes_df, relationships_df, target, features):
    print("\n=== Step 5: Create Graph Representation ===")

    if 'id' not in nodes_df.columns:
        raise ValueError("No 'id' column found in nodes_df after preprocessing. Cannot create graph.")

    # Map node IDs to indices
    id_to_idx = {node_id: i for i, node_id in enumerate(nodes_df["id"])}

    # Remove 'id' from features if present
    if 'id' in features:
        features.remove('id')

    # Build node features
    node_features = nodes_df[features].values
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Build edge_index
    edge_list = []
    for _, row in relationships_df.iterrows():
        start = row["start_id"]
        end = row["end_id"]
        if start in id_to_idx and end in id_to_idx:
            edge_list.append([id_to_idx[start], id_to_idx[end]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Build target
    y_values = nodes_df[target].values
    # Convert to float
    y_values = torch.tensor(y_values, dtype=torch.float)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=y_values
    )
    print("[INFO] Graph representation created.")
    return data


###############################################################################
#                           GNN Architecture
###############################################################################
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # For binary classification with BCEWithLogitsLoss, we return raw logits (no sigmoid here)
        return x


###############################################################################
#                           Split Data
###############################################################################
def split_data(data, test_size=0.2, random_state=42):
    num_nodes = data.num_nodes
    idx_all = torch.arange(num_nodes)
    train_idx, test_idx = train_test_split(idx_all, test_size=test_size, random_state=random_state)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


###############################################################################
#                           Training Loop
###############################################################################
def train_model(model, graph_data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)

        # shape handling for single-output
        # if output_dim == 1, out is [N,1], we can squeeze to [N]
        if out.shape[1] == 1:
            out = out.view(-1)

        # same for y
        y = graph_data.y
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.view(-1)

        loss = criterion(out[graph_data.train_mask], y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


###############################################################################
#                           Evaluation
###############################################################################
def evaluate_model(model, graph_data, criterion, output_file="inferences.txt", driver=None, target=None):
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        if out.shape[1] == 1:
            out = out.view(-1)
        y_true = graph_data.y
        if y_true.dim() > 1 and y_true.shape[1] == 1:
            y_true = y_true.view(-1)

        test_out = out[graph_data.test_mask]
        test_y = y_true[graph_data.test_mask]

        loss = criterion(test_out, test_y)
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            # Probability via sigmoid
            probs = torch.sigmoid(test_out)
            preds = (probs >= 0.5).float()
            acc = accuracy_score(test_y.cpu(), preds.cpu())
            f1 = f1_score(test_y.cpu(), preds.cpu())
            print(f"Test Loss: {loss.item():.6f}, Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")
            
            # Ensure the output directory exists (only if output_file contains a directory path)
            output_dir = os.path.dirname(output_file)
            if output_dir:  # Only create the directory if output_dir is not empty
                os.makedirs(output_dir, exist_ok=True)
            
            # Log to file
            try:
                with open(output_file, "w") as f:
                    f.write("=== Model Evaluation (Binary Classification) ===\n")
                    f.write(f"Test Loss: {loss.item():.6f}\n")
                    f.write(f"Test Accuracy: {acc:.4f}\n")
                    f.write(f"Test F1: {f1:.4f}\n\n")
                    f.write("=== Predictions ===\n")
                    idx_test_nodes = graph_data.test_mask.nonzero().view(-1)
                    for i, node_idx in enumerate(idx_test_nodes):
                        f.write(f"Node {node_idx.item()}: ")
                        f.write(f"True Label = {test_y[i].item():.0f}, ")
                        f.write(f"Predicted Prob = {probs[i].item():.4f}, Predicted Class = {preds[i].item():.0f}\n")
                print(f"[INFO] Inferences logged to {output_file}.")
            except Exception as e:
                print(f"[ERROR] Failed to write to {output_file}: {e}")

            # Print updates to Neo4j (without actually applying them)
            if driver is not None and target is not None:
                node_ids = graph_data.test_mask.nonzero().view(-1).tolist()
                inferred_labels = preds.tolist()
                update_neo4j_with_inferred_labels(driver, node_ids, inferred_labels, target)

        elif isinstance(criterion, torch.nn.CrossEntropyLoss):
            preds = test_out.argmax(dim=1)
            acc = accuracy_score(test_y.cpu(), preds.cpu())
            print(f"Test Loss: {loss.item():.6f}, Test Accuracy: {acc:.4f}")
            
            # Print updates to Neo4j (without actually applying them)
            if driver is not None and target is not None:
                node_ids = graph_data.test_mask.nonzero().view(-1).tolist()
                inferred_labels = preds.tolist()
                update_neo4j_with_inferred_labels(driver, node_ids, inferred_labels, target)

        else:
            rmse = mean_squared_error(test_y.cpu(), test_out.cpu(), squared=False)
            print(f"Test Loss: {loss.item():.6f}, Test RMSE: {rmse:.4f}")

###############################################################################
#                           Update inferences
###############################################################################
def update_neo4j_with_inferred_labels(driver, node_ids, inferred_labels, target):
    """
    Print the updates that would be made to the Neo4j database with inferred labels.
    
    :param driver: Neo4j driver instance
    :param node_ids: List of node IDs
    :param inferred_labels: List of inferred labels
    :param target: Target attribute to update
    """
    print("\n=== Updates to Neo4j Database (Preview) ===")
    for node_id, label in zip(node_ids, inferred_labels):
        print(f"MATCH (n) WHERE ID(n) = {node_id} SET n.{target} = {label};")
    print(f"[INFO] {len(node_ids)} nodes would be updated with inferred labels for attribute '{target}'.")


###############################################################################
#                           Main Workflow
###############################################################################
def main():
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Nodes and Relationships Data ===")
    nodes_df, relationships_df = extract_data(driver)
    print(f"[INFO] Retrieved {len(nodes_df)} nodes and {len(relationships_df)} relationships.")

    print("\n=== Step 3: Data Cleaning and Preprocessing ===")
    nodes_df, relationships_df = preprocess_data(nodes_df, relationships_df)
    print("[INFO] Data cleaned and preprocessed.")

    print("\n=== Step 4: Identify Features and Target ===")
    target, features = identify_features_and_target(nodes_df)

    print("\n=== Step 5: Create Graph Representation ===")
    graph_data = create_graph_representation(nodes_df, relationships_df, target, features)

    print("\n=== Step 6: Choose a GNN Architecture ===")
    input_dim = graph_data.num_node_features
    hidden_dim = 16
    if graph_data.y.dim() == 1:
        output_dim = 1
    else:
        output_dim = graph_data.y.size(1)

    model = GCN(input_dim, hidden_dim, output_dim)
    print("[INFO] GCN model created.")

    print("\n=== Step 7: Train-Test Split ===")
    graph_data = split_data(graph_data, test_size=0.2, random_state=42)
    print("[INFO] Data split into training and testing sets.")

    print("\n=== Step 8: Model Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if output_dim == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_model(model, graph_data, optimizer, criterion, epochs=100)
    print("[INFO] Model training complete.")

    print("\n=== Step 9: Model Evaluation ===")
    evaluate_model(model, graph_data, criterion, driver=driver, target=target)
    print("[INFO] Model evaluation complete.")

    nodes_df.to_excel("nodes_preprocessed.xlsx", index=False)
    relationships_df.to_excel("relationships_preprocessed.xlsx", index=False)
    print("[INFO] Preprocessed data exported.")

    print("\n=== Workflow Complete ===")
    driver.close()


if __name__ == "__main__":
    main()