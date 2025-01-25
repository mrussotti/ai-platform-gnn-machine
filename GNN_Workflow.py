#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib
import ast
import numpy as np

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator

# For classification fallback
from sklearn.linear_model import LogisticRegression

# xgboost
from xgboost import XGBRegressor

# PyTorch Geometric
from torch_geometric.data import Data

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
            # Convert everything to str, ensuring no list-like objects remain
            col = X[:, i].astype(str)
            unique, counts = np.unique(col, return_counts=True)
            # Keep top_k categories
            top_k = unique[np.argsort(counts)[-self.top_k:]]
            self.top_categories_.append(set(top_k))
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
#                            Neo4j Queries
###############################################################################
def connect_to_neo4j(uri=None, username=None, password=None):
    # Update these credentials or read from environment
    HARDCODED_URI = "neo4j+s://8cd5bbe1.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "dobMHdjo7r0g70Oz_HKZy-qcyP2PY6UN8yxu_rb82fo"
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def _get_nodes(tx):
    query = """
    MATCH (n)
    RETURN n.tmdbId AS id, labels(n) AS labels, properties(n) AS properties
    """
    result = tx.run(query)
    nodes = []
    for record in result:
        node_data = record["properties"]
        node_data["id"] = record["id"]  # Ensure the ID is stored as "id"
        node_data["labels"] = record["labels"]
        nodes.append(node_data)
    return nodes

def get_nodes(driver):
    with driver.session() as session:
        return session.execute_read(_get_nodes)

def _get_relationships(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN r.tmdbId AS id, n.tmdbId AS start_id, m.tmdbId AS end_id, type(r) AS type, properties(r) AS properties
    """
    result = tx.run(query)
    relationships = []
    for record in result:
        rel_data = record["properties"]
        rel_data["id"] = record["id"]
        rel_data["start_id"] = record["start_id"]
        rel_data["end_id"] = record["end_id"]
        rel_data["type"] = record["type"]
        relationships.append(rel_data)
    return relationships

def get_relationships(driver):
    with driver.session() as session:
        return session.execute_read(_get_relationships)

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    nodes_df = pd.DataFrame(nodes)
    relationships_df = pd.DataFrame(relationships)
    
    # Debugging: Print the columns and first few rows of nodes_df
    print("Nodes DataFrame columns:", nodes_df.columns)
    print("First few rows of nodes_df:")
    print(nodes_df.head())
    
    # If 'id' column is missing, generate one
    if 'id' not in nodes_df.columns:
        nodes_df['id'] = nodes_df.index
        print("[WARNING] 'id' column was missing. Generated using DataFrame index.")

    # ---------------------------------------------------------------------
    # DROP columns that are known to contain complex objects (e.g., lists/dicts)
    # You can customize this if you want to transform them instead.
    # ---------------------------------------------------------------------
    for col in ["labels", "properties"]:
        if col in nodes_df.columns:
            nodes_df.drop(columns=col, inplace=True)

    # For relationships, if 'properties' is a dict or list, remove if unneeded:
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
    # If all values can be numeric, treat as numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    # If it's a string or categorical
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return "categorical"
    else:
        # Return 'other' for columns that have lists, dicts, or any non-scalar objects
        return "other"

def preprocess_data(nodes_df, relationships_df):
    """
    Preprocess nodes and relationships data.
    """
    # Step 1: Handle missing values
    # For nodes DataFrame
    for col in nodes_df.columns:
        if col == "id":
            # Do NOT fill or alter the 'id' column
            continue

        feature_type = detect_feature_type(nodes_df[col])
        if feature_type == "numerical":
            nodes_df[col].fillna(nodes_df[col].mean(), inplace=True)
        elif feature_type == "categorical":
            # Fill missing with the mode or "unknown" if mode is empty
            if nodes_df[col].mode().size > 0:
                nodes_df[col].fillna(nodes_df[col].mode()[0], inplace=True)
            else:
                nodes_df[col].fillna("unknown", inplace=True)
        else:
            # For 'other', convert everything to string (so it becomes categorical)
            # Then fill NaN with 'unknown'
            nodes_df[col] = nodes_df[col].astype(str)
            nodes_df[col].fillna("unknown", inplace=True)

    # For relationships DataFrame
    for col in relationships_df.columns:
        if col in ["id", "start_id", "end_id"]:
            # Do NOT fill or alter these ID columns
            continue

        feature_type = detect_feature_type(relationships_df[col])
        if feature_type == "numerical":
            relationships_df[col].fillna(relationships_df[col].mean(), inplace=True)
        elif feature_type == "categorical":
            if relationships_df[col].mode().size > 0:
                relationships_df[col].fillna(relationships_df[col].mode()[0], inplace=True)
            else:
                relationships_df[col].fillna("unknown", inplace=True)
        else:
            relationships_df[col] = relationships_df[col].astype(str)
            relationships_df[col].fillna("unknown", inplace=True)

    # Step 2: Convert categorical data into numerical format
    # For nodes DataFrame
    # Exclude 'id' from transformations
    categorical_cols_nodes = [
        col for col in nodes_df.columns
        if detect_feature_type(nodes_df[col]) == "categorical" and col != "id"
    ]
    numerical_cols_nodes = [
        col for col in nodes_df.columns
        if detect_feature_type(nodes_df[col]) == "numerical" and col != "id"
    ]

    if categorical_cols_nodes:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_categorical = onehot_encoder.fit_transform(nodes_df[categorical_cols_nodes])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=onehot_encoder.get_feature_names_out(categorical_cols_nodes)
        )
        nodes_df = pd.concat([nodes_df.drop(categorical_cols_nodes, axis=1), encoded_categorical_df], axis=1)

    # For relationships DataFrame
    categorical_cols_relationships = [
        col for col in relationships_df.columns
        if detect_feature_type(relationships_df[col]) == "categorical"
           and col not in ["id", "start_id", "end_id"]
    ]
    numerical_cols_relationships = [
        col for col in relationships_df.columns
        if detect_feature_type(relationships_df[col]) == "numerical"
           and col not in ["id", "start_id", "end_id"]
    ]

    if categorical_cols_relationships:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_categorical = onehot_encoder.fit_transform(relationships_df[categorical_cols_relationships])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=onehot_encoder.get_feature_names_out(categorical_cols_relationships)
        )
        relationships_df = pd.concat([relationships_df.drop(categorical_cols_relationships, axis=1),
                                      encoded_categorical_df], axis=1)

    # Step 3: Normalize or standardize numerical features
    if numerical_cols_nodes:
        scaler = StandardScaler()
        nodes_df[numerical_cols_nodes] = scaler.fit_transform(nodes_df[numerical_cols_nodes])

    if numerical_cols_relationships:
        scaler = StandardScaler()
        relationships_df[numerical_cols_relationships] = scaler.fit_transform(relationships_df[numerical_cols_relationships])

    return nodes_df, relationships_df

###############################################################################
#                           Identify Features and Target
###############################################################################
def identify_features_and_target(nodes_df):
    """
    Prompt the user to specify the target attribute and features for training.
    Group columns by type and allow the user to select a type of attribute to infer.
    """
    print("\n=== Step 4: Identify Features and Target ===")

    def group_columns_by_type(columns):
        grouped = {}
        for col in columns:
            if "_" in col:
                prefix = col.split("_")[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(col)
            else:
                grouped.setdefault(col, []).append(col)
        return grouped

    grouped_columns = group_columns_by_type(nodes_df.columns)

    print("\nAvailable attribute types:")
    for attr_type, columns in grouped_columns.items():
        print(f"- {attr_type} ({len(columns)} columns)")

    while True:
        attr_type = input("\nEnter the type of attribute to infer (e.g., 'fax', 'phone', 'address'): ").strip()
        if attr_type in grouped_columns:
            break
        else:
            print(f"Error: '{attr_type}' is not a valid attribute type. Please choose from the list above.")

    print(f"\nColumns of type '{attr_type}':")
    for col in grouped_columns[attr_type]:
        print(f"- {col}")

    while True:
        target = input(f"\nEnter the specific {attr_type} attribute to infer (e.g., '{attr_type}_unknown'): ").strip()
        if target in grouped_columns[attr_type]:
            break
        else:
            print(f"Error: '{target}' is not a valid column of type '{attr_type}'.")

    while True:
        features = input("\nEnter the features to use for training (comma-separated, or type 'all' to use all columns): ").strip()
        if features.lower() == "all":
            features = nodes_df.columns.tolist()
            break
        else:
            features = [f.strip() for f in features.split(",")]
            invalid_features = [f for f in features if f not in nodes_df.columns]
            if not invalid_features:
                break
            else:
                print(f"Error: The following columns are not valid: {invalid_features}")

    print(f"\n[INFO] Target attribute: {target}")
    print(f"[INFO] Features for training: {features}")
    return target, features




###############################################################################
#                           Create Graph Representation
###############################################################################
def create_graph_representation(nodes_df, relationships_df, target, features):
    """
    Convert nodes and relationships into a graph representation using PyTorch Geometric.
    """
    print("\n=== Step 2.2: Create Graph Representation ===")

    # Ensure 'id' is present
    if 'id' not in nodes_df.columns:
        raise ValueError("No 'id' column found in nodes_df after preprocessing. Cannot create graph.")

    # Map node IDs to indices
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df["id"])}

    # Exclude 'id' from the feature vector if still present
    if 'id' in features:
        features.remove('id')

    # Build node features as float tensor
    node_features = nodes_df[features].values
    # If any column is still object dtype, this line will fail.
    # By here, all columns in `features` should be numeric (float/int).
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Build edge indices
    edge_indices = []
    for _, row in relationships_df.iterrows():
        if "start_id" in row and "end_id" in row:
            start_id = row["start_id"]
            end_id = row["end_id"]
            if start_id in node_id_to_idx and end_id in node_id_to_idx:
                edge_indices.append([node_id_to_idx[start_id], node_id_to_idx[end_id]])

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Build target labels
    target_labels = nodes_df[target].values
    target_labels = torch.tensor(target_labels, dtype=torch.float)

    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=node_features,
        edge_index=edge_indices,
        y=target_labels
    )

    print("[INFO] Graph representation created.")
    return graph_data


###############################################################################
#                           Feature Selection
###############################################################################
def select_features(nodes_df, target):
    """
    Select relevant features from the nodes DataFrame to use as input features for the GNN.
    Allows the user to choose features based on domain knowledge or feature importance.
    """
    print("\n=== Step 2.3: Feature Selection ===")

    # Exclude the target column from feature selection
    available_features = [col for col in nodes_df.columns if col != target]

    print("\nAvailable features for selection:")
    for i, feature in enumerate(available_features):
        print(f"{i + 1}. {feature}")

    while True:
        try:
            # Prompt the user to select features
            selected_indices = input(
                "\nEnter the indices of the features to use (comma-separated, e.g., '1,2,3'): "
            ).strip()
            selected_indices = [int(idx) - 1 for idx in selected_indices.split(",")]

            # Validate selected indices
            if all(0 <= idx < len(available_features) for idx in selected_indices):
                selected_features = [available_features[idx] for idx in selected_indices]
                break
            else:
                print("Error: One or more indices are invalid. Please try again.")
        except ValueError:
            print("Error: Invalid input. Please enter comma-separated indices.")

    print(f"\n[INFO] Selected features: {selected_features}")
    return selected_features

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

###############################################################################
#                           GNN Architecture
###############################################################################
class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x

###############################################################################
#                           Train-Test Split
###############################################################################
def split_data(graph_data, test_size=0.2, random_state=42):
    """
    Split the graph data into training and testing sets.
    """
    # Node-wise split
    num_nodes = graph_data.num_nodes
    indices = torch.arange(num_nodes)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    # Create masks for training and testing
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    graph_data.train_mask = train_mask
    graph_data.test_mask = test_mask

    return graph_data

###############################################################################
#                           Model Training
###############################################################################
def train_model(model, graph_data, optimizer, criterion, epochs=100):
    """
    Train the GNN model.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        out = model(graph_data.x, graph_data.edge_index)
        # Compute loss only on training nodes
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

###############################################################################
#                           Model Evaluation
###############################################################################
def evaluate_model(model, graph_data, criterion, output_file="inferences.txt"):
    """
    Evaluate the GNN model on the test set and log inferences to a text file.
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        out = model(graph_data.x, graph_data.edge_index)
        # Compute loss on test nodes
        loss = criterion(out[graph_data.test_mask], graph_data.y[graph_data.test_mask])
        # Compute predictions
        preds = out[graph_data.test_mask].argmax(dim=1) if criterion == F.cross_entropy else out[graph_data.test_mask]
        
        # Compute metrics
        if criterion == F.cross_entropy:
            accuracy = accuracy_score(graph_data.y[graph_data.test_mask].cpu(), preds.cpu())
            print(f"Test Loss: {loss.item()}, Test Accuracy: {accuracy}")
        else:
            rmse = mean_squared_error(graph_data.y[graph_data.test_mask].cpu(), preds.cpu(), squared=False)
            print(f"Test Loss: {loss.item()}, Test RMSE: {rmse}")

        # Log inferences to a text file
        with open(output_file, "w") as f:
            f.write("=== Model Evaluation ===\n")
            f.write(f"Test Loss: {loss.item()}\n")
            if criterion == F.cross_entropy:
                f.write(f"Test Accuracy: {accuracy}\n")
            else:
                f.write(f"Test RMSE: {rmse}\n")
            
            f.write("\n=== Predictions ===\n")
            for i in range(len(preds)):
                f.write(f"Node {graph_data.test_mask.nonzero()[i].item()}: ")
                f.write(f"True Label = {graph_data.y[graph_data.test_mask][i].item()}, ")
                f.write(f"Predicted Label = {preds[i].item()}\n")
        
        print(f"[INFO] Inferences logged to {output_file}.")

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

    print("\n=== Step 2.3: Feature Selection ===")
    selected_features = select_features(nodes_df, target)

    print("\n=== Step 2.2: Create Graph Representation ===")
    graph_data = create_graph_representation(nodes_df, relationships_df, target, selected_features)

    print("\n=== Step 3.1: Choose a GNN Architecture ===")
    input_dim = graph_data.x.size(1)  # Number of features per node
    hidden_dim = 16  # Hidden layer dimension
    output_dim = 1 if graph_data.y.dim() == 1 else graph_data.y.size(1)  # Output dimension
    model = GCN(input_dim, hidden_dim, output_dim)
    print("[INFO] GCN model created.")

    print("\n=== Step 3.2: Train-Test Split ===")
    graph_data = split_data(graph_data, test_size=0.2)
    print("[INFO] Data split into training and testing sets.")

    print("\n=== Step 3.3: Model Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss() if output_dim == 1 else F.cross_entropy  # Choose loss function
    train_model(model, graph_data, optimizer, criterion, epochs=100)
    print("[INFO] Model training complete.")

    print("\n=== Step 3.4: Model Evaluation ===")
    evaluate_model(model, graph_data, criterion)
    print("[INFO] Model evaluation complete.")

    # Export to Excel for debugging/further analysis
    nodes_df.to_excel("nodes_preprocessed.xlsx", index=False)
    relationships_df.to_excel("relationships_preprocessed.xlsx", index=False)
    print("[INFO] Preprocessed nodes and relationships exported to Excel files.")

    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()

