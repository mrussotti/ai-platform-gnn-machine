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
        node_data["id"] = record["id"]
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
    return nodes_df, relationships_df

###############################################################################
#                           Prep Data
###############################################################################

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

    print(f"how to prepare these dataframes for NN training")



   
    print("\n=== Workflow Complete ===")
    driver.close()


if __name__ == "__main__":
    main()
