#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase


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
#                           Main Workflow
###############################################################################
def main():
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Nodes and Relationships Data ===")
    nodes_df, relationships_df = extract_data(driver)
    print(f"[INFO] Retrieved {len(nodes_df)} nodes and {len(relationships_df)} relationships.")

    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()

