# workflow.py

import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib

def connect_to_neo4j(uri, username, password):
    """
    Connects to the Neo4j Graph Database.

    **Get from user:**
    - uri (str): The connection URI for the Neo4j database.
    - username (str): The username for authentication.
    - password (str): The password for authentication.

    **Outputs:**
    - driver (GraphDatabase.driver): An instance of the Neo4j driver for executing queries.
    """
    pass

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    return nodes, relationships

def identify_missing_attributes(nodes_df):
    """
    Identifies nodes that are missing specified attributes.

    **Inputs:**
    - nodes_df (pd.DataFrame): DataFrame containing node information.    
    
    **Get from user:**
    - present them with the list of available attributes from the nodes we pulled from the db. 
    make them select one (start with one) or more attributes for us to train models on. 

    **Outputs:**
    - missing_attributes_df (pd.DataFrame): DataFrame containing nodes with missing attributes.
    - key_attributes: a datatype that holds the list of attributes the user wants us inference
    """
    pass

def preprocess_data(nodes_df, relationships_df):
    """
    Preprocesses data for AI model training.

    **Inputs:**
    - nodes_df (pd.DataFrame): DataFrame containing node information.
    - relationships_df (pd.DataFrame): DataFrame containing relationship information.

    **Outputs:**
    - preprocessed_features (pd.DataFrame or torch.Tensor): Processed feature set ready for model training.
    - labels (pd.Series or torch.Tensor): Labels for supervised learning tasks.
    """
    pass

def train_ai_models(preprocessed_features, labels, key_attributes):
    """
    Trains one AI model for each attribute we want to find (start with one attribute).     
    We use classification architecture for discrete attributes and regression for continuos attributes

    **Inputs:**
    - preprocessed_features (pd.DataFrame or torch.Tensor): Processed feature set.
    - labels (pd.Series or torch.Tensor): Labels for supervised learning.
    - key_attributes: a datatype that holds the list of attributes the user wants us inference

    **Outputs:**
    - trained_models (dict): Dictionary containing trained models for each attribute.
    """
    pass

def infer_missing_attributes(trained_models, preprocessed_features, missing_attributes_df):
    """
    Uses trained AI models to infer missing attributes.

    **Inputs:**
    - trained_models (dict): Dictionary containing trained models.
    - preprocessed_features (pd.DataFrame or torch.Tensor): Features for nodes with missing attributes.
    - missing_attributes_df (pd.DataFrame): DataFrame containing nodes with missing attributes.

    **Outputs:**
    - inferred_attributes (dict): Dictionary containing inferred attribute values for each node.
    """
    pass

def generate_report(inferred_attributes):
    """
    Generates and presents a report of the inferred attributes to the user.

    **Inputs:**
    - inferred_attributes (dict): Dictionary containing inferred attribute values.

    **Outputs:**
    - None (saves the report to the specified path)
    """
    pass

def update_neo4j(driver, inferred_attributes):
    """
    Updates the Neo4j database with inferred attribute values only if the user wants to.

    **Inputs:**
    - driver (GraphDatabase.driver): The Neo4j driver instance.
    - inferred_attributes (dict): Dictionary containing inferred attribute values.

    **Ask the user:**
    - Ask the user if they would like to update the DB with these inferences. (one at a time or whole thing?)

    **Outputs:**
    - None
    """
    pass


def save_trained_models(trained_models):
    """
    Saves the trained AI models locally. 

    **Inputs:**
    - trained_models (dict): Dictionary containing trained models.

    **Outputs:**
    - None (models are saved to the specified directory)
    """
    pass

def main():
    """
    Main function to orchestrate the workflow.

    **Process Flow:**
    1. Connect to Neo4j Graph Database
    2. Extract Nodes and Relationships Data
    3. Identify The Attributes We Want to Inference and Find the Nodes with Missing Attributes
    4. Preprocess Data for AI Model
    5. Train AI Models for Missing Attributes
    6. Infer Missing Attributes Using Trained Models
    7. Generate and Present Inference Report to User
    8. Update Neo4j Database with Inferred Attributes
    9. Save Trained Models Locally
    """
    # Step 1: Connect to Neo4j Graph Database
    driver = connect_to_neo4j()

    # Step 2: Extract Nodes and Relationships Data
    nodes_df, relationships_df = extract_data(driver)

    # Step 3. Identify The Attributes We Want to Inference and Find the Nodes with Missing Attributes
    missing_attributes_df, key_attributes = identify_missing_attributes(nodes_df)

    # Step 4: Preprocess Data for AI Model
    preprocessed_features, labels = preprocess_data(nodes_df, relationships_df)

    # Step 5: Train AI Models for Missing Attributes
    trained_models = train_ai_models(preprocessed_features, labels, key_attributes)

    # Step 6: Infer Missing Attributes Using Trained Models
    inferred_attributes = infer_missing_attributes(trained_models, preprocessed_features, missing_attributes_df)

    # Step 7: Generate and Present Inference Report to User
    generate_report(inferred_attributes)

    # Step 8: Update Neo4j Database with Inferred Attributes if the User Wants to
    update_neo4j(driver, inferred_attributes)

    # Step 9: Save Trained Models Locally
    #save_trained_models(trained_models) hold off on this. need to see the usecase. are models disposable??

    # Close the Neo4j driver connection
    driver.close()

if __name__ == "__main__":
    main()

def get_nodes(tx):
    query = """
    MATCH (n)
    RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
    """
    result = tx.run(query)
    nodes = []
    for record in result:
        node_data = record["properties"]
        node_data["id"] = record["id"]
        node_data["labels"] = record["labels"]
        nodes.append(node_data)
    return nodes

def get_relationships(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN id(r) AS id, id(n) AS start_id, id(m) AS end_id, type(r) AS type, properties(r) AS properties
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