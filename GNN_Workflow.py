import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib


def get_nodes(driver):
    with driver.session() as session:
        return session.read_transaction(_get_nodes)

def _get_nodes(tx):
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

def get_relationships(driver):
    with driver.session() as session:
        return session.read_transaction(_get_relationships)

def _get_relationships(tx):
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

def connect_to_neo4j(uri=None, username=None, password=None):
    # Hardcoded connection details
    HARDCODED_URI = "neo4j+s://ae86bd83.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "h0ZknFPHLkSPXFU_O0eEI1_VG-AnHa-p1uN6NfrNdFY"
    
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    # Convert them to DataFrames
    nodes_df = pd.DataFrame(nodes)
    relationships_df = pd.DataFrame(relationships)
    return nodes_df, relationships_df

def identify_missing_attributes(nodes_df):
    # List out the columns in the nodes_df (excluding id/labels if you like)
    available_attributes = list(nodes_df.columns)
    print("\nAvailable Attributes in nodes_df:")
    for attr in available_attributes:
        print(f"  - {attr}")

    # For simplicity, let's just have the user pick one attribute
    chosen_attribute = input(
        "\nEnter the attribute you want to model/infer (e.g. 'age' or 'status'): "
    )

    # If the user picks an invalid attribute, handle it gracefully
    if chosen_attribute not in available_attributes:
        print(
            f"[WARNING] '{chosen_attribute}' not found in the DataFrame. "
            "Please ensure your input matches one of the listed attributes.\n"
        )
        return pd.DataFrame(), []  # Return empty DataFrame and empty attribute list

    # Identify nodes that have that attribute missing (NaN or empty string)
    missing_mask = (nodes_df[chosen_attribute].isna()) | (nodes_df[chosen_attribute] == "")
    missing_attributes_df = nodes_df[missing_mask].copy()

    # Pack chosen attributes in a list
    key_attributes = [chosen_attribute]

    return missing_attributes_df, key_attributes

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
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Nodes and Relationships Data ===")
    nodes_df, relationships_df = extract_data(driver)
    print(f"[INFO] Retrieved {len(nodes_df)} nodes and {len(relationships_df)} relationships.")

    print("\n=== Step 3: Identify Missing Attributes ===")
    missing_attributes_df, key_attributes = identify_missing_attributes(nodes_df)
    print(f"[INFO] Chosen attributes: {key_attributes}")
    print(f"[INFO] Found {len(missing_attributes_df)} nodes missing values for those attributes.")

    # Step 4: Preprocess Data for AI Model
    print("\n=== Step 4: Preprocess Data ===")
    preprocessed_features, labels = preprocess_data(nodes_df, relationships_df)
    print("[INFO] Data preprocessed.")

    # Step 5: Train AI Models for Missing Attributes
    print("\n=== Step 5: Train AI Models ===")
    trained_models = train_ai_models(preprocessed_features, labels, key_attributes)
    print("[INFO] Models trained.")

    # Step 6: Infer Missing Attributes Using Trained Models
    print("\n=== Step 6: Inference ===")
    inferred_attributes = infer_missing_attributes(trained_models, preprocessed_features, missing_attributes_df)
    print("[INFO] Missing attributes inferred.")

    # Step 7: Generate and Present Inference Report to User
    print("\n=== Step 7: Generate Report ===")
    generate_report(inferred_attributes)
    print("[INFO] Inference report generated.")

    # Step 8: Update Neo4j Database with Inferred Attributes (if user wants)
    print("\n=== Step 8: Update Neo4j Database ===")
    update_neo4j(driver, inferred_attributes)
    print("[INFO] Neo4j updated (if requested by user).")

    # Close the driver
    print("\n=== Workflow Complete: Closing Neo4j Driver ===")
    driver.close()

if __name__ == "__main__":
    main()

