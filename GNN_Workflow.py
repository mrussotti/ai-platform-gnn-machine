#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase

###############################################################################
#                            Neo4j Queries
###############################################################################
def connect_to_neo4j(uri=None, username=None, password=None):
    # Update these credentials or read from environment
    HARDCODED_URI = "neo4j+s://09ed30ec.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "N7azcFeea5x3mkCafrBbdjAbptvfVW4RVyKUdtsie70"
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def _get_incident_transcripts(tx):
    """
    Retrieve pairs of incident nature and transcript text.
    
    This query matches Incident nodes with a CONTAINS relationship to
    Transcript nodes and returns the 'nature' property (the label you wish
    to predict) along with the transcript text (the model input).
    """
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(t:Transcript)
    RETURN i.nature AS nature, t.TEXT AS transcript
    """
    result = tx.run(query)
    # Return a list of dictionaries
    return [record.data() for record in result]

def extract_training_data(driver):
    """
    Extract training data from the graph.
    
    Returns a DataFrame with two columns:
        - 'nature': the target label (e.g., "Chest Pain")
        - 'transcript': the free-text transcript from the incident.
    """
    with driver.session() as session:
        data = session.execute_read(_get_incident_transcripts)
        training_df = pd.DataFrame(data)
        
        # Debug output: show the columns and a sample of rows
        print("Training DataFrame columns:", training_df.columns)
        print("First few rows of training data:")
        print(training_df.head())
        
        # Optionally, drop rows with missing values
        training_df.dropna(subset=['nature', 'transcript'], inplace=True)
        return training_df

###############################################################################
#                           Main Workflow
###############################################################################
def main():
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Incident and Transcript Data for Training ===")
    training_df = extract_training_data(driver)
    print(f"[INFO] Retrieved {len(training_df)} training records.")
    
    # Write the training data to an Excel file for inspection (checkpoint)
    excel_file = "training_data_checkpoint.xlsx"
    training_df.to_excel(excel_file, index=False)
    print(f"[INFO] Training data written to '{excel_file}' for verification.")
    
    # At this point, 'training_df' contains the incident 'nature' and its corresponding 'transcript'.
    # You can now proceed to pre-process the transcript text and train your ML model.
    
    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()
