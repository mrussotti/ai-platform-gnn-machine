import json
import ast
from neo4j import GraphDatabase

## THIS SCRIPT PULLS ALL 'Incident' NODES FROM NEO4J WITH PARAMETERS ("summary","transcript","metadata")

# Configure your Neo4j connection parameters here
NEO4J_URI = "neo4j+ssc://2b9a3029.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "4GGm6B1aeQdjWyFxs7MpVCtqc8xZU0aueO_kqeAwXto"  # Replace with your actual password

# Create the Neo4j driver instance
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_nodes_with_properties(label, property_keys):
    """
    Retrieve nodes of a certain label and return only specific properties.

    :param label: The label of the nodes you want to query.
    :param property_keys: A list of property keys to retrieve.
    :return: A list of dictionaries representing each node's selected properties.
    """
    nodes = []
    
    # Open a session with Neo4j
    with driver.session() as session:
        # Build a Cypher query to return only the specified properties from nodes of the given label.
        # For example, if label is 'Person' and properties are ['name', 'age'], the query becomes:
        # MATCH (n:Person) RETURN n.name AS name, n.age AS age
        return_clause = ", ".join([f"n.{key} AS {key}" for key in property_keys])
        query = f"MATCH (n:{label}) RETURN {return_clause}"
        
        result = session.run(query)
        # Iterate through each record in the result and build a dictionary with only the requested keys.
        for record in result:
            node_data = {key: record.get(key) for key in property_keys}
            node_data['metadata'] = ast.literal_eval(node_data['metadata'])
            node_data['date'] = node_data["metadata"].get('Date_target')
            node_data['address'] = node_data["metadata"].get('clean_address_extracted')
 
            nodes.append(node_data)
    
    return nodes

if __name__ == "__main__":
    # Define the label and properties you want to query
    node_label = "Incident"              # Change this to your node's label
    properties = ["summary","transcript","metadata"]       # Replace with the property keys you need
    
    # Execute the function to get nodes with only the selected properties
    data = get_nodes_with_properties(node_label, properties)
    
    print(data)
    # Convert the result to a formatted JSON string and print it
    json_output = json.dumps(data, indent=2)
    print(json_output)
    
    filename = "data_pull.json"
    with open(filename, 'w') as file:
        json.dump(json_output, file, indent=4)
    # Close the driver when finished
    driver.close()
