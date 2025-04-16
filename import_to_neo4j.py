import json
import os
from neo4j import GraphDatabase
import time
from tqdm import tqdm  # For progress bars

# Neo4j Connection Details
NEO4J_URI = "neo4j+ssc://4a946ffe.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "k-njZdN37CLEDE3p-g7YolnA78v2OiTGt6H2iILUH9o"  # Replace with your actual password

# Folder containing the batch files
BATCH_FOLDER = "attempt2"

# Get all JSON files in the folder
def get_json_files():
    json_files = []
    try:
        for file in os.listdir(BATCH_FOLDER):
            if file.endswith('.json') and file.startswith('batch_'):
                json_files.append(os.path.join(BATCH_FOLDER, file))
        print(f"Found {len(json_files)} batch files in folder '{BATCH_FOLDER}'")
        return json_files
    except Exception as e:
        print(f"Error accessing folder '{BATCH_FOLDER}': {str(e)}")
        return []

# This will be populated when the script runs
JSON_FILES = []

def connect_to_neo4j():
    """Establish connection to Neo4j database"""
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    # Test the connection
    with driver.session() as session:
        result = session.run("RETURN 'Connection successful' AS message").single()
        if result:
            print(f"✅ {result['message']}")
        else:
            print("⚠️ Connection established but no result returned")
    
    return driver

def create_call_graph(tx, data):
    """
    Create nodes and relationships for a single 911 call
    This follows the schema from the GNN_Workflow.py file
    """
    # Create Incident node
    incident_query = """
    CREATE (i:Incident {
        summary: $summary,
        timestamp: $timestamp,
        nature: $nature,
        severity: $severity,
        hazards: $hazards,
        transcript: $transcript
    })
    RETURN id(i) AS incident_id
    """
    incident_result = tx.run(
        incident_query,
        summary=data["incident"]["summary"],
        timestamp=data["incident"]["timestamp"],
        nature=data["incident"]["nature"],
        severity=data["incident"]["severity"],
        hazards=data["incident"]["hazards"],
        transcript=data["incident"]["transcript"]
    ).single()
    incident_id = incident_result["incident_id"]
    
    # Create Call nodes and relationships
    call_ids = []
    for call in data["calls"]:
        call_query = """
        CREATE (c:Call {
            summary: $summary,
            timestamp: $timestamp
        })
        RETURN id(c) AS call_id
        """
        call_result = tx.run(
            call_query, 
            summary=call["summary"],
            timestamp=call["timestamp"]
        ).single()
        call_id = call_result["call_id"]
        call_ids.append(call_id)
        
        # Create relationship between Call and Incident
        tx.run("""
        MATCH (c:Call), (i:Incident)
        WHERE id(c) = $call_id AND id(i) = $incident_id
        CREATE (c)-[:ABOUT]->(i)
        """, call_id=call_id, incident_id=incident_id)
    
    # Create Location node
    location_query = """
    CREATE (l:Location {
        address: $address,
        type: $type,
        features: $features,
        time: $time
    })
    RETURN id(l) AS location_id
    """
    location_result = tx.run(
        location_query,
        address=data["location"]["address"],
        type=data["location"]["type"],
        features=data["location"]["features"],
        time=data["location"]["time"]
    ).single()
    location_id = location_result["location_id"]
    
    # Create relationship between Incident and Location
    tx.run("""
    MATCH (i:Incident), (l:Location)
    WHERE id(i) = $incident_id AND id(l) = $location_id
    CREATE (i)-[:AT]->(l)
    """, incident_id=incident_id, location_id=location_id)
    
    # Create Person nodes and relationships
    person_ids = []
    for person in data["persons"]:
        if not any(person.values()):  # Skip if all values are empty
            continue
            
        person_query = """
        CREATE (p:Person {
            name: $name,
            phone: $phone,
            role: $role,
            relationship: $relationship,
            conditions: $conditions,
            age: $age,
            sex: $sex
        })
        RETURN id(p) AS person_id
        """
        person_result = tx.run(
            person_query,
            name=person["name"],
            phone=person["phone"],
            role=person["role"],
            relationship=person["relationship"],
            conditions=person["conditions"],
            age=person["age"],
            sex=person["sex"]
        ).single()
        person_id = person_result["person_id"]
        person_ids.append(person_id)
        
        # Create relationship between Person and Incident
        tx.run("""
        MATCH (p:Person), (i:Incident)
        WHERE id(p) = $person_id AND id(i) = $incident_id
        CREATE (p)-[:INVOLVED_IN]->(i)
        """, person_id=person_id, incident_id=incident_id)
        
        # If person is a caller, create relationship with Call
        if person["role"].lower() == "caller" and call_ids:
            # Connect to the first call by default
            tx.run("""
            MATCH (p:Person), (c:Call)
            WHERE id(p) = $person_id AND id(c) = $call_id
            CREATE (p)-[:MADE]->(c)
            """, person_id=person_id, call_id=call_ids[0])
    
    # Add metadata if available
    if "metadata" in data and data["metadata"]:
        metadata_query = """
        MATCH (i:Incident)
        WHERE id(i) = $incident_id
        SET i.metadata = $metadata
        """
        tx.run(metadata_query, incident_id=incident_id, metadata=str(data["metadata"]))
    
    return {
        "incident_id": incident_id,
        "call_ids": call_ids,
        "location_id": location_id,
        "person_ids": person_ids
    }

def save_911_call_to_neo4j(call_data, driver):
    """
    Save the extracted 911 call data to Neo4j.
    Creates all nodes and relationships according to the database schema.
    
    Parameters:
    - call_data: Dictionary containing structured data
    - driver: Neo4j driver connection
    
    Returns:
    - Dictionary with status information
    """
    try:
        with driver.session() as session:
            # Execute the transaction function
            result = session.execute_write(create_call_graph, call_data)
            
            return {
                "status": "success",
                "message": "911 call data successfully saved to Neo4j",
                "node_ids": result
            }
            
    except Exception as e:
        print(f"Error saving to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to save 911 call data: {str(e)}"
        }

def process_batch_file(file_path, driver):
    """Process a single batch JSON file"""
    try:
        with open(file_path, 'r') as f:
            batch_data = json.load(f)
        
        print(f"\n📂 Processing file: {file_path}")
        print(f"   Batch: {batch_data.get('batch', 'Unknown')} of {batch_data.get('total_batches', 'Unknown')}")
        print(f"   Records to process: {len(batch_data.get('results', []))}")
        
        results = batch_data.get("results", [])
        success_count = 0
        error_count = 0
        
        for i, record in enumerate(tqdm(results, desc="Processing records")):
            result = save_911_call_to_neo4j(record, driver)
            
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1
                print(f"⚠️ Error on record {i}: {result['message']}")
            
            # Sleep briefly to avoid overwhelming the database
            time.sleep(0.1)
        
        print(f"✅ Batch complete: {success_count} successful, {error_count} failed")
        return {"success": success_count, "error": error_count}
        
    except Exception as e:
        print(f"❌ Error processing batch file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": 0, "error": len(batch_data.get("results", []))}

def main():
    """Main function to process all batch files"""
    print("🚀 Starting Neo4j Import Process")
    
    # Get JSON files from the folder
    global JSON_FILES
    JSON_FILES = get_json_files()
    
    if not JSON_FILES:
        print(f"❌ No batch files found in folder '{BATCH_FOLDER}'. Please check the folder path.")
        return
    
    # Connect to Neo4j
    print("\n🔌 Connecting to Neo4j...")
    driver = connect_to_neo4j()
    
    total_success = 0
    total_error = 0
    
    # Process each file
    for file_path in JSON_FILES:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        result = process_batch_file(file_path, driver)
        total_success += result["success"]
        total_error += result["error"]
    
    # Close the driver
    driver.close()
    
    print("\n📊 Import Summary:")
    print(f"   Total records processed: {total_success + total_error}")
    print(f"   Successfully imported: {total_success}")
    print(f"   Failed: {total_error}")
    print("\n🏁 Import process complete!")

if __name__ == "__main__":
    main()