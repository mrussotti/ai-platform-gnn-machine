import pickle
import re
import json
import pandas as pd
import requests
import os
import base64
from io import BytesIO
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from openai import OpenAI
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly

app = Flask(__name__)
CORS(app)

###############################################################################
#                            Neo4j Functions
###############################################################################
def connect_to_neo4j():
    # Using the same connection details as in sample.py for compatibility
    uri = "neo4j+ssc://2b9a3029.databases.neo4j.io"
    username = "neo4j"
    password = "4GGm6B1aeQdjWyFxs7MpVCtqc8xZU0aueO_kqeAwXto"
    
    # Create driver without the encrypted parameter for secure protocols
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()

    return driver


def _get_incident_transcripts(tx):
    query = "MATCH (i:Incident)-[:CONTAINS]->(t:Transcript) RETURN i.nature AS nature, t.TEXT AS transcript"
    result = tx.run(query)
    return [record.data() for record in result]

def extract_training_data(driver):
    with driver.session() as session:
        data = session.execute_read(_get_incident_transcripts)
        training_df = pd.DataFrame(data)
        print("Training DataFrame columns:", training_df.columns)
        print("First few rows of training data:")
        print(training_df.head())
        training_df.dropna(subset=['nature', 'transcript'], inplace=True)
        return training_df

def preprocess_training_data(df):
    def clean_transcript(text):
        # Remove timestamps and speaker markers, then clean extra whitespace and lowercase the text.
        cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().lower()

    df["clean_transcript"] = df["transcript"].apply(clean_transcript)
    df["clean_nature"] = df["nature"].str.strip().str.lower()

    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])

    return df, le

###############################################################################
#         Legacy Training and Evaluation (Old Way)
###############################################################################
def train_and_evaluate_encodings(training_df, le):
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizers = {
        "TfidfVectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "CountVectorizer": CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "HashingVectorizer": HashingVectorizer(n_features=5000, ngram_range=(1, 2), stop_words='english', alternate_sign=False)
    }

    results = {}
    for vec_name, vectorizer in vectorizers.items():
        print(f"\n=== Training with {vec_name} ===")
        if vec_name == "HashingVectorizer":
            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        else:
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(report)

        results[vec_name] = {
            "accuracy": accuracy,
            "report": report
        }
    return results

###############################################################################
#         New Training and Saving (Additional Features)
###############################################################################
# Define paths for models
LABEL_ENCODER_PATH = "label_encoder.pkl"

def train_and_save_model(training_df, le):
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizers = {
        "TfidfVectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "CountVectorizer": CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "HashingVectorizer": HashingVectorizer(n_features=5000, ngram_range=(1, 2), stop_words='english', alternate_sign=False)
    }

    messages = []
    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\n=== Results for {vec_name} ===")
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(report)

        model_filename = f"model_{vec_name}.pkl"
        vectorizer_filename = f"vectorizer_{vec_name}.pkl"

        with open(model_filename, "wb") as f:
            pickle.dump(clf, f)
        with open(vectorizer_filename, "wb") as f:
            pickle.dump(vectorizer, f)

        messages.append(f"{vec_name} trained and saved successfully as {model_filename} and {vectorizer_filename}.")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return "\n".join(messages)

def load_count_model():
    with open("model_CountVectorizer.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("vectorizer_CountVectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return clf, vectorizer, le


###############################################################################
#           New Feature: Extract Complete 911 Call Data using DeepSeek API
###############################################################################
def extract_all_911_call_data(transcript):
    """
    Use the DeepSeek API via OpenAI client to extract complete 911 call information from a transcript,
    following the Neo4j graph structure:
    - Incident details (nature, severity, hazards)
    - Person information (name, phone, role, etc.)
    - Location/time data (address, type, features, time)
    - Call metadata (summary, timestamp)
    
    Returns a dictionary with structured data for all Neo4j nodes.
    """
    
    # Initialize the client with DeepSeek base URL and your API key
    client = OpenAI(
        api_key="sk-d0a34cbfde64466eb6e7c7b07f12e2c9",  # Replace with your actual DeepSeek API key
        base_url="https://api.deepseek.com"
    )
    
    system_prompt = """
    You are an expert emergency call analyzer. Extract the following information from the 911 call transcript.
    Return ONLY a valid JSON object that strictly follows this structure:
    
    {
        "incident": {
            "summary": "Brief summary of the incident",
            "timestamp": "",
            "transcript": "The call transcript (pass through as is)",
            "nature": "Nature of the incident (e.g., heart attack, car accident)",
            "severity": "Severity level (e.g., life-threatening, minor)",
            "hazards": "Any hazards on scene (e.g., active shooter, fire)"
        },
        "calls": [
            {
                "summary": "Brief summary of this specific call",
                "timestamp": "Timestamp of when the call was made if mentioned"
            }
        ],
        "persons": [
            {
                "name": "Person's name if mentioned",
                "phone": "Phone number if mentioned",
                "role": "Role in the incident (e.g., caller, victim, witness)",
                "relationship": "Relationship with other persons mentioned",
                "conditions": "Any preexisting conditions/medications mentioned",
                "age": "Age if mentioned",
                "sex": "Sex/gender if mentioned"
            }
        ],
        "location": {
            "address": "Address of the incident if mentioned",
            "type": "Type of location (e.g., residence, business, public space)",
            "features": "Identifying features (e.g., color of house, car description)",
            "time": "Time when the incident occurred"
        }
    }
    
    IMPORTANT:
    1. Do not wrap your JSON in markdown code blocks (```). Return just the JSON.
    2. Only include values that are explicitly mentioned in the transcript. Use empty strings for unknown values.
    3. Include all persons mentioned in the transcript as separate objects in the 'persons' array.
    4. Include at least one call in the calls array.
    5. The incident will always have one summary (a general summary of what happened).
    6. Each call will have its own summary focusing on what was communicated in that call.
    """
    
    user_prompt = f"Extract all information from the following 911 call transcript:\n\n{transcript}\n"
    
    try:
        # Using the OpenAI client with DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",  # Use the appropriate DeepSeek model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            stream=False
        )
        
        # Add detailed logging to debug API response
        # print(f"API Response status: {response.status_code if hasattr(response, 'status_code') else 'No status code'}")
        # print(f"API Response type: {type(response)}")
        # print(f"API Response: {response}")
        
        # Extract content from the response
        content = response.choices[0].message.content
        print(f"Content to parse: {content[:100]}...")  # Log first 100 chars
        
        # Strip any Markdown code block formatting
        # This will remove ```json and ``` from the content
        cleaned_content = content
        if content.startswith("```"):
            # Find where the opening code block ends and where the closing code block starts
            content_start = content.find("\n") + 1
            content_end = content.rfind("```")
            if content_end > content_start:
                cleaned_content = content[content_start:content_end].strip()
            else:
                # If we can't find the proper end, just remove the first line
                cleaned_content = content[content_start:].strip()
        
        # Parse the JSON response
        try:
            call_data = json.loads(cleaned_content)
            
            # Add the original transcript to the incident data if not already there
            if "incident" in call_data and "transcript" not in call_data["incident"]:
                call_data["incident"]["transcript"] = transcript
            
            # Ensure at least one call exists
            if "calls" not in call_data or not call_data["calls"]:
                call_data["calls"] = [{"summary": "911 emergency call", "timestamp": ""}]
            
            # Ensure we have location data
            if "location" not in call_data:
                call_data["location"] = {
                    "address": "",
                    "type": "",
                    "features": "",
                    "time": ""
                }
                
            # Ensure we have at least an empty persons array
            if "persons" not in call_data:
                call_data["persons"] = []
                
            return call_data
            
        except json.JSONDecodeError as json_err:
            print(f"JSON Decode Error: {json_err}")
            print(f"Failed JSON content: {cleaned_content}")
            return fallback_response(transcript)
            
    except Exception as e:
        print(f"Error in DeepSeek API call: {e}")
        import traceback
        traceback.print_exc()
        # In case of an error, return a minimal structure with empty values
        return fallback_response(transcript)

def fallback_response(transcript):
    """Return a fallback response structure when API call fails that matches our graph model"""
    return {
        "incident": {
            "summary": "Failed to extract information from transcript",
            "timestamp": "",
            "transcript": transcript,
            "nature": "",
            "severity": "",
            "hazards": ""
        },
        "calls": [
            {
                "summary": "911 emergency call",
                "timestamp": ""
            }
        ],
        "persons": [],
        "location": {
            "address": "",
            "type": "",
            "features": "",
            "time": ""
        }
    }

def preprocess_transcript(transcript):
    """
    Clean and preprocess the transcript to make it more suitable for analysis.
    
    Parameters:
    - transcript: Raw transcript text
    
    Returns:
    - Cleaned transcript text
    """
    # Remove timestamp and speaker patterns
    cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", transcript)
    
    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Check if the transcript is too short
    if len(cleaned.split()) < 10:
        print(f"Warning: Transcript might be too short for accurate analysis: {len(cleaned.split())} words")
    
    return cleaned

def analyze_transcript_quality(transcript):
    """
    Analyze the quality of the transcript to determine if it's suitable for extraction.
    
    Parameters:
    - transcript: Raw or cleaned transcript text
    
    Returns:
    - Dictionary with quality metrics and warnings
    """
    words = transcript.split()
    word_count = len(words)
    
    # Calculate some basic metrics
    metrics = {
        "word_count": word_count,
        "warnings": [],
        "is_suitable": True
    }
    
    # Check for minimum word count
    if word_count < 20:
        metrics["warnings"].append("Transcript is very short and may not contain enough information")
        metrics["is_suitable"] = False
    
    
    # Check for potential formatting issues
    if transcript.count("\n") > word_count / 10:
        metrics["warnings"].append("Transcript has unusual formatting with many line breaks")
    
    # Check for potential relevant keywords
    emergency_keywords = ["emergency", "help", "911", "accident", "injured", "hurt", 
                         "medical", "fire", "police", "ambulance", "bleeding", "pain",
                         "unconscious", "breath", "weapon", "gun", "knife"]
    
    found_keywords = [word for word in emergency_keywords if word in transcript.lower()]
    metrics["emergency_keywords_found"] = found_keywords
    
    if not found_keywords:
        metrics["warnings"].append("No emergency keywords detected - might not be a 911 call")
    
    return metrics

def save_911_call_to_neo4j(call_data, driver):
    """
    Save the extracted 911 call data to Neo4j.
    Creates all nodes and relationships according to the database schema.
    
    Parameters:
    - call_data: Dictionary containing structured data from extract_all_911_call_data
    - driver: Neo4j driver connection
    
    Returns:
    - Dictionary with status information
    """
    try:
        with driver.session() as session:
            # Create the transaction function
            def create_call_graph(tx, data):
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
                
                return {
                    "incident_id": incident_id,
                    "call_ids": call_ids,
                    "location_id": location_id,
                    "person_ids": person_ids
                }
            
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


###############################################################################
#                         Flask Endpoints
###############################################################################

# uses the old training method (multiple vectorizers).
@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        driver = connect_to_neo4j()
        training_df = extract_training_data(driver)
        training_df, le = preprocess_training_data(training_df)
        results = train_and_evaluate_encodings(training_df, le)
        driver.close()
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# trains using TfidfVectorizer and saves the model.
@app.route('/train_and_save', methods=['POST'])
def train_and_save_endpoint():
    try:
        driver = connect_to_neo4j()
        training_df = extract_training_data(driver)
        training_df, le = preprocess_training_data(training_df)
        message = train_and_save_model(training_df, le)
        driver.close()
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# load saved model and make predictions.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        transcript = data.get("transcript", "").strip()

        if not transcript:
            return jsonify({"status": "error", "message": "No transcript provided."}), 400

        clf, vectorizer, le = load_count_model()

        cleaned_transcript = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", transcript)
        cleaned_transcript = re.sub(r"\s+", " ", cleaned_transcript).strip().lower()

        transcript_vectorized = vectorizer.transform([cleaned_transcript])
        prediction_label = clf.predict(transcript_vectorized)[0]
        predicted_nature = le.inverse_transform([prediction_label])[0]

        return jsonify({
            "status": "success",
            "transcript": transcript,
            "predicted_nature": predicted_nature
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# New endpoint to process complete 911 call data
@app.route('/process_911_call', methods=['POST'])
def process_911_call_endpoint():
    try:
        data = request.json
        transcript = data.get("transcript", "")
        if not isinstance(transcript, str):
            transcript = json.dumps(transcript)
        transcript = transcript.strip()
        
        if not transcript:
            return jsonify({
                "status": "error", 
                "message": "No transcript provided."
            }), 400
        
        # First, preprocess and analyze transcript quality
        cleaned_transcript = preprocess_transcript(transcript)
        quality_metrics = analyze_transcript_quality(cleaned_transcript)
        
        # If transcript quality is poor, warn the user but continue processing
        warnings = quality_metrics.get("warnings", [])
        
        # Extract all data from the transcript
        call_data = extract_all_911_call_data(transcript)
        
        # Optionally save to Neo4j
        neo4j_result = None
        # if data.get("save_to_neo4j", True):
        #     driver = connect_to_neo4j()
        #     neo4j_result = save_911_call_to_neo4j(call_data, driver)
        #     driver.close()
        
        # Return the extracted data and Neo4j save status
        return jsonify({
            "status": "success",
            "call_data": call_data,
            "quality_metrics": quality_metrics,
            "neo4j_result": neo4j_result,
            "warnings": warnings,
            "message": "Successfully processed 911 call data"
        })
        
    except Exception as e:
        print(f"Error in process_911_call_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/process_all_transcripts', methods=['POST'])
def process_all_transcripts():
    try:
        # Get parameters from request
        data = request.json
        file_name = data.get("file_name", "911_dataset3.csv")
        save_to_neo4j = data.get("save_to_neo4j", False)  # Default to false
        batch_size = data.get("batch_size", 10)  # Process in batches
        limit_records = data.get("limit_records", False)  # Default to processing all
        max_records = data.get("max_records", 50)  # Limit if requested
        
        # Check if file exists
        import os
        if not os.path.exists(file_name):
            return jsonify({
                "status": "error",
                "message": f"File {file_name} not found in the current directory."
            }), 404
        
        # Read the CSV file
        try:
            # Try different encodings since UTF-8 failed
            encodings_to_try = ['latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    print(f"Attempting to read CSV with {encoding} encoding...")
                    df = pd.read_csv(file_name, encoding=encoding)
                    print(f"CSV loaded successfully with {encoding} encoding. Shape: {df.shape}")
                    print(f"Columns: {df.columns.tolist()}")
                    break
                except Exception as enc_error:
                    print(f"Failed with {encoding} encoding: {str(enc_error)}")
            
            if df is None:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to read CSV file with any of the attempted encodings: {encodings_to_try}"
                }), 500
            
            # Check if the TEXT column exists
            if "TEXT" not in df.columns:
                return jsonify({
                    "status": "error",
                    "message": f"Column 'TEXT' not found in the file. Available columns: {df.columns.tolist()}"
                }), 400
            
            # Filter out empty transcripts
            df = df[df["TEXT"].notna() & (df["TEXT"].str.strip() != "")]
            total_transcripts = len(df)
            print(f"Found {total_transcripts} non-empty transcripts")
            
            # Limit records if requested
            if limit_records:
                print(f"Limiting to first {max_records} records as requested.")
                df = df.head(max_records)
                total_transcripts = len(df)
            
            # Process in batches
            results = []
            errors = []
            total_batches = (total_transcripts + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_transcripts)
                print(f"Processing batch {batch_num + 1}/{total_batches} (records {start_idx} to {end_idx})")
                
                batch_df = df.iloc[start_idx:end_idx]
                
                for idx, row in batch_df.iterrows():
                    transcript = row["TEXT"]
                    try:
                        # Get additional metadata if available
                        metadata = {col: row[col] for col in df.columns if col != "TEXT"}
                        
                        # Process transcript
                        call_data = extract_all_911_call_data(transcript)
                        
                        # Add row index and metadata
                        call_data["row_index"] = int(idx)
                        call_data["metadata"] = {
                            k: (str(v) if pd.notna(v) else "") 
                            for k, v in metadata.items()
                        }
                        
                        # No Neo4j saving for now
                        results.append(call_data)
                        print(f"Successfully processed record {idx}")
                        
                    except Exception as e:
                        print(f"Error processing transcript at index {idx}: {str(e)}")
                        errors.append({
                            "index": int(idx),
                            "error": str(e),
                            "transcript_preview": transcript[:100] + "..." if len(transcript) > 100 else transcript
                        })
                
                # Save batch results to file as we go (as a backup)
                batch_output_file = f"batch_{batch_num+1}_of_{total_batches}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(batch_output_file, "w") as f:
                    batch_data = {
                        "batch": batch_num + 1,
                        "total_batches": total_batches,
                        "batch_size": batch_size,
                        "processed_successfully": len([r for r in results if r["row_index"] in batch_df.index]),
                        "batch_errors": sum(1 for e in errors if int(e["index"]) in batch_df.index),
                        "results": [r for r in results if r["row_index"] in batch_df.index],
                        "error_details": [e for e in errors if int(e["index"]) in batch_df.index]
                    }
                    json.dump(batch_data, f, indent=2)
                print(f"Saved batch results to {batch_output_file}")
                
# Console progress report
                print(f"Progress: {end_idx}/{total_transcripts} records processed ({(end_idx/total_transcripts)*100:.1f}%)")
                
            # Save all results to JSON file
            output_file = f"processed_911_calls_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "total_transcripts": total_transcripts,
                    "processed_successfully": len(results),
                    "errors": len(errors),
                    "results": results,
                    "error_details": errors
                }, f, indent=2)
            
            return jsonify({
                "status": "success",
                "message": f"Successfully processed {len(results)} out of {total_transcripts} transcripts.",
                "output_file": output_file,
                "error_count": len(errors),
                "sample_results": results[:3] if results else []  # Return first 3 results as a sample
            })
            
        except Exception as e:
            print(f"Error reading or processing CSV: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error", 
                "message": f"Error reading or processing CSV: {str(e)}"
            }), 500
            
    except Exception as e:
        print(f"Unexpected error in process_all_transcripts: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/test_neo4j_connection', methods=['GET'])
def test_neo4j_connection():
    """
    Test the Neo4j connection using the connect_to_neo4j function.
    """
    try:
        print("Attempting to connect to Neo4j...")
        
        # Try to establish a connection using the existing function
        driver = connect_to_neo4j()
        
        # Test a simple query to validate the connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message").single()
            message = result["message"] if result else "Query executed but no result returned"
        
        # Close the driver
        driver.close()
        
        print(f"SUCCESS: {message}")
        
        return jsonify({
            "status": "success",
            "message": message,
            "connection_details": {
                "uri": "neo4j+ssc://2b9a3029.databases.neo4j.io",
                "username": "neo4j",
                # Password is masked for security
                "password": "****"
            }
        })
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendation": "Please ensure the Neo4j credentials are correct and that the database is accessible from your environment",
            "connection_details": {
                "uri": "neo4j+ssc://2b9a3029.databases.neo4j.io",
                "username": "neo4j",
                # Password is masked for security
                "password": "****"
            }
        }), 500

###############################################################################
#                   Compatibility Functions for Demo
###############################################################################

def save_compatible_to_neo4j(call_data, driver):
    """
    Save 911 call data to Neo4j in a format compatible with the structure used in sample.py.
    This creates a flatter structure with metadata as a serialized string rather than the
    full graph structure.
    
    Parameters:
    - call_data: Dictionary containing structured data from extract_all_911_call_data
    - driver: Neo4j driver connection
    
    Returns:
    - Dictionary with status information
    """
    try:
        with driver.session() as session:
            # Combine the data into a single Incident node with metadata as a string
            def create_compatible_node(tx, data):
                # Create metadata object that matches the expected format in sample.py and similarity_v2.py
                metadata = {
                    "clean_address_EMS": data["location"]["address"],
                    "clean_address_extracted": data["location"]["address"],
                    "Date_target": data["location"]["time"] or data["incident"]["timestamp"],
                    "start": data["incident"]["timestamp"],
                    "nature": data["incident"]["nature"],
                    "severity": data["incident"]["severity"],
                    "hazards": data["incident"]["hazards"],
                    "persons": [
                        {
                            "name": p["name"],
                            "role": p["role"],
                            "age": p["age"],
                            "sex": p["sex"],
                            "conditions": p["conditions"],
                            "relationship": p["relationship"],
                            "phone": p["phone"]
                        } for p in data["persons"] if any(p.values())
                    ],
                    "location_type": data["location"]["type"],
                    "location_features": data["location"]["features"],
                    "source_file": data.get("source_file", "")
                }
                
                # Add calls data to metadata
                metadata["calls"] = [
                    {
                        "summary": call["summary"],
                        "timestamp": call["timestamp"]
                    } for call in data["calls"]
                ]
                
                # Create a single Incident node with all data
                incident_query = """
                CREATE (i:Incident {
                    summary: $summary,
                    transcript: $transcript,
                    metadata: $metadata
                })
                RETURN id(i) AS incident_id
                """
                
                # Convert metadata to a string
                metadata_str = str(metadata)
                
                # Extract the main summary from the incident data
                summary = data["incident"]["summary"]
                transcript = data["incident"]["transcript"]
                
                # Run the query
                incident_result = tx.run(
                    incident_query,
                    summary=summary,
                    transcript=transcript,
                    metadata=metadata_str
                ).single()
                
                incident_id = incident_result["incident_id"]
                
                # Also create a transcript node and relationship (to be compatible with some queries)
                transcript_query = """
                CREATE (t:Transcript {
                    TEXT: $transcript
                })
                RETURN id(t) AS transcript_id
                """
                
                transcript_result = tx.run(
                    transcript_query,
                    transcript=transcript
                ).single()
                
                transcript_id = transcript_result["transcript_id"]
                
                # Create relationship between Incident and Transcript
                tx.run("""
                MATCH (i:Incident), (t:Transcript)
                WHERE id(i) = $incident_id AND id(t) = $transcript_id
                CREATE (i)-[:CONTAINS]->(t)
                """, incident_id=incident_id, transcript_id=transcript_id)
                
                return {
                    "incident_id": incident_id,
                    "transcript_id": transcript_id
                }
            
            # Execute the transaction function
            result = session.execute_write(create_compatible_node, call_data)
            
            return {
                "status": "success",
                "message": "911 call data successfully saved to Neo4j in a compatible format",
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

###############################################################################
#                       Demo Visualization Functions
###############################################################################

def generate_network_graph(call_data):
    """
    Generate a network visualization of the 911 call graph data.
    
    Args:
        call_data: Dictionary containing incident, calls, persons, and location data
    
    Returns:
        Base64 encoded PNG image of the network graph
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with different colors based on type
    incident_id = "incident_1"
    G.add_node(incident_id, label=call_data["incident"]["nature"] or "Incident", 
               node_type="incident", data=call_data["incident"])
    
    # Add location node
    location_id = "location_1"
    G.add_node(location_id, label=call_data["location"]["address"] or "Location", 
               node_type="location", data=call_data["location"])
    
    # Connect incident to location
    G.add_edge(incident_id, location_id, relationship="AT")
    
    # Add call nodes
    for i, call in enumerate(call_data["calls"]):
        call_id = f"call_{i+1}"
        G.add_node(call_id, label=f"Call {i+1}", node_type="call", data=call)
        G.add_edge(call_id, incident_id, relationship="ABOUT")
    
    # Add person nodes
    for i, person in enumerate(call_data["persons"]):
        if not any(person.values()):  # Skip if all values are empty
            continue
        person_id = f"person_{i+1}"
        G.add_node(person_id, label=person["name"] or f"Person {i+1}", 
                   node_type="person", data=person)
        G.add_edge(person_id, incident_id, relationship="INVOLVED_IN")
        
        # If person is a caller, connect to first call
        if person["role"].lower() == "caller" and call_data["calls"]:
            G.add_edge(person_id, "call_1", relationship="MADE")
    
    # Set up the figure and draw the graph
    plt.figure(figsize=(12, 10))
    
    # Define positions
    pos = nx.spring_layout(G, seed=42)
    
    # Define node colors based on type
    node_colors = {
        "incident": "#FF6B6B",  # Red
        "location": "#4ECDC4",  # Teal
        "call": "#FFD166",      # Yellow
        "person": "#6B5B95"     # Purple
    }
    
    # Draw nodes with different colors
    for node_type in node_colors:
        node_list = [node for node, data in G.nodes(data=True) if data.get("node_type") == node_type]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=node_list,
                              node_color=node_colors[node_type],
                              node_size=1500,
                              alpha=0.8)
    
    # Draw edges
    edge_labels = {(u, v): d["relationship"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color="gray", arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Draw labels
    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
    
    # Add title and adjust layout
    plt.title("911 Call Graph Visualization", size=16)
    plt.axis("off")
    plt.tight_layout()
    
    # Convert plot to base64 encoded string
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def generate_plotly_graph(call_data):
    """
    Generate an interactive plotly visualization of the 911 call graph data.
    
    Args:
        call_data: Dictionary containing incident, calls, persons, and location data
    
    Returns:
        HTML representation of the interactive plot
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with different types
    incident_id = "incident_1"
    G.add_node(incident_id, label=call_data["incident"]["nature"] or "Incident", 
               type="incident", data=call_data["incident"])
    
    # Add location node
    location_id = "location_1"
    G.add_node(location_id, label=call_data["location"]["address"] or "Location", 
               type="location", data=call_data["location"])
    
    # Connect incident to location
    G.add_edge(incident_id, location_id, relationship="AT")
    
    # Add call nodes
    for i, call in enumerate(call_data["calls"]):
        call_id = f"call_{i+1}"
        G.add_node(call_id, label=f"Call {i+1}", type="call", data=call)
        G.add_edge(call_id, incident_id, relationship="ABOUT")
    
    # Add person nodes
    for i, person in enumerate(call_data["persons"]):
        if not any(person.values()):  # Skip if all values are empty
            continue
        person_id = f"person_{i+1}"
        G.add_node(person_id, label=person["name"] or f"Person {i+1}", 
                   type="person", data=person)
        G.add_edge(person_id, incident_id, relationship="INVOLVED_IN")
        
        # If person is a caller, connect to first call
        if person["role"].lower() == "caller" and call_data["calls"]:
            G.add_edge(person_id, "call_1", relationship="MADE")
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node color map
    node_colors = {
        "incident": "#FF6B6B",  # Red
        "location": "#4ECDC4",  # Teal
        "call": "#FFD166",      # Yellow
        "person": "#6B5B95"     # Purple
    }
    
    # Create node traces
    node_traces = {}
    for node_type, color in node_colors.items():
        node_traces[node_type] = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            name=node_type.capitalize(),
            marker=dict(
                color=color,
                size=30,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        )
    
    # Add node information
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node]['type']
        node_label = G.nodes[node]['label']
        data = G.nodes[node]['data']
        
        # Prepare hover text based on node type
        if node_type == 'incident':
            hover_text = (f"<b>Incident</b><br>"
                         f"Nature: {data['nature']}<br>"
                         f"Severity: {data['severity']}<br>"
                         f"Hazards: {data['hazards']}<br>"
                         f"Summary: {data['summary']}")
        elif node_type == 'location':
            hover_text = (f"<b>Location</b><br>"
                         f"Address: {data['address']}<br>"
                         f"Type: {data['type']}<br>"
                         f"Features: {data['features']}")
        elif node_type == 'call':
            hover_text = (f"<b>Call</b><br>"
                         f"Summary: {data['summary']}<br>"
                         f"Timestamp: {data['timestamp']}")
        elif node_type == 'person':
            hover_text = (f"<b>Person</b><br>"
                         f"Name: {data['name']}<br>"
                         f"Role: {data['role']}<br>"
                         f"Age: {data['age']}<br>"
                         f"Sex: {data['sex']}")
        
        # Add to appropriate trace
        node_traces[node_type].x = list(node_traces[node_type].x) + [x]
        node_traces[node_type].y = list(node_traces[node_type].y) + [y]
        node_traces[node_type].text = list(node_traces[node_type].text) + [hover_text]
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Add edge information
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.x = list(edge_trace.x) + [x0, x1, None]
        edge_trace.y = list(edge_trace.y) + [y0, y1, None]
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + list(node_traces.values()),
        layout=go.Layout(
            title='Interactive 911 Call Network',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"911 Call: {call_data['incident']['nature']}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    
    # Convert to HTML
    html = plotly.io.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return html

def generate_incident_summary_plot(call_data):
    """
    Generate a visualization showing key elements of the incident.
    
    Args:
        call_data: Dictionary containing incident, calls, persons, and location data
    
    Returns:
        Base64 encoded PNG image of the summary visualization
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"911 Call Analysis: {call_data['incident']['nature']}", fontsize=16)
    
    # Extract key information
    incident = call_data["incident"]
    location = call_data["location"]
    persons = call_data["persons"]
    
    # Subplot 1: Incident Summary
    axs[0, 0].text(0.5, 0.95, "Incident Summary", ha="center", fontsize=14, fontweight="bold")
    summary_text = (
        f"Nature: {incident['nature']}\n"
        f"Severity: {incident['severity']}\n"
        f"Hazards: {incident['hazards']}\n"
        f"Location: {location['address']}\n"
        f"Location Type: {location['type']}\n"
        f"Time: {location['time']}\n"
        f"\nSummary: {incident['summary']}"
    )
    axs[0, 0].text(0.1, 0.8, summary_text, va="top", ha="left", wrap=True)
    axs[0, 0].axis("off")
    
    # Subplot 2: Persons Involved
    axs[0, 1].text(0.5, 0.95, "Persons Involved", ha="center", fontsize=14, fontweight="bold")
    
    if persons:
        person_data = []
        for i, person in enumerate(persons):
            if any(person.values()):
                person_text = (
                    f"Person {i+1}:\n"
                    f"  Name: {person['name']}\n"
                    f"  Role: {person['role']}\n"
                    f"  Age: {person['age']}\n"
                    f"  Sex: {person['sex']}\n"
                    f"  Conditions: {person['conditions']}\n"
                )
                person_data.append(person_text)
        
        person_summary = "\n".join(person_data)
        axs[0, 1].text(0.1, 0.8, person_summary, va="top", ha="left")
    else:
        axs[0, 1].text(0.5, 0.5, "No person information available", ha="center")
    
    axs[0, 1].axis("off")
    
    # Subplot 3: Extracted Entities
    axs[1, 0].text(0.5, 0.95, "Extracted Information", ha="center", fontsize=14, fontweight="bold")
    
    # Count entities by type
    entity_counts = {
        "Location Details": len([v for v in location.values() if v]),
        "Person Information": sum(sum(1 for v in p.values() if v) for p in persons),
        "Incident Details": len([v for v in incident.values() if v and v != incident['transcript']])
    }
    
    if sum(entity_counts.values()) > 0:
        # Create horizontal bar chart
        bars = axs[1, 0].barh(list(entity_counts.keys()), list(entity_counts.values()), color=["#4ECDC4", "#6B5B95", "#FF6B6B"])
        axs[1, 0].set_xlabel("Number of Extracted Fields")
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else 0
            axs[1, 0].text(label_x_pos + 0.5, bar.get_y() + bar.get_height()/2, f"{width}", 
                           va='center', fontweight='bold')
    else:
        axs[1, 0].text(0.5, 0.5, "No entities extracted", ha="center")
        axs[1, 0].axis("off")
    
    # Subplot, 4: Incident Type Visualization
    axs[1, 1].text(0.5, 0.95, "Incident Classification", ha="center", fontsize=14, fontweight="bold")
    
    # Simplified incident type classifier
    incident_type = incident['nature'].lower() if incident['nature'] else "unknown"
    incident_categories = {
        'medical': ['medical', 'heart', 'attack', 'breathing', 'chest', 'pain', 'unconscious', 'injury', 'sick', 'stroke'],
        'fire': ['fire', 'burning', 'smoke', 'flames', 'burn', 'explosion'],
        'crime': ['crime', 'assault', 'violence', 'weapon', 'disturbance', 'domestic', 'theft', 'robbery'],
        'traffic': ['traffic', 'accident', 'crash', 'collision', 'car', 'vehicle'],
        'other': ['other', 'assistance', 'unknown']
    }
    
    # Determine incident category
    incident_category = 'other'
    max_matches = 0
    
    for category, keywords in incident_categories.items():
        matches = sum(1 for keyword in keywords if keyword in incident_type)
        if matches > max_matches:
            max_matches = matches
            incident_category = category
    
    # Create a pie chart with just the determined category highlighted
    categories = list(incident_categories.keys())
    sizes = [1 if cat == incident_category else 0.05 for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#6B5B95', '#F9F9F9']
    explode = [0.1 if cat == incident_category else 0 for cat in categories]
    
    axs[1, 1].pie(sizes, explode=explode, labels=categories, colors=colors, autopct=lambda p: f'{p:.0f}%' if p > 1 else '',
                 shadow=False, startangle=90)
    axs[1, 1].axis('equal')
    axs[1, 1].text(0, -1.2, f"Classified as: {incident_category.upper()}", ha="center", fontweight="bold")
    
    # Adjust layout and convert to image
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Convert plot to base64 encoded string
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

###############################################################################
#                         Demo Endpoint
###############################################################################
@app.route('/demo', methods=['GET'])
def demo_endpoint():
    """
    Demo endpoint that processes sample 911 call transcripts,
    extracts information using DeepSeek API, creates graph data structure,
    uploads to Neo4j, and provides visualizations.
    """
    try:
        # Set up logging
        import logging
        logging.basicConfig(filename='demo_processing.log', level=logging.INFO)
        logger = logging.getLogger('demo')
        logger.info("Starting demo processing...")
        
        # Transcript file paths
        transcript_files = [
            "Transcripts/fire transcript.txt",
            "Transcripts/fire transcript 2.txt",
            "Transcripts/heart attack transcript.txt"
        ]
        
        # Process each transcript
        processed_data = []
        network_graphs = []
        interactive_graphs = []
        summary_graphs = []
        
        # Connect to Neo4j
        driver = connect_to_neo4j()
        
        for file_path in transcript_files:
            try:
                # Read transcript file
                with open(file_path, 'r') as f:
                    transcript = f.read()
                
                logger.info(f"Processing transcript from {file_path}")
                
                # Clean and preprocess transcript
                cleaned_transcript = preprocess_transcript(transcript)
                
                # Extract data using DeepSeek API
                call_data = extract_all_911_call_data(cleaned_transcript)
                
                # Add source file information
                call_data["source_file"] = file_path
                
                # Save to Neo4j in a format compatible with sample.py structure
                neo4j_result = save_compatible_to_neo4j(call_data, driver)
                call_data["neo4j_status"] = neo4j_result["status"]
                call_data["neo4j_node_ids"] = neo4j_result.get("node_ids", {})
                
                # Generate visualizations
                try:
                    network_graph = generate_network_graph(call_data)
                    interactive_graph = generate_plotly_graph(call_data)
                    summary_graph = generate_incident_summary_plot(call_data)
                    
                    network_graphs.append({
                        "source_file": file_path, 
                        "image": network_graph
                    })
                    
                    interactive_graphs.append({
                        "source_file": file_path, 
                        "html": interactive_graph
                    })
                    
                    summary_graphs.append({
                        "source_file": file_path, 
                        "image": summary_graph
                    })
                except Exception as vis_error:
                    logger.error(f"Error generating visualizations for {file_path}: {str(vis_error)}")
                
                # Add to processed data
                processed_data.append(call_data)
                logger.info(f"Successfully processed {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Close Neo4j connection
        driver.close()
        
        # Prepare response
        response = {
            "status": "success",
            "message": f"Successfully processed {len(processed_data)} transcripts",
            "compatibility_note": "Data was saved to Neo4j using a structure compatible with sample.py and similarity_v2.py",
            "processed_data": processed_data,
            "visualizations": {
                "network_graphs": network_graphs,
                "interactive_graphs": interactive_graphs,
                "summary_graphs": summary_graphs
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in demo endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/demo/visualization/<file_name>/<vis_type>', methods=['GET'])
def get_visualization(file_name, vis_type):
    """
    Endpoint to retrieve a specific visualization as an image or HTML.
    
    Args:
        file_name: Name of the source file (e.g., "fire_transcript.txt")
        vis_type: Type of visualization (network, interactive, summary)
    """
    try:
        # Parse file name from parameter (might contain full path)
        file_base = os.path.basename(file_name)
        
        # Find the full file path from the available transcript files
        transcript_files = [
            "Transcripts/fire transcript.txt",
            "Transcripts/fire transcript 2.txt",
            "Transcripts/heart attack transcript.txt"
        ]
        
        found_file = None
        for f in transcript_files:
            if file_base in f:
                found_file = f
                break
        
        if not found_file:
            return jsonify({"status": "error", "message": f"File {file_name} not found"}), 404
        
        # Read and process the transcript
        with open(found_file, 'r') as f:
            transcript = f.read()
        
        # Clean and extract data
        cleaned_transcript = preprocess_transcript(transcript)
        call_data = extract_all_911_call_data(cleaned_transcript)
        
        # Generate the requested visualization
        if vis_type == 'network':
            img_str = generate_network_graph(call_data)
            img_data = base64.b64decode(img_str)
            return Response(img_data, mimetype='image/png')
            
        elif vis_type == 'interactive':
            html_content = generate_plotly_graph(call_data)
            return Response(html_content, mimetype='text/html')
            
        elif vis_type == 'summary':
            img_str = generate_incident_summary_plot(call_data)
            img_data = base64.b64decode(img_str)
            return Response(img_data, mimetype='image/png')
            
        else:
            return jsonify({"status": "error", "message": f"Visualization type {vis_type} not supported"}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500
        
if __name__ == "__main__":
    app.run(debug=True)