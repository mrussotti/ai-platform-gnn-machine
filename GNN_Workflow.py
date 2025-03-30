import pickle
import re
import json
import pandas as pd
import openai
from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

app = Flask(__name__)
CORS(app)

###############################################################################
#                            Neo4j Functions
###############################################################################
def connect_to_neo4j():
    uri = "neo4j+s://09ed30ec.databases.neo4j.io"
    username = "neo4j"
    password = "N7azcFeea5x3mkCafrBbdjAbptvfVW4RVyKUdtsie70"
    driver = GraphDatabase.driver(uri, auth=(username, password))
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
#           New Feature: Extract Incident Information using ChatGPT API
###############################################################################
def extract_incident_information_pipeline(transcript):
    """
    Use the OpenAI API (GPT-4) to extract incident information from a transcript.
    The API is prompted to extract:
      - nature: the nature of the incident,
      - hazards: hazards present on the scene,
      - summary: a concise summary of the incident.
      
    Only include values if you are confident they are accurate and relevant.
    Returns a dictionary with the keys 'nature', 'hazards', and 'summary'.
    """
    # Import OpenAI client at the beginning of your file
    from openai import OpenAI
    
    # Initialize the client with your API key
    client = OpenAI(api_key="")
    
    system_prompt = (
        "You are an expert incident analyzer. Extract the following information from the transcript: "
        "1. The nature of the incident. "
        "2. Hazards present on the scene. "
        "3. A concise summary of the incident. "
        "Only include values if you are confident they are accurate and relevant. "
        "Return your answer as valid JSON with the keys 'nature', 'hazards', and 'summary'."
    )
    
    user_prompt = f"Extract the incident information from the following transcript:\n\n{transcript}\n"
    
    try:
        # Using the new API format
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        # Access the content from the new response structure
        content = response.choices[0].message.content
        info = json.loads(content)
        return info
    except Exception as e:
        print(f"Error in ChatGPT API call: {e}")
        # In case of an error, return empty values.
        return {"nature": "", "hazards": "", "summary": ""}

###############################################################################
#           New Feature: Extract Complete 911 Call Data using ChatGPT API
###############################################################################
def extract_all_911_call_data(transcript):
    """
    Use the OpenAI API (GPT-4) to extract complete 911 call information from a transcript,
    including all nodes in the Neo4j structure:
    - Incident details (nature, severity, hazards)
    - Person information (name, phone, role, etc.)
    - Location/time data (address, type, features, time)
    - Call metadata (summary)
    
    Returns a dictionary with structured data for all Neo4j nodes.
    """
    from openai import OpenAI
    
    # Initialize the client with your API key
    client = OpenAI(api_key="Insert_key_here")
    
    system_prompt = """
    You are an expert emergency call analyzer. Extract the following information from the 911 call transcript.
    Return ONLY a valid JSON object with these keys and nested objects:
    
    {
        "call": {
            "summary": "Brief summary of the call"
        },
        "incident": {
            "nature": "Nature of the incident (e.g., heart attack, car accident)",
            "severity": "Severity level (e.g., life-threatening, minor)",
            "hazards": "Any hazards on scene (e.g., active shooter, fire)",
            "transcript": "The call transcript (pass through as is)"
        },
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
            "time": "Time when the incident occurred or when call was made"
        }
    }
    
    Only include values that are explicitly mentioned in the transcript. Use empty strings for unknown values.
    Include all persons mentioned in the transcript as separate objects in the 'persons' array.
    """
    
    user_prompt = f"Extract all information from the following 911 call transcript:\n\n{transcript}\n"
    
    try:
        # Using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        # Access the content from the response
        content = response.choices[0].message.content
        
        # Parse the JSON response
        call_data = json.loads(content)
        
        # Add the original transcript to the incident data
        if "incident" in call_data:
            call_data["incident"]["transcript"] = transcript
            
        return call_data
    except Exception as e:
        print(f"Error in ChatGPT API call: {e}")
        # In case of an error, return a minimal structure with empty values
        return {
            "call": {"summary": ""},
            "incident": {"nature": "", "severity": "", "hazards": "", "transcript": transcript},
            "persons": [{"name": "", "phone": "", "role": "", "relationship": "", "conditions": "", "age": "", "sex": ""}],
            "location": {"address": "", "type": "", "features": "", "time": ""}
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
                # Create Call node
                call_query = """
                CREATE (c:Call {summary: $summary})
                RETURN id(c) AS call_id
                """
                call_result = tx.run(call_query, summary=data["call"]["summary"]).single()
                call_id = call_result["call_id"]
                
                # Create Incident node
                incident_query = """
                CREATE (i:Incident {
                    nature: $nature,
                    severity: $severity,
                    hazards: $hazards,
                    transcript: $transcript
                })
                RETURN id(i) AS incident_id
                """
                incident_result = tx.run(
                    incident_query,
                    nature=data["incident"]["nature"],
                    severity=data["incident"]["severity"],
                    hazards=data["incident"]["hazards"],
                    transcript=data["incident"]["transcript"]
                ).single()
                incident_id = incident_result["incident_id"]
                
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
                    if person["role"].lower() == "caller":
                        tx.run("""
                        MATCH (p:Person), (c:Call)
                        WHERE id(p) = $person_id AND id(c) = $call_id
                        CREATE (p)-[:MADE]->(c)
                        """, person_id=person_id, call_id=call_id)
                
                return {
                    "call_id": call_id,
                    "incident_id": incident_id,
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

@app.route('/extract_incident', methods=['POST'])
def extract_incident_endpoint():
    try:
        data = request.json
        transcript = data.get("transcript", "")
        if not isinstance(transcript, str):
            transcript = json.dumps(transcript)
        transcript = transcript.strip()
        if not transcript:
            return jsonify({"status": "error", "message": "No transcript provided."}), 400
        
        print("About to call extract_incident_information_pipeline function")
        incident_info = extract_incident_information_pipeline(transcript)
        print("Function completed successfully")
        
        if 'summary' in incident_info and incident_info['summary']:
            summary = incident_info['summary']
            summary = re.sub(r'^\{\"value\":\s*\"', '', summary) 
            summary = re.sub(r'\"\}$', '', summary) 
            summary = re.sub(r'\\[rn"]', ' ', summary)  
            incident_info['summary'] = summary
        
        return jsonify({
            "status": "success", 
            "incident_info": incident_info,
            "message": "Successfully extracted incident information"
        })
    except Exception as e:
        print(f"Error in extract_incident_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
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
       # neo4j_result = None
       # if data.get("save_to_neo4j", True):
        #    driver = connect_to_neo4j()
         #   neo4j_result = save_911_call_to_neo4j(call_data, driver)
         #   driver.close()
        
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

if __name__ == "__main__":
    app.run(debug=True)
