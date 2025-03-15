import pickle
import re
import json
import pandas as pd
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
#           New Feature: Extract Incident Information using LLM
###############################################################################

def extract_incident_information_pipeline(transcript):
    """
    Extracts information from 911 transcripts using a simpler approach without keywords.
    """
    from transformers import pipeline
    import torch
    import re
    
    # Clean the transcript
    cleaned_transcript = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", transcript)
    cleaned_transcript = re.sub(r"\s+", " ", cleaned_transcript).strip()
    transcript_lower = cleaned_transcript.lower()
    
    try:
        # ======== NATURE OF INCIDENT EXTRACTION ========
        # Keep your existing classifier-based approach for nature of incident
        nature_of_incident = ""
        try:
            clf, vectorizer, le = load_count_model()
            transcript_vectorized = vectorizer.transform([transcript_lower])
            
            prediction_label = clf.predict(transcript_vectorized)[0]
            prediction_proba = clf.predict_proba(transcript_vectorized)[0]
            max_proba = max(prediction_proba)
            
            NATURE_CONFIDENCE_THRESHOLD = 0.7
            
            if max_proba >= NATURE_CONFIDENCE_THRESHOLD:
                nature_of_incident = le.inverse_transform([prediction_label])[0]
                print(f"Using classifier prediction for nature: {nature_of_incident} (confidence: {max_proba:.2f})")
            else:
                print(f"Classifier confidence too low: {max_proba:.2f} < {NATURE_CONFIDENCE_THRESHOLD}")
        except Exception as e:
            print(f"Error using classifier: {str(e)}")
        
        # ======== CREATE ZERO-SHOT CLASSIFIER (USED FOR BOTH SEVERITY AND HAZARDS) ========
        print("Loading zero-shot classification model...")
        try:
            classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli",
                                 device=-1)  # Always use CPU for compatibility
            
            print("Zero-shot classifier loaded successfully")
        except Exception as e:
            print(f"Error loading zero-shot classifier: {str(e)}")
            # If we can't load the classifier, we'll return empty values
            return {
                "transcript": transcript,
                "nature_of_incident": nature_of_incident,
                "severity_of_incident": "",
                "hazards_on_scene": ""
            }
            
        # ======== SEVERITY OF INCIDENT EXTRACTION ========
        print("Attempting severity extraction...")
        severity_of_incident = ""
        
        try:
            # Candidate labels for severity
            candidate_labels = [
                "life-threatening emergency",
                "serious medical emergency", 
                "moderate medical issue",
                "minor medical concern",
                "non-urgent situation"
            ]
            
            # Classify the transcript
            result = classifier(cleaned_transcript, candidate_labels)
            print(f"Severity classification: {result['labels'][0]} (score: {result['scores'][0]:.2f})")
            
            # Map the classification to a numeric severity
            severity_mapping = {
                "life-threatening emergency": "5/5",
                "serious medical emergency": "4/5",
                "moderate medical issue": "3/5", 
                "minor medical concern": "2/5",
                "non-urgent situation": "1/5"
            }
            
            # Only use the result if the confidence is high enough
            SEVERITY_CONFIDENCE_THRESHOLD = 0.4
            if result['scores'][0] >= SEVERITY_CONFIDENCE_THRESHOLD:
                severity_of_incident = severity_mapping[result['labels'][0]]
                print(f"Determined severity: {severity_of_incident}")
            else:
                print(f"Severity classification confidence too low: {result['scores'][0]:.2f}")
        except Exception as e:
            print(f"Error in severity classification: {str(e)}")
        
        # ======== HAZARDS ON SCENE EXTRACTION ========
        print("Attempting hazards extraction...")
        hazards_on_scene = ""
        
        try:
            # Simpler approach for hazards - just use zero-shot classification
            hazard_class_labels = [
                "scene contains safety hazards", 
                "scene is safe with no hazards",
                "scene safety is unclear"
            ]
            
            hazard_classification = classifier(cleaned_transcript, hazard_class_labels)
            print(f"Hazard classification: {hazard_classification['labels'][0]} (score: {hazard_classification['scores'][0]:.2f})")
            
            # Only set hazards_on_scene if we have high confidence
            HAZARD_CONFIDENCE_THRESHOLD = 0.6
            if hazard_classification['scores'][0] >= HAZARD_CONFIDENCE_THRESHOLD:
                if hazard_classification['labels'][0] == "scene contains safety hazards":
                    # Additional classification to determine type of hazard
                    hazard_type_labels = [
                        "medical risk",
                        "environmental hazard",
                        "vehicle-related danger",
                        "violent situation",
                        "fire hazard"
                    ]
                    
                    type_result = classifier(cleaned_transcript, hazard_type_labels)
                    top_type = type_result['labels'][0]
                    top_score = type_result['scores'][0]
                    
                    if top_score >= 0.5:
                        hazards_on_scene = f"{top_type.capitalize()}"
                        print(f"Hazard type identified: {hazards_on_scene} (score: {top_score:.2f})")
                    else:
                        hazards_on_scene = "Potential hazards detected"
                        print(f"Generic hazard detected, type unclear (score: {top_score:.2f})")
                
                elif hazard_classification['labels'][0] == "scene is safe with no hazards":
                    hazards_on_scene = "None"
                    print("Scene classified as safe with no hazards")
            else:
                print(f"Hazard classification confidence too low: {hazard_classification['scores'][0]:.2f}")
                
        except Exception as e:
            print(f"Error in hazards extraction: {str(e)}")
        
        incident_node = {
            "transcript": transcript,
            "nature_of_incident": nature_of_incident,
            "severity_of_incident": severity_of_incident,
            "hazards_on_scene": hazards_on_scene
        }
        
        return incident_node
        
    except Exception as e:
        print(f"Error in incident extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "transcript": transcript,
            "nature_of_incident": "",
            "severity_of_incident": "",
            "hazards_on_scene": ""
        }

###############################################################################
#                         Flask Endpoints
###############################################################################

#uses the old training method (multiple vectorizers).
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

#trains using TfidfVectorizer and saves the model.
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

#load saved model and make predictions.
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
    


if __name__ == "__main__":
    app.run(debug=True)