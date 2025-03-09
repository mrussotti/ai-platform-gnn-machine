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
    Extracts information from 911 transcripts using ML-based approaches with confidence checks.
    Includes enhanced debugging for severity extraction.
    """
    # Clean the transcript
    cleaned_transcript = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", transcript)
    cleaned_transcript = re.sub(r"\s+", " ", cleaned_transcript).strip()
    transcript_lower = cleaned_transcript.lower()
    
    try:
        # Initialize QA pipeline
        qa_pipeline = pipeline("question-answering")
        
        # ======== NATURE OF INCIDENT EXTRACTION ========
        # Use trained classifier for nature prediction with confidence check
        try:
            # Load and use the existing classification model
            clf, vectorizer, le = load_count_model()
            transcript_vectorized = vectorizer.transform([transcript_lower])
            
            # Get prediction and prediction probability
            prediction_label = clf.predict(transcript_vectorized)[0]
            prediction_proba = clf.predict_proba(transcript_vectorized)[0]
            max_proba = max(prediction_proba)
            
            # Only use prediction if confidence is high enough
            NATURE_CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed based on your model
            
            if max_proba >= NATURE_CONFIDENCE_THRESHOLD:
                nature_of_incident = le.inverse_transform([prediction_label])[0]
                print(f"Using classifier prediction for nature: {nature_of_incident} (confidence: {max_proba:.2f})")
            else:
                nature_of_incident = ""
                print(f"Classifier confidence too low: {max_proba:.2f} < {NATURE_CONFIDENCE_THRESHOLD}")
        except Exception as e:
            print(f"Error using classifier: {str(e)}")
            nature_of_incident = ""
            
            # Fall back to QA extraction if classifier fails
            nature_questions = [
                "What type of emergency is this?",
                "What is the nature of this incident?",
                "What happened?",
                "What is the problem?"
            ]
            
            best_score = 0
            QA_CONFIDENCE_THRESHOLD = 0.5  # Adjust based on testing
            
            for question in nature_questions:
                try:
                    result = qa_pipeline(question=question, context=cleaned_transcript[:800])
                    if result["score"] > best_score and len(result["answer"]) > 2:
                        best_score = result["score"]
                        if best_score >= QA_CONFIDENCE_THRESHOLD:
                            nature_of_incident = result["answer"]
                except Exception as err:
                    print(f"Error in nature extraction question: {str(err)}")
        
        # ======== SEVERITY OF INCIDENT EXTRACTION (Scale 1-5) ========
        # Use a simpler, more robust approach for severity
        SEVERITY_CONFIDENCE_THRESHOLD = 0.1  # Very low threshold for testing
        
        # First approach: Direct question about numerical severity
        print("Attempting severity extraction...")
        severity_of_incident = "3/5"  # Default middle severity as fallback
        
        try:
            # Try a simple direct question first
            simple_severity = qa_pipeline(
                question="On a scale of 1 to 5, how severe is this emergency?",
                context=cleaned_transcript[:800]
            )
            
            print(f"Severity extraction answer: '{simple_severity['answer']}' (confidence: {simple_severity['score']:.2f})")
            
            # Look for any digit in the answer
            severity_match = re.search(r'[1-5]', simple_severity['answer'])
            if severity_match:
                severity_digit = severity_match.group(0)
                severity_of_incident = f"{severity_digit}/5"
                print(f"Found severity digit: {severity_digit}, setting to {severity_of_incident}")
        except Exception as e:
            print(f"Error in simple severity extraction: {str(e)}")
        
        # Second approach: Focused extraction with multiple questions
        if severity_of_incident == "3/5":  # If we're still using the default
            print("Trying alternative severity extraction approaches...")
            severity_indicators = {
                "life threatening": 5,
                "severe": 5,
                "critical": 5,
                "serious": 4,
                "moderate": 3,
                "mild": 2,
                "minor": 1
            }
            
            try:
                # Alternative question about severity without requiring numerical answer
                alt_severity = qa_pipeline(
                    question="How would you describe the severity of this emergency: minor, moderate, or severe?",
                    context=cleaned_transcript[:800]
                )
                
                print(f"Alternative severity result: '{alt_severity['answer']}' (confidence: {alt_severity['score']:.2f})")
                
                # Check for severity indicators in the answer
                answer_lower = alt_severity['answer'].lower()
                for indicator, level in severity_indicators.items():
                    if indicator in answer_lower:
                        severity_of_incident = f"{level}/5"
                        print(f"Found severity indicator '{indicator}', setting to {severity_of_incident}")
                        break
            except Exception as e:
                print(f"Error in alternative severity extraction: {str(e)}")
            
        # ======== HAZARDS ON SCENE EXTRACTION ========
        hazard_questions = [
            "Are there any hazards for emergency responders at the scene?",
            "Is the scene safe for emergency personnel?",
            "What safety concerns exist for responders?"
        ]
        
        hazards_on_scene = "None"  # Default to None for safety
        best_score = 0
        HAZARD_CONFIDENCE_THRESHOLD = 0.6  # Higher threshold for hazards - safety critical
        
        negative_responses = ["no", "none", "no hazards", "safe", "not unsafe"]
        
        for question in hazard_questions:
            try:
                result = qa_pipeline(question=question, context=cleaned_transcript[:800])
                answer_lower = result["answer"].lower().strip()
                
                # Check if the answer indicates no hazards with high confidence
                if result["score"] >= HAZARD_CONFIDENCE_THRESHOLD:
                    if any(neg in answer_lower for neg in negative_responses) and len(answer_lower) < 15:
                        hazards_on_scene = "None"
                        print(f"No hazards detected (confidence: {result['score']:.2f})")
                        break
                    # If answer is substantial and confident, use it
                    elif len(answer_lower) > 2 and not any(neg in answer_lower for neg in negative_responses):
                        hazards_on_scene = result["answer"]
                        print(f"Hazards detected: {hazards_on_scene} (confidence: {result['score']:.2f})")
                        break
            except Exception as e:
                print(f"Error in hazards extraction: {str(e)}")
            
        # Return the incident node structure
        incident_node = {
            "transcript": transcript,  # Original transcript
            "nature_of_incident": nature_of_incident,
            "severity_of_incident": severity_of_incident,
            "hazards_on_scene": hazards_on_scene
        }
        
        return incident_node
        
    except Exception as e:
        print(f"Error in incident extraction: {str(e)}")
        # Fallback with minimal structure - empty strings instead of guesses
        return {
            "transcript": transcript,
            "nature_of_incident": "",
            "severity_of_incident": "",  # Default middle severity
            "hazards_on_scene": "None"
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