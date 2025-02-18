import pickle
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

###############################################################################
#                            Neo4j Functions
###############################################################################
def connect_to_neo4j():
    driver = GraphDatabase.driver(
        "neo4j+s://09ed30ec.databases.neo4j.io", 
        auth=("neo4j", "N7azcFeea5x3mkCafrBbdjAbptvfVW4RVyKUdtsie70")
    )
    return driver

def _get_incident_transcripts(tx):
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(t:Transcript)
    RETURN i.nature AS nature, t.TEXT AS transcript
    """
    result = tx.run(query)
    return [record.data() for record in result]

def extract_training_data(driver):
    with driver.session() as session:
        data = session.execute_read(_get_incident_transcripts)
        training_df = pd.DataFrame(data)
        training_df.dropna(subset=['nature', 'transcript'], inplace=True)
        return training_df

def preprocess_training_data(df):
    def clean_transcript(text):
        cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().lower()
    
    df["clean_transcript"] = df["transcript"].apply(clean_transcript)
    df["clean_nature"] = df["nature"].str.strip().str.lower()
    
    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])
    
    return df, le

###############################################################################
#                  Training and Saving the Model
###############################################################################
def train_and_save_model(training_df, le):
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    
    return "Model trained and saved successfully."

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        driver = connect_to_neo4j()
        training_df = extract_training_data(driver)
        training_df, le = preprocess_training_data(training_df)
        
        message = train_and_save_model(training_df, le)
        driver.close()
        
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

###############################################################################
#                     Load Model and Predict
###############################################################################
def load_model():
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return clf, vectorizer, le

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        transcript = data.get("transcript", "").strip()
        
        if not transcript:
            return jsonify({"status": "error", "message": "No transcript provided."}), 400
        
        clf, vectorizer, le = load_model()
        
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

if __name__ == "__main__":
    app.run(debug=True)
