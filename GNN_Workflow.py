#!/usr/bin/env python

from flask import Flask, jsonify, request
from flask_cors import CORS  # Import flask-cors
import pandas as pd
import re
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

###############################################################################
#                            Neo4j Functions
###############################################################################
def connect_to_neo4j(uri=None, username=None, password=None):
    HARDCODED_URI = "neo4j+s://09ed30ec.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "N7azcFeea5x3mkCafrBbdjAbptvfVW4RVyKUdtsie70"
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
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
        print("Training DataFrame columns:", training_df.columns)
        print("First few rows of training data:")
        print(training_df.head())
        training_df.dropna(subset=['nature', 'transcript'], inplace=True)
        return training_df

def preprocess_training_data(df):
    def clean_transcript(text):
        # Remove timestamps and speaker markers, then clean extra whitespace and lowercase the text
        cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().lower()
    
    df["clean_transcript"] = df["transcript"].apply(clean_transcript)
    df["clean_nature"] = df["nature"].str.strip().str.lower()
    
    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])
    
    return df, le

###############################################################################
#         Training and Evaluation Functions
###############################################################################
def train_and_evaluate_encodings(training_df, le):
    # Split data into training and testing sets (80% train, 20% test)
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define a set of vectorizers to compare.
    vectorizers = {
        "TfidfVectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "CountVectorizer": CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "HashingVectorizer": HashingVectorizer(n_features=5000, ngram_range=(1, 2), stop_words='english', alternate_sign=False)
    }
    
    results = {}
    
    for vec_name, vectorizer in vectorizers.items():
        print(f"\n=== Training with {vec_name} ===")
        # For HashingVectorizer, only transform (since it's stateless)
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
#                   Flask Endpoint for Model Training
###############################################################################
@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Connect to Neo4j and get the data
        driver = connect_to_neo4j()
        training_df = extract_training_data(driver)
        
        # Preprocess the data and encode labels
        training_df, le = preprocess_training_data(training_df)
        
        # Train models using different vectorizers and get results
        results = train_and_evaluate_encodings(training_df, le)
        
        driver.close()
        
        # Return a summary of the results as JSON
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
