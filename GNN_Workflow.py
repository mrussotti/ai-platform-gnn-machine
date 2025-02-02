#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
    Transcript nodes and returns the 'nature' property along with the transcript text.
    """
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(t:Transcript)
    RETURN i.nature AS nature, t.TEXT AS transcript
    """
    result = tx.run(query)
    return [record.data() for record in result]

def extract_training_data(driver):
    """
    Extract training data from the graph.
    
    Returns a DataFrame with columns: 'nature' and 'transcript'.
    """
    with driver.session() as session:
        data = session.execute_read(_get_incident_transcripts)
        training_df = pd.DataFrame(data)
        print("Training DataFrame columns:", training_df.columns)
        print("First few rows of training data:")
        print(training_df.head())
        training_df.dropna(subset=['nature', 'transcript'], inplace=True)
        return training_df

def preprocess_training_data(df):
    """
    Preprocess the training DataFrame for model training.
    
    Cleans the transcript text by removing timestamps and speaker markers,
    collapsing extra whitespace, and converting text to lowercase.
    It also cleans the 'nature' column and creates numeric labels.
    
    Returns:
        df (pandas.DataFrame): DataFrame with additional columns:
            - 'clean_transcript'
            - 'clean_nature'
            - 'nature_label'
        le (LabelEncoder): The fitted LabelEncoder.
    """
    def clean_transcript(text):
        # Remove timestamp and speaker markers (e.g., "0002.0s 0002.5s SPEAKER_01:")
        cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)
        # Replace multiple whitespace (including newlines) with a single space
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().lower()
    
    df["clean_transcript"] = df["transcript"].apply(clean_transcript)
    df["clean_nature"] = df["nature"].str.strip().str.lower()
    
    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])
    
    return df, le

def train_and_evaluate_model(training_df, le):
    """
    Vectorizes the text, splits the data, trains a logistic regression classifier,
    evaluates its performance, and writes the detailed results to an Excel file.
    
    Returns:
        clf: The trained classifier.
        vectorizer: The fitted TF-IDF vectorizer.
    """
    # Extract features and labels
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert text to TF-IDF features with bigrams and stop word removal
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test_vec)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    
    # Create a detailed results DataFrame for test samples
    test_results_df = pd.DataFrame({
        "transcript": X_test,
        "actual_nature": le.inverse_transform(y_test),
        "predicted_nature": le.inverse_transform(y_pred)
    })
    
    # Write the detailed test results to an Excel file
    test_results_df.to_excel("detailed_test_results.xlsx", index=False)
    print("[INFO] Detailed test results written to 'detailed_test_results.xlsx'.")
    
    # Optionally, also write the summary test results to a text file
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("Accuracy: {:.4f}\n\n".format(accuracy))
        f.write("Classification Report:\n")
        f.write(report)
    
    return clf, vectorizer

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

    #chatgptAPIcallmethod()
    
    print("\n=== Step 3: Preprocess the Training Data ===")
    training_df, le = preprocess_training_data(training_df)
    print("Preprocessed training data sample:")
    print(training_df[['clean_nature', 'clean_transcript', 'nature_label']].head())
    training_df.to_excel("preprocessed_training_data.xlsx", index=False)
    print("[INFO] Preprocessed training data written to 'preprocessed_training_data.xlsx'.")
    
    print("\n=== Step 4: Train and Evaluate the Model ===")
    clf, vectorizer = train_and_evaluate_model(training_df, le)
    
    # Optionally, save the trained model and vectorizer for later use (e.g., using joblib)
    # import joblib
    # joblib.dump(clf, 'classifier.joblib')
    # joblib.dump(vectorizer, 'vectorizer.joblib')
    
    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()
