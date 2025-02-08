#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

###############################################################################
#                            Neo4j Queries
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
#         Train and Evaluate with Different Text Encoding Methods
###############################################################################
def train_and_evaluate_encodings(training_df, le):
    # Split data into training and testing sets (80% train, 20% test)
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define a set of vectorizers (text encoding methods) to compare.
    vectorizers = {
        "TfidfVectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        "CountVectorizer": CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
        # For HashingVectorizer, setting alternate_sign=False avoids negative feature values,
        # which some models (or further processing) might not expect.
        "HashingVectorizer": HashingVectorizer(n_features=5000, ngram_range=(1, 2), stop_words='english', alternate_sign=False)
    }
    
    results = {}
    
    for vec_name, vectorizer in vectorizers.items():
        print(f"\n=== Training with {vec_name} ===")
        
        # For vectorizers that require fitting (Tfidf and Count), we fit on the training data.
        # HashingVectorizer is stateless, so we only transform.
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
        
        # Save the results for later comparison
        results[vec_name] = {
            "vectorizer": vectorizer,
            "model": clf,
            "accuracy": accuracy,
            "report": report,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        # Save detailed test results to an Excel file
        test_results_df = pd.DataFrame({
            "transcript": X_test,
            "actual_nature": le.inverse_transform(y_test),
            "predicted_nature": le.inverse_transform(y_pred)
        })
        excel_filename = f"detailed_test_results_{vec_name}.xlsx"
        test_results_df.to_excel(excel_filename, index=False)
        print(f"[INFO] Detailed test results written to '{excel_filename}'.")
        
        # Also write a summary text file
        txt_filename = f"test_results_{vec_name}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("Accuracy: {:.4f}\n\n".format(accuracy))
            f.write("Classification Report:\n")
            f.write(report)
        print(f"[INFO] Summary results written to '{txt_filename}'.")
        
    # Print a summary table comparing each vectorizer's performance.
    print("\n=== Summary of Encoding Methods ===")
    for vec_name, result in results.items():
        print(f"{vec_name}: Accuracy = {result['accuracy']:.4f}")
    
    return results

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

    print("\n=== Step 3: Preprocess the Training Data ===")
    training_df, le = preprocess_training_data(training_df)
    print("Preprocessed training data sample:")
    print(training_df[['clean_nature', 'clean_transcript', 'nature_label']].head())
    training_df.to_excel("preprocessed_training_data.xlsx", index=False)
    print("[INFO] Preprocessed training data written to 'preprocessed_training_data.xlsx'.")
    
    print("\n=== Step 4: Train and Evaluate Models with Different Text Encodings ===")
    _ = train_and_evaluate_encodings(training_df, le)
    
    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()
