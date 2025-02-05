#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import openai
import math

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





client = openai.OpenAI(api_key="inputkeyhere")

def call_openai_model_batch(transcripts, batch_size=100):
    """
    Calls OpenAI's GPT model in batches to classify incident transcripts.
    Ensures that the number of predictions matches the number of inputs.
    """
    predictions = []
    num_batches = math.ceil(len(transcripts) / batch_size)

    for i in range(num_batches):
        batch = transcripts[i * batch_size: (i + 1) * batch_size]
        batch_prompt = "\n\n".join(
            [f"Transcript {idx+1}: {text}" for idx, text in enumerate(batch)]
        )

        full_prompt = (
            "You are an AI trained to classify emergency call transcripts.\n"
            "Classify each transcript into an incident category. Provide one classification per transcript.\n\n"
            + batch_prompt
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": full_prompt}]
        )

        response_text = response.choices[0].message.content.strip().lower()
        batch_predictions = response_text.split("\n")

        # ✅ Ensure batch_predictions is the same length as batch
        if len(batch_predictions) != len(batch):
            print(f"[WARNING] OpenAI returned {len(batch_predictions)} predictions for {len(batch)} transcripts. Adjusting length.")
            batch_predictions = batch_predictions[:len(batch)]  # Trim excess predictions
            batch_predictions += ["unknown"] * (len(batch) - len(batch_predictions))  # Fill missing predictions

        predictions.extend(batch_predictions)

    return predictions



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
        cleaned = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)# we remove timestamp and speaker markers
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().lower()
    
    df["clean_transcript"] = df["transcript"].apply(clean_transcript)
    df["clean_nature"] = df["nature"].str.strip().str.lower()
    
    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])
    
    return df, le

def train_and_evaluate_model(training_df, le):
    """
    Train a logistic regression model and compare its predictions with OpenAI GPT model.
    Ensures OpenAI returns the same number of predictions as the test set.
    """
    X = training_df["clean_transcript"]
    y = training_df["nature_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    logistic_pred_labels = le.inverse_transform(y_pred)
    actual_labels = le.inverse_transform(y_test)

    # ✅ Batch Process OpenAI API Calls
    openai_predictions = call_openai_model_batch(X_test.tolist(), batch_size=50)

    # ✅ Ensure all arrays have the same length
    min_length = min(len(X_test), len(actual_labels), len(logistic_pred_labels), len(openai_predictions))
    
    X_test = X_test[:min_length]
    actual_labels = actual_labels[:min_length]
    logistic_pred_labels = logistic_pred_labels[:min_length]
    openai_predictions = openai_predictions[:min_length]

    comparison_df = pd.DataFrame({
        "transcript": X_test,
        "actual_nature": actual_labels,
        "logistic_pred": logistic_pred_labels,
        "openai_pred": openai_predictions
    })

    comparison_df.to_excel("comparison_results.xlsx", index=False)
    print("[INFO] Comparison results saved to 'comparison_results.xlsx'.")

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
    
    
    print("\n=== Workflow Complete ===")
    driver.close()

if __name__ == "__main__":
    main()
