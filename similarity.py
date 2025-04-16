import re
import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from GNN_Workflow import extract_all_911_call_data, app
from import_to_neo4j import process_batch_file

# Sample JSON data (assumed to be provided)
data1 = {
    "incident": {
        "metadata": {
            "clean_address_EMS": "317 Chelsea St Delaware OH 43015 USA",
            "start": "1/1/2021 01:08"
        },
        "transcript": (
            "0002.0s 0002.5s SPEAKER_01: Umm ... Caller reports being in a car at the front "
            "of a neighborhood complex, experiencing head pain and shortness of breath. ..."
        ),
        "summary": (
            "Caller reports being in a car at the front of a neighborhood complex, experiencing "
            "head pain and shortness of breath."
        )
    }
}

data2 = {
    "incident": {
        "metadata": {
            "clean_address_EMS": "317 Chelsea Street, Delaware, OH 43015, USA",
            "start": "1/1/2021 01:30"
        },
        "transcript": (
            "0001.0s 0002.0s SPEAKER_A: Hello ... Caller states that they are located in front "
            "of a residential complex with symptoms including head pain and difficulty breathing. ..."
        ),
        "summary": (
            "Caller states that they are outside a residential complex, suffering from head pain "
            "and breathing difficulties."
        )
    }
}

# ---------------------------
# Extraction and Utility Functions
# ---------------------------
def extract_address_zip_time(json_data):
    """
    Extracts the address, ZIP code, and call time from the given JSON data.
    Assumes:
      - JSON structure has an 'incident' key with a nested 'metadata' dict.
      - 'clean_address_EMS' contains the address.
      - 'start' field provides the time in "%m/%d/%Y %H:%M" format.
    """
    metadata = json_data.get("incident", {}).get("metadata", {})
    address = metadata.get("clean_address_EMS", "")
    
    # Extract the ZIP code: look for a 5-digit sequence.
    zip_search = re.search(r'\b\d{5}\b', address)
    zip_code = zip_search.group(0) if zip_search else None
    
    # Parse the call time
    start_time_str = metadata.get("start", "")
    call_time = None
    if start_time_str:
        try:
            call_time = datetime.strptime(start_time_str, "%m/%d/%y %H:%M")
        except ValueError as ve:
            print("Time parsing error:", ve)
    
    return address, zip_code, call_time

def is_time_within_one_hour(time1, time2):
    """
    Returns True if the absolute difference between time1 and time2 is one hour or less.
    """
    if time1 is None or time2 is None:
        return False
    delta_seconds = abs((time1 - time2).total_seconds())
    return delta_seconds <= 3600

def jaccard_similarity(str1, str2):
    """
    Computes Jaccard similarity between two strings based on token overlap.
    """
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def tfidf_similarity(text1, text2):
    """
    Computes cosine similarity between two texts based on their TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# ---------------------------
# Main Comparison Logic
# ---------------------------
def compare_json_records(data1, data2):
    # --- Extract Address, ZIP, and Call Time ---
    address1, zip1, time1 = extract_address_zip_time(data1)
    address2, zip2, time2 = extract_address_zip_time(data2)
    
    # print("Record 1:")
    # print("  Address:", address1)
    # print("  ZIP Code:", zip1)
    # print("  Call Time:", time1)
    # print("\nRecord 2:")
    # print("  Address:", address2)
    # print("  ZIP Code:", zip2)
    # print("  Call Time:", time2)
    
    # --- Compare Addresses ---
    same_zip = (zip1 == zip2) if (zip1 and zip2) else False
    address_jaccard = jaccard_similarity(address1, address2)
    # print("\nAddress Comparison:")
    # print("  Same ZIP Code:", same_zip)
    # print("  Address Jaccard Similarity:", round(address_jaccard, 4))
    
    # --- Compare Call Times ---
    time_within_one_hour = is_time_within_one_hour(time1, time2)
    # print("\nCall Time Comparison:")
    # print("  Calls within one hour:", time_within_one_hour)
    
    # --- Compare Transcript and Summary using Jaccard and TF-IDF ---
    transcript1 = data1.get("incident", {}).get("transcript", "")
    transcript2 = data2.get("incident", {}).get("transcript", "")
    summary1 = data1.get("incident", {}).get("summary", "")
    summary2 = data2.get("incident", {}).get("summary", "")
    
    transcript_jaccard = jaccard_similarity(transcript1, transcript2)
    transcript_tfidf = tfidf_similarity(transcript1, transcript2)
    summary_jaccard = jaccard_similarity(summary1, summary2)
    summary_tfidf = tfidf_similarity(summary1, summary2)
    
    # print("\nTranscript Comparison:")
    # print("  Jaccard Similarity:", round(transcript_jaccard, 4))
    # print("  TF-IDF Cosine Similarity:", round(transcript_tfidf, 4))
    
    # print("\nSummary Comparison:")
    # print("  Jaccard Similarity:", round(summary_jaccard, 4))
    # print("  TF-IDF Cosine Similarity:", round(summary_tfidf, 4))

    return same_zip, address_jaccard, time_within_one_hour, transcript_jaccard, transcript_tfidf, summary_jaccard, summary_tfidf


def call_process_911_call_api(transcript):

    with app.test_client() as client:
        # Define the JSON payload to send with the POST request
        # print('testing API client')
        payload = {
            "transcript": transcript
        }
        
        # Use the test client to post the data to the endpoint
        response = client.post("/process_911_call", json=payload)
        
        # Print the status code and the JSON response
        print("Status Code:", response.status_code)
        # print("Response JSON:", response.get_json())
        return response.get_json()
    # # The API endpoint URL
    # url = 'http://127.0.0.1:5000/process_911_call'

    # # The JSON payload you want to send in your POST request
    # payload = {
    #     "trancript": transcript
    # }
    # response = requests.post(url, json=payload)
    # print(response)

    # try:
    #     # Make the POST request with the JSON payload
    #     response = requests.post(url, json=payload)
        
    #     # Raise an exception if the request returned an unsuccessful status code
    #     response.raise_for_status()

    #     # Attempt to decode the JSON response
    #     data = response.json()
    #     print("Success!")
    #     return data

    # except requests.exceptions.HTTPError as http_err:
    #     # Exception raised for HTTP errors
    #     print(f"HTTP error occurred: {http_err}")
        
    # except requests.exceptions.RequestException as req_err:
    #     # Exception raised for other errors during the request (connection errors, timeouts, etc.)
    #     print(f"Request error occurred: {req_err}")
        
    # except ValueError:
    #     # If response.json() fails to decode the JSON response
    #     print("Failed to decode JSON from response")

def compare_new_data_to_data_pull(new_data):
    data_pull = []
    with open('data_pull.json', 'r') as file:
        loaded_content = json.load(file)
    # Check if the loaded content is a string
    if isinstance(loaded_content, str):
        # Parse the string to a Python object
        data_pull = json.loads(loaded_content)
    else:
        data_pull = loaded_content

    for data in data_pull:
        
        same_zip, address_jaccard, time_within_one_hour, transcript_jaccard, transcript_tfidf, summary_jaccard, summary_tfidf = compare_json_records(data, new_data)

        if same_zip or address_jaccard > 0.1 or time_within_one_hour or transcript_jaccard > 0.1 \
            or summary_jaccard > 0.1 or transcript_tfidf > 0.1 or summary_tfidf > 0.1:
            print('#'*50 + '\n\n')
            print("Potentially similar records found:")
            print("Record from data pull:")
            print(data)
            print("\n\nNew record:")
            print(new_data)
            print("\nAddress Comparison:")
            print("  Same ZIP Code:", same_zip)
            print("  Address Jaccard Similarity:", round(address_jaccard, 4))
            print("\nCall Time Comparison:")
            print("  Calls within one hour:", time_within_one_hour)

            print("\nTranscript Comparison:")
            print("  Jaccard Similarity:", round(transcript_jaccard, 4))
            print("  TF-IDF Cosine Similarity:", round(transcript_tfidf, 4))
            
            print("\nSummary Comparison:")
            print("  Jaccard Similarity:", round(summary_jaccard, 4))
            print("  TF-IDF Cosine Similarity:", round(summary_tfidf, 4))
            print('#'*50 + '\n\n')


def process_csv_data(filename):

    data_df = pd.read_csv(filename)
    for index, row in data_df.iterrows():

        transcript_csv_to_json = {"results": []}
        # Extract the transcript from the row
        transcript = row['TEXT']
        address = row['clean_address_EMS']
        start_date = row['start']
        metadata = {col: row[col] for col in data_df.columns if col != "TEXT"}
        row_idx = index + 1
        # Call the API with the transcript
        print("Processing row:", row_idx)
        processed_call_data = call_process_911_call_api(transcript)
        processed_call_data['call_data']['metadata'] = metadata
        processed_call_data['call_data']['row_index'] = row_idx

        summary = processed_call_data['call_data'].get('incident', {}).get('summary', '')
        processed_transcript = processed_call_data['call_data'].get('incident', {}).get('transcript', '')

        new_data_to_compare = {
            'incident': {
                'metadata': metadata,
                'transcript': processed_transcript,
                'summary': summary
            }
        }
        print(new_data_to_compare)
        compare_new_data_to_data_pull(new_data_to_compare)

        transcript_csv_to_json['results'].append(processed_call_data)
        # print(transcript)
        # print(address)
        # print(start_date)
        # print(processed_call_data)
    
        # with open('processed_data.json', 'w') as json_file:
        #     json.dump(transcript_csv_to_json, json_file, indent=4)

def save_csv_to_json():
    {
        "results": [
            {
            "incident": {},
            "calls": [
                {}
            ],
            "persons": [
                {},
                {}
            ],
            "location": {},
            "row_index": 780,
            "metadata": {}
            }
            ]
    }  
    
if __name__ == "__main__":
    process_csv_data('single_data.csv')
    # compare_json_records(data1, data2)
