import re
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from GNN_Workflow import extract_all_911_call_data, app
from import_to_neo4j import process_batch_file, connect_to_neo4j

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("911_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Type aliases for clarity
AddressData = Tuple[str, Optional[str], Optional[datetime]]
SimilarityScores = Tuple[bool, float, bool, float, float, float, float]
JSONData = Dict[str, Any]

class CallDataExtractor:
    """Class to extract and process 911 call data from different sources."""
    
    @staticmethod
    def extract_address_zip_time(json_data: JSONData) -> AddressData:
        """
        Extracts the address, ZIP code, and call time from the given JSON data.
        
        Args:
            json_data: JSON object containing incident data
            
        Returns:
            Tuple containing (address, zip_code, call_time)
        """
        metadata = json_data["metadata"]
        address = metadata["clean_address_EMS"]
        
        # Extract the ZIP code: look for a 5-digit sequence.
        zip_search = re.search(r'\b\d{5}\b', address)
        zip_code = zip_search.group(0) if zip_search else None
        
        # Parse the call time
        start_time_str = metadata["start"]
        call_time = None
        if start_time_str:
            try:
                # Try multiple date formats for robustness
                for fmt in ["%m/%d/%Y %H:%M", "%m/%d/%y %H:%M", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        call_time = datetime.strptime(start_time_str, fmt)
                        break
                    except ValueError:
                        continue
                        
                if call_time is None:
                    logger.warning(f"Could not parse time: {start_time_str}")
            except Exception as e:
                logger.error(f"Error parsing time {start_time_str}: {e}")
        

        return address, zip_code, call_time

class SimilarityAnalyzer:
    """Class to compute various similarity metrics between call records."""
    
    @staticmethod
    def is_time_within_one_hour(time1: Optional[datetime], time2: Optional[datetime]) -> bool:
        """
        Returns True if the absolute difference between time1 and time2 is one hour or less.
        
        Args:
            time1: First datetime object
            time2: Second datetime object
            
        Returns:
            Boolean indicating if times are within one hour of each other
        """
        if time1 is None or time2 is None:
            return False
        delta_seconds = abs((time1 - time2).total_seconds())
        return delta_seconds <= 3600

    @staticmethod
    def jaccard_similarity(str1: str, str2: str) -> float:
        """
        Computes Jaccard similarity between two strings based on token overlap.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not str1 or not str2:
            return 0.0
            
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    def tfidf_similarity(text1: str, text2: str) -> float:
        """
        Computes cosine similarity between two texts based on their TF-IDF vectors.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            TF-IDF cosine similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {e}")
            return 0.0

class CallRecordComparer:
    """Class to compare 911 call records for similarity."""
    
    def __init__(self, similarity_thresholds: Dict[str, float] = None):
        """
        Initialize the comparer with configurable similarity thresholds.
        
        Args:
            similarity_thresholds: Dictionary of threshold values for different metrics
        """
        self.extractor = CallDataExtractor()
        self.analyzer = SimilarityAnalyzer()
        
        # Default thresholds
        self.thresholds = {
            'address_jaccard': 0.5,
            'transcript_jaccard': 0.5,
            'transcript_tfidf': 0.8,
            'summary_jaccard': 0.5,
            'summary_tfidf': 0.6
        }
        
        # Update with custom thresholds if provided
        if similarity_thresholds:
            self.thresholds.update(similarity_thresholds)
    
    def compare_json_records(self, data1: JSONData, data2: JSONData) -> SimilarityScores:
        """
        Compare two JSON records and compute similarity metrics.
        
        Args:
            data1: First JSON record
            data2: Second JSON record
            
        Returns:
            Tuple of similarity scores
        """

        address1, zip1, time1 = self.extractor.extract_address_zip_time(data1)
        address2, zip2, time2 = self.extractor.extract_address_zip_time(data2)

        # Compare Addresses
        same_zip = (zip1 == zip2) if (zip1 and zip2) else False
        address_jaccard = self.analyzer.jaccard_similarity(address1, address2)
        
        # Compare Call Times
        time_within_one_hour = self.analyzer.is_time_within_one_hour(time1, time2)
        
        # Compare Transcript and Summary using Jaccard and TF-IDF
        transcript1 = data1["transcript"]
        transcript2 = data2["transcript"]
        summary1 = data1["summary"]
        summary2 = data2["summary"]
        
        transcript_jaccard = self.analyzer.jaccard_similarity(transcript1, transcript2)
        transcript_tfidf = self.analyzer.tfidf_similarity(transcript1, transcript2)
        summary_jaccard = self.analyzer.jaccard_similarity(summary1, summary2)
        summary_tfidf = self.analyzer.tfidf_similarity(summary1, summary2)

        return (
            same_zip, 
            address_jaccard, 
            time_within_one_hour, 
            transcript_jaccard, 
            transcript_tfidf, 
            summary_jaccard, 
            summary_tfidf
        )
    
    def is_similar(self, scores: SimilarityScores) -> bool:
        """
        Determine if two records are similar based on the computed scores.
        
        Args:
            scores: Tuple of similarity scores from compare_json_records
            
        Returns:
            Boolean indicating if records are similar
        """
        same_zip, address_jaccard, time_within_one_hour, transcript_jaccard, transcript_tfidf, summary_jaccard, summary_tfidf = scores
        
        return ((same_zip and time_within_one_hour) and
                address_jaccard > self.thresholds['address_jaccard'] and
                (transcript_jaccard > self.thresholds['transcript_jaccard']  or transcript_tfidf > self.thresholds['transcript_tfidf']) and 
                (summary_jaccard > self.thresholds['summary_jaccard'] or summary_tfidf > self.thresholds['summary_tfidf']))
    
    def format_comparison_report(self, data1: JSONData, data2: JSONData, scores: SimilarityScores) -> str:
        """
        Format a detailed comparison report between two records.
        
        Args:
            data1: First JSON record
            data2: Second JSON record
            scores: Tuple of similarity scores
            
        Returns:
            Formatted report string
        """
        same_zip, address_jaccard, time_within_one_hour, transcript_jaccard, transcript_tfidf, summary_jaccard, summary_tfidf = scores
        
        report = []
        report.append("#" * 50 + "\n\n")
        report.append("Potentially similar records found:")
        report.append("Record from data pull:")
        report.append(json.dumps(data1, indent=2))
        report.append("\n\nNew record:")
        report.append(json.dumps(data2, indent=2))
        report.append("\nAddress Comparison:")
        report.append(f"  Same ZIP Code: {same_zip}")
        report.append(f"  Address Jaccard Similarity: {round(address_jaccard, 4)}")
        report.append("\nCall Time Comparison:")
        report.append(f"  Calls within one hour: {time_within_one_hour}")
        report.append("\nTranscript Comparison:")
        report.append(f"  Jaccard Similarity: {round(transcript_jaccard, 4)}")
        report.append(f"  TF-IDF Cosine Similarity: {round(transcript_tfidf, 4)}")
        report.append("\nSummary Comparison:")
        report.append(f"  Jaccard Similarity: {round(summary_jaccard, 4)}")
        report.append(f"  TF-IDF Cosine Similarity: {round(summary_tfidf, 4)}")
        report.append("#" * 50 + "\n\n")
        
        return "\n".join(report)

class API911Processor:
    """Class to interact with the 911 call processing API."""
    
    def __init__(self, api_url: str = None):
        """
        Initialize with the API URL or use test client.
        
        Args:
            api_url: URL of the external API (if None, use test client)
        """
        self.api_url = api_url
        self.use_test_client = api_url is None
    
    def process_transcript(self, transcript: str) -> JSONData:
        """
        Send a transcript to the API for processing.
        
        Args:
            transcript: The 911 call transcript text
            
        Returns:
            Processed data from the API
        """
        if self.use_test_client:
            return self._use_test_client(transcript)
        else:
            return self._use_external_api(transcript)
    
    def _use_test_client(self, transcript: str) -> JSONData:
        """Use the Flask test client to process a transcript."""
        try:
            with app.test_client() as client:
                payload = {"transcript": transcript}
                response = client.post("/process_911_call", json=payload)
                
                logger.info(f"API Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    return response.get_json()
                else:
                    logger.error(f"API Error: {response.status_code}")
                    return {"error": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Test client error: {e}")
            return {"error": str(e)}
    
    def _use_external_api(self, transcript: str) -> JSONData:
        """Send a request to an external API endpoint."""
        try:
            payload = {"transcript": transcript}
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return {"error": f"HTTP error: {http_err}"}
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
            return {"error": f"Request error: {req_err}"}
        except ValueError:
            logger.error("Failed to decode JSON from response")
            return {"error": "Failed to decode JSON response"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": f"Unexpected error: {e}"}

class DataManager:
    """Class to manage loading and saving data."""
    
    @staticmethod
    def load_data_pull(file_path: str) -> List[JSONData]:
        """
        Load existing data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of JSON records
        """
        try:
            with open(file_path, 'r') as file:
                loaded_content = json.load(file)
                
            # Check if the loaded content is a string
            if isinstance(loaded_content, str):
                # Parse the string to a Python object
                return json.loads(loaded_content)
            else:
                return loaded_content
                
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    @staticmethod
    def save_data(data: Union[List, Dict], file_path: str, indent: int = 4) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save the file
            indent: JSON indentation level
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=indent)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
            return False

class Call911Processor:
    """Main class to orchestrate the 911 call processing workflow."""
    
    def __init__(
        self, 
        data_pull_path: str = 'data_pull.json',
        api_url: str = None,
        similarity_thresholds: Dict[str, float] = None
    ):
        """
        Initialize the processor with configurable parameters.
        
        Args:
            data_pull_path: Path to the existing data file
            api_url: URL for the 911 call processing API
            similarity_thresholds: Custom thresholds for similarity metrics
        """
        self.data_pull_path = data_pull_path
        self.api_processor = API911Processor(api_url)
        self.comparer = CallRecordComparer(similarity_thresholds)
        self.data_manager = DataManager()
    
    def compare_new_data_to_data_pull(self, new_data: JSONData) -> List[Tuple[JSONData, SimilarityScores]]:
        """
        Compare new data against all existing records.
        
        Args:
            new_data: New 911 call record to compare
            
        Returns:
            List of similar records with their similarity scores
        """
        similar_records = []
        is_similar_flag = False
        data_pull = self.data_manager.load_data_pull(self.data_pull_path)
        
        for data in data_pull:
            scores = self.comparer.compare_json_records(data, new_data)
            
            if self.comparer.is_similar(scores):
                is_similar_flag = True
                logger.info("Found similar record")
                print(self.comparer.format_comparison_report(data, new_data, scores))
                similar_records.append((data, scores))
        
        return is_similar_flag, similar_records
    
    def process_csv_data(self, filename: str, save_results: bool = True) -> List[JSONData]:
        """
        Process a CSV file containing 911 call data.
        
        Args:
            filename: Path to the CSV file
            save_results: Whether to save processed results
            
        Returns:
            List of processed call data
        """
        results = []
        
        try:
            data_df = pd.read_csv(filename)
            total_rows = len(data_df)
            logger.info(f"Processing {total_rows} rows from {filename}")
            
            for index, row in data_df.iterrows():
                row_idx = index + 1
                logger.info(f"Processing row {row_idx}/{total_rows}")
                
                # Extract data from the row
                transcript = row['TEXT']
                metadata = {col: row[col] for col in data_df.columns if col != "TEXT"}
                
                # Call the API with the transcript
                processed_call_data = self.api_processor.process_transcript(transcript)
                
                if "error" in processed_call_data:
                    logger.error(f"Error processing row {row_idx}: {processed_call_data['error']}")
                    continue
                
                # Add metadata and row index
                processed_call_data['call_data']['metadata'] = metadata
                processed_call_data['call_data']['row_index'] = row_idx
                
                # Extract required fields for comparison
                summary = processed_call_data['call_data'].get('incident', {}).get('summary', '')
                processed_transcript = processed_call_data['call_data'].get('incident', {}).get('transcript', '')
                
                # Create new data object for comparison
                new_data_to_compare = {
                        'metadata': metadata,
                        'transcript': processed_transcript,
                        'summary': summary
                }
                
                # Compare with existing data
                is_similar_flag, similar_record = self.compare_new_data_to_data_pull(new_data_to_compare)
                
                # Add to results
                results.append(processed_call_data['call_data'])
            
                # Save processed data if requested
                if save_results and results:
                    output = {"results": results}
                    self.data_manager.save_data(output, 'processed_data.json')

                if is_similar_flag:
                    logger.info(f"Similar records found for row {row_idx}")
                else:
                    logger.info(f"No similar records found for row {row_idx}")
                    logger.info(f"Pushing the record to Neo4j")
                    # driver = connect_to_neo4j()
                    # process_batch_file('processed_data.json', driver) #Our data is different hence error while pushing, you can just comment out and similairty will work fine
                
        
            
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {filename}")
            return []
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing CSV {filename}: {e}")
            return []

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 911 call data and find similar records.')
    parser.add_argument('--csv', type=str, default='single_data.csv', help='Path to CSV file with 911 call data')
    parser.add_argument('--data-pull', type=str, default='data_pull.json', 
                        help='Path to existing data pull JSON file')
    parser.add_argument('--api-url', type=str, help='URL for external 911 call processing API')
    parser.add_argument('--save', action='store_true', help='Save processed results')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Configure processor
    processor = Call911Processor(
        data_pull_path=args.data_pull,
        api_url=args.api_url
    )
    
    # Process CSV if provided
    if args.csv:
        processor.process_csv_data(args.csv, save_results=args.save)
    else:
        logger.error("No CSV file specified. Use --csv to specify a file.")
        parser.print_help()

if __name__ == "__main__":
    main()