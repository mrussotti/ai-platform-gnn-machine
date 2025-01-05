import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib
import ast  # For parsing string representations of lists
import numpy as np  # Import numpy for numerical operations

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class TopKCategories(TransformerMixin, BaseEstimator):
    """
    Custom transformer to keep only the top K categories for each categorical feature.
    Categories outside the top K are replaced with 'Other'.
    """
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.top_categories_ = {}
    
    def fit(self, X, y=None):
        """
        Fit the transformer by finding the top K categories for each column.
        
        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input samples.
        - y: Ignored
        
        Returns:
        - self
        """
        self.top_categories_ = []
        for i in range(X.shape[1]):
            # Ensure that the column is of string type for consistent processing
            col = X[:, i].astype(str)
            unique, counts = np.unique(col, return_counts=True)
            top_k = unique[np.argsort(counts)[-self.top_k:]]
            self.top_categories_.append(set(top_k))
        return self
    
    def transform(self, X):
        """
        Transform the data by replacing less frequent categories with 'Other'.
        
        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input samples.
        
        Returns:
        - X_transformed: np.ndarray, shape (n_samples, n_features)
            The transformed samples.
        """
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            # Ensure that the column is of string type for consistent processing
            col = X_transformed[:, i].astype(str)
            # Replace categories not in top_k with 'Other'
            X_transformed[:, i] = np.where(
                np.isin(col, list(self.top_categories_[i])),
                col,
                'Other'
            )
        return X_transformed

def get_nodes(driver):
    with driver.session() as session:
        return session.execute_read(_get_nodes)  # Replaced read_transaction with execute_read

def _get_nodes(tx):
    query = """
    MATCH (n)
    RETURN n.tmdbId AS id, labels(n) AS labels, properties(n) AS properties
    """
    result = tx.run(query)
    nodes = []
    for record in result:
        node_data = record["properties"]
        node_data["id"] = record["id"]  # Now refers to tmdbId
        node_data["labels"] = record["labels"]
        nodes.append(node_data)
    return nodes

def get_relationships(driver):
    with driver.session() as session:
        return session.execute_read(_get_relationships)  # Replaced read_transaction with execute_read

def _get_relationships(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN r.tmdbId AS id, n.tmdbId AS start_id, m.tmdbId AS end_id, type(r) AS type, properties(r) AS properties
    """
    result = tx.run(query)
    relationships = []
    for record in result:
        rel_data = record["properties"]
        rel_data["id"] = record["id"]  # Now refers to r.tmdbId
        rel_data["start_id"] = record["start_id"]  # n.tmdbId
        rel_data["end_id"] = record["end_id"]  # m.tmdbId
        rel_data["type"] = record["type"]
        relationships.append(rel_data)
    return relationships

def connect_to_neo4j(uri=None, username=None, password=None):
    # Hardcoded connection details
    HARDCODED_URI = "neo4j+s://ae86bd83.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "h0ZknFPHLkSPXFU_O0eEI1_VG-AnHa-p1uN6NfrNdFY"
    
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    # Convert them to DataFrames
    nodes_df = pd.DataFrame(nodes)
    relationships_df = pd.DataFrame(relationships)
    return nodes_df, relationships_df

def identify_missing_attributes(nodes_df):
    # List out the columns in the nodes_df (excluding id/labels if you like)
    available_attributes = list(nodes_df.columns)
    print("\nAvailable Attributes in nodes_df:")
    for attr in available_attributes:
        print(f"  - {attr}")

    # For simplicity, let's just have the user pick one attribute
    chosen_attribute = input(
        "\nEnter the attribute you want to model/infer (e.g. 'age' or 'status'): "
    )

    # If the user picks an invalid attribute, handle it gracefully
    if chosen_attribute not in available_attributes:
        print(
            f"[WARNING] '{chosen_attribute}' not found in the DataFrame. "
            "Please ensure your input matches one of the listed attributes.\n"
        )
        return pd.DataFrame(), []  # Return empty DataFrame and empty attribute list

    # Identify nodes that have that attribute missing (NaN or empty string)
    missing_mask = (nodes_df[chosen_attribute].isna()) | (nodes_df[chosen_attribute] == "")
    missing_attributes_df = nodes_df[missing_mask].copy()

    # Pack chosen attributes in a list
    key_attributes = [chosen_attribute]

    return missing_attributes_df, key_attributes

def preprocess_data(nodes_df, relationships_df, target_attributes, verbose=True):
    """
    Preprocesses data for AI model training dynamically based on the target attributes.

    **Inputs:**
    - nodes_df (pd.DataFrame): DataFrame containing node information.
    - relationships_df (pd.DataFrame): DataFrame containing relationship information.
    - target_attributes (list of str): List of attributes to model/infer.

    **Outputs:**
    - preprocessed_features (torch.Tensor): Processed feature set ready for model training.
    - labels (torch.Tensor): Labels for supervised learning tasks.
    """
    if verbose:
        print("\n--- Preprocess Data Function Started ---")
        print(f"Number of nodes received: {len(nodes_df)}")
        print(f"Number of relationships received: {len(relationships_df)}")
        print(f"Target attributes: {target_attributes}")

    # Ensure target_attributes is a list
    if isinstance(target_attributes, str):
        target_attributes = [target_attributes]
        if verbose:
            print(f"Converted target_attributes to list: {target_attributes}")

    # Initialize dictionaries to hold features and labels for each target attribute
    feature_dict = {}
    label_dict = {}

    for target in target_attributes:
        if verbose:
            print(f"\nProcessing target attribute: '{target}'")

        # Step 1: Filter nodes with the target attribute present for training
        initial_count = len(nodes_df)
        training_df = nodes_df.dropna(subset=[target]).copy()
        filtered_count = len(training_df)
        if verbose:
            print(f"Filtered nodes: {filtered_count} out of {initial_count} have '{target}' present.")

        # Step 1.1: Verify no duplication after filtering
        if training_df.duplicated(subset=['tmdbId']).any():
            duplicated = training_df.duplicated(subset=['tmdbId']).sum()
            print(f"[WARNING] Found {duplicated} duplicated 'tmdbId's in training data.")
            # Optionally, remove duplicates
            training_df = training_df.drop_duplicates(subset=['tmdbId'])
            if verbose:
                print(f"Dropped duplicated 'tmdbId's. New training data count: {len(training_df)}")

        # Handle different data types for labels
        label_dtype = training_df[target].dtype
        if verbose:
            print(f"Original label dtype for '{target}': {label_dtype}")

        # Attempt to convert to numeric if possible
        try:
            training_df[target] = training_df[target].astype(float)
            labels = training_df[target]
            if verbose:
                print(f"Converted '{target}' to float. dtype after conversion: {training_df[target].dtype}")
        except ValueError:
            # If conversion fails, treat as categorical
            labels = training_df[target]
            if verbose:
                print(f"Could not convert '{target}' to float. Treating as categorical.")

        # Step 2: Drop target and non-informative columns
        non_informative_cols = ['id', 'labels'] + target_attributes
        existing_non_info_cols = [col for col in non_informative_cols if col in training_df.columns]
        training_df = training_df.drop(columns=existing_non_info_cols)
        if verbose:
            print(f"Dropped non-informative columns: {existing_non_info_cols}")

        # Step 3: Handle 'embedding' or similar list-like columns dynamically
        # Identify only original embedding columns, excluding any PCA-transformed columns
        embedding_cols = [col for col in training_df.columns if col.lower() == 'embedding']
        if verbose:
            print(f"Identified embedding columns: {embedding_cols}")

        for emb_col in embedding_cols:
            if verbose:
                print(f"Processing embedding column: '{emb_col}'")
            # Convert string representations of lists to actual lists if necessary
            if training_df[emb_col].dtype == object:
                before_conversion = training_df[emb_col].isna().sum()
                training_df[emb_col] = training_df[emb_col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                after_conversion = training_df[emb_col].isna().sum()
                if verbose:
                    print(f"Converted string embeddings to lists. Missing after conversion: {after_conversion} (was {before_conversion})")

            # Define expected embedding length
            expected_length = 1536

            # Identify embeddings with incorrect lengths
            correct_length_mask = training_df[emb_col].apply(lambda x: isinstance(x, list) and len(x) == expected_length)
            incorrect_length_count = (~correct_length_mask).sum()
            if verbose:
                print(f"Found {incorrect_length_count} embeddings with incorrect length.")

            # Option B: Replace incorrect embeddings with zero vectors
            if incorrect_length_count > 0:
                # Assign zero vectors using .apply to ensure correct alignment
                training_df.loc[~correct_length_mask, emb_col] = training_df.loc[~correct_length_mask, emb_col].apply(lambda x: [0.0]*expected_length)
                if verbose:
                    print(f"Replaced {incorrect_length_count} embeddings with incorrect lengths with zero vectors.")

            # Convert embeddings to numpy array for PCA
            try:
                embeddings = np.vstack(training_df[emb_col].values)
                if verbose:
                    print(f"Embeddings shape before PCA: {embeddings.shape}")
            except ValueError as e:
                print(f"[ERROR] Failed to stack embeddings: {e}")
                raise

            # Apply PCA
            pca = PCA(n_components=300)  # Reduce to 300 dimensions; adjust as needed
            reduced_embeddings = pca.fit_transform(embeddings)
            if verbose:
                print(f"Embeddings shape after PCA: {reduced_embeddings.shape}")

            # Create DataFrame from reduced embeddings
            reduced_embedding_df = pd.DataFrame(
                reduced_embeddings,
                columns=[f'{emb_col}_pca_{i}' for i in range(pca.n_components_)]
            )
            if verbose:
                print(f"Reduced embeddings into {len(reduced_embedding_df.columns)} columns.")

            # Concatenate with the feature dataframe ensuring indices align
            training_df = pd.concat([
                training_df.drop(columns=[emb_col]).reset_index(drop=True), 
                reduced_embedding_df.reset_index(drop=True)
            ], axis=1)
            if verbose:
                print(f"Concatenated reduced embeddings into training dataframe. New training data shape: {training_df.shape}")

        # Step 4: Automatically identify categorical and numerical columns
        categorical_cols = training_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = training_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        boolean_cols = training_df.select_dtypes(include=['bool']).columns.tolist()

        if boolean_cols:
            if verbose:
                print(f"Identified boolean columns: {boolean_cols}")
            categorical_cols += boolean_cols
            numerical_cols = [col for col in numerical_cols if col not in boolean_cols]

        if verbose:
            print(f"Number of categorical columns: {len(categorical_cols)}")
            print(f"Categorical columns: {categorical_cols}")
            print(f"Number of numerical columns: {len(numerical_cols)}")
            print(f"Numerical columns: {numerical_cols}")

        # **Optional Step:** Convert list-like entries in categorical columns to strings
        for col in categorical_cols:
            if training_df[col].apply(lambda x: isinstance(x, list)).any():
                training_df[col] = training_df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
                if verbose:
                    print(f"Converted list-like entries in '{col}' to comma-separated strings.")

        # Exclude identifier columns from categorical features
        identifier_cols = ['tmdbId', 'imdbId', 'name', 'userId']
        categorical_cols = [col for col in categorical_cols if col not in identifier_cols]
        if verbose:
            print(f"Excluded identifier columns from categorical features: {identifier_cols}")
            print(f"Updated categorical columns: {categorical_cols}")

        # Remove categorical columns with all missing values
        cols_with_all_missing = [col for col in categorical_cols if training_df[col].isna().all()]
        if cols_with_all_missing:
            training_df = training_df.drop(columns=cols_with_all_missing)
            categorical_cols = [col for col in categorical_cols if col not in cols_with_all_missing]
            if verbose:
                print(f"Dropped categorical columns with all missing values: {cols_with_all_missing}")
        else:
            if verbose:
                print("No categorical columns with all missing values found.")

        # Step 5: Define preprocessing pipelines
        if verbose:
            print("\nSetting up preprocessing pipelines...")
        # Categorical pipeline with TopKCategories
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('topk', TopKCategories(top_k=100)),  # Keep top 100 categories
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        if verbose:
            print("Defined categorical preprocessing pipeline.")

        # Numerical pipeline
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        if verbose:
            print("Defined numerical preprocessing pipeline.")

        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_pipeline, categorical_cols),
                ('num', numerical_pipeline, numerical_cols)
            ]
        )
        if verbose:
            print("Combined categorical and numerical pipelines into ColumnTransformer.")

        # Step 6: Fit and transform the feature data
        if verbose:
            print("\nFitting and transforming the feature data...")
        try:
            preprocessed_features_np = preprocessor.fit_transform(training_df)
            if verbose:
                print(f"Feature data transformation complete. Shape: {preprocessed_features_np.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to transform features: {e}")
            raise

        # Optional: Convert to a dense array if it's sparse
        if hasattr(preprocessed_features_np, 'toarray'):
            if verbose:
                print("Converting sparse matrix to dense array...")
            preprocessed_features_np = preprocessed_features_np.toarray()
            if verbose:
                print("Conversion to dense array complete.")

        # Step 7: Store preprocessed features and labels
        feature_dict[target] = preprocessed_features_np
        label_dict[target] = labels.values
        if verbose:
            print(f"Stored preprocessed features and labels for target '{target}'.")

    # After processing all targets
    # For simplicity, handle only one target attribute at a time
    if len(target_attributes) == 1:
        target = target_attributes[0]
        if verbose:
            print(f"\nAggregating preprocessed data for target: '{target}'")

        # Convert to torch.Tensor
        try:
            preprocessed_features = torch.tensor(feature_dict[target], dtype=torch.float32)
            if verbose:
                print(f"Converted features to torch.Tensor with shape {preprocessed_features.shape}.")
        except Exception as e:
            print(f"[ERROR] Failed to convert features to torch.Tensor: {e}")
            raise

        # Determine if the task is regression or classification based on label dtype
        if np.issubdtype(label_dict[target].dtype, np.number):
            labels_tensor = torch.tensor(label_dict[target], dtype=torch.float32).unsqueeze(1)  # For regression
            if verbose:
                print(f"Labels are numeric. Converted to torch.Tensor with shape {labels_tensor.shape} for regression.")
        else:
            # For classification, encode labels as integers
            unique_labels = list(set(label_dict[target]))
            label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
            labels_encoded = [label_to_int[label] for label in label_dict[target]]
            labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
            if verbose:
                print(f"Labels are categorical. Encoded and converted to torch.Tensor with shape {labels_tensor.shape} for classification.")

        if verbose:
            print("--- Preprocess Data Function Completed Successfully ---\n")
        return preprocessed_features, labels_tensor
    else:
        # If multiple target attributes, return dictionaries
        if verbose:
            print("\nProcessing multiple target attributes...")
        preprocessed_features_tensor = {}
        labels_tensor = {}

        for target in target_attributes:
            if verbose:
                print(f"\nAggregating preprocessed data for target: '{target}'")
            preprocessed_features_tensor[target] = torch.tensor(feature_dict[target], dtype=torch.float32)
            if verbose:
                print(f"Converted features for '{target}' to torch.Tensor with shape {preprocessed_features_tensor[target].shape}.")

            if np.issubdtype(label_dict[target].dtype, np.number):
                labels_tensor[target] = torch.tensor(label_dict[target], dtype=torch.float32).unsqueeze(1)  # For regression
                if verbose:
                    print(f"Labels for '{target}' are numeric. Converted to torch.Tensor with shape {labels_tensor[target].shape} for regression.")
            else:
                unique_labels = list(set(label_dict[target]))
                label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
                labels_encoded = [label_to_int[label] for label in label_dict[target]]
                labels_tensor[target] = torch.tensor(labels_encoded, dtype=torch.long)
                if verbose:
                    print(f"Labels for '{target}' are categorical. Encoded and converted to torch.Tensor with shape {labels_tensor[target].shape} for classification.")

        if verbose:
            print("\n--- Preprocess Data Function Completed Successfully for Multiple Targets ---\n")
        return preprocessed_features_tensor, labels_tensor



def train_ai_models(preprocessed_features, labels, key_attributes):
    """
    Trains one AI model for each attribute we want to find. 
    Uses Linear Regression for numerical targets and Logistic Regression for categorical targets.

    **Inputs:**
    - preprocessed_features (torch.Tensor): Processed feature set.
    - labels (torch.Tensor): Labels for supervised learning.
    - key_attributes (list of str): List of attributes to model/infer

    **Outputs:**
    - trained_models (dict): Dictionary containing trained models for each attribute.
    """
    trained_models = {}
    target = key_attributes[0]

    # Convert torch tensors back to numpy for scikit-learn
    X = preprocessed_features.numpy()
    y = labels.numpy().flatten()

    # Determine if the task is regression or classification
    if np.issubdtype(y.dtype, np.floating):
        model = LinearRegression()
        task = 'regression'
    elif np.issubdtype(y.dtype, np.integer):
        model = LogisticRegression(max_iter=1000)
        task = 'classification'
    else:
        raise ValueError(f"Unsupported label dtype: {y.dtype}")

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    if task == 'regression':
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        print(f"Trained Linear Regression for '{target}' with MSE: {mse:.4f}")
    else:
        predictions = model.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        print(f"Trained Logistic Regression for '{target}' with Accuracy: {acc:.4f}")

    # Store the trained model
    trained_models[target] = model

    return trained_models

def infer_missing_attributes(trained_models, preprocessed_features, missing_attributes_df):
    """
    Uses trained AI models to infer missing attributes.

    **Inputs:**
    - trained_models (dict): Dictionary containing trained models.
    - preprocessed_features (torch.Tensor): Features for nodes with missing attributes.
    - missing_attributes_df (pd.DataFrame): DataFrame containing nodes with missing attributes.

    **Outputs:**
    - inferred_attributes (dict): Dictionary containing inferred attribute values for each node.
    """
    inferred_attributes = {}
    target = list(trained_models.keys())[0]
    model = trained_models[target]

    # Convert torch tensor to numpy
    X_missing = preprocessed_features.numpy()

    # Predict using the model
    if isinstance(model, LinearRegression):
        predictions = model.predict(X_missing)
    elif isinstance(model, LogisticRegression):
        predictions = model.predict(X_missing)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Store predictions
    inferred_attributes[target] = predictions

    print(f"[INFO] Inferred missing attributes for '{target}'.")

    return inferred_attributes

def generate_report(inferred_attributes):
    """
    Generates and presents a report of the inferred attributes to the user.

    **Inputs:**
    - inferred_attributes (dict): Dictionary containing inferred attribute values.

    **Outputs:**
    - None (saves the report to the specified path)
    """
    # Example: Save the inferred attributes to a CSV file
    for target, predictions in inferred_attributes.items():
        df = pd.DataFrame(predictions, columns=[f'inferred_{target}'])
        df.to_csv(f'inferred_{target}_report.csv', index=False)
        print(f"[INFO] Generated report for '{target}' at 'inferred_{target}_report.csv'.")

def update_neo4j(driver, inferred_attributes, missing_attributes_df, batch_size=1000):
    """
    Updates the Neo4j database with inferred attribute values only if the user wants to.

    **Inputs:**
    - driver (GraphDatabase.driver): The Neo4j driver instance.
    - inferred_attributes (dict): Dictionary containing inferred attribute values.
    - missing_attributes_df (pd.DataFrame): DataFrame containing nodes with missing attributes.
    - batch_size (int): Number of updates per transaction.

    **Outputs:**
    - None
    """
    target = list(inferred_attributes.keys())[0]
    predictions = inferred_attributes[target]
    node_ids = missing_attributes_df['tmdbId'].tolist()

    update_choice = input(f"\nDo you want to update the Neo4j database with the inferred '{target}' values? (yes/no): ").strip().lower()
    if update_choice not in ['yes', 'y']:
        print("[INFO] User opted not to update the Neo4j database.")
        return

    with driver.session() as session:
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i+batch_size]
            batch_predictions = predictions[i:i+batch_size]
            session.write_transaction(update_batch, batch_ids, batch_predictions, target)
    print(f"[INFO] Updated Neo4j database with inferred '{target}' values in batches.")

def update_batch(tx, batch_ids, batch_predictions, target):
    for node_id, prediction in zip(batch_ids, batch_predictions):
        if isinstance(prediction, float):
            # Regression task
            query = """
            MATCH (n)
            WHERE n.tmdbId = $id
            SET n.year = $value
            """
            params = {'id': node_id, 'value': float(prediction)}
        else:
            # Classification task
            query = """
            MATCH (n)
            WHERE n.tmdbId = $id
            SET n.year = $value
            """
            params = {'id': node_id, 'value': int(prediction)}
        tx.run(query, params)

def save_trained_models(trained_models):
    """
    Saves the trained AI models locally. 

    **Inputs:**
    - trained_models (dict): Dictionary containing trained models.

    **Outputs:**
    - None (models are saved to the specified directory)
    """
    for target, model in trained_models.items():
        joblib.dump(model, f'trained_model_{target}.joblib')
        print(f"[INFO] Saved trained model for '{target}' at 'trained_model_{target}.joblib'.")

def main():
    """
    Main function to orchestrate the workflow.

    **Process Flow:**
    1. Connect to Neo4j Graph Database
    2. Extract Nodes and Relationships Data
    3. Identify The Attributes We Want to Inference and Find the Nodes with Missing Attributes
    4. Preprocess Data for AI Model
    5. Train AI Models for Missing Attributes
    6. Infer Missing Attributes Using Trained Models
    7. Generate and Present Inference Report to User
    8. Update Neo4j Database with Inferred Attributes
    9. Save Trained Models Locally
    """
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Nodes and Relationships Data ===")
    nodes_df, relationships_df = extract_data(driver)
    print(f"[INFO] Retrieved {len(nodes_df)} nodes and {len(relationships_df)} relationships.")

    print("\n=== Step 3: Identify Missing Attributes ===")
    missing_attributes_df, key_attributes = identify_missing_attributes(nodes_df)
    if key_attributes:
        print(f"[INFO] Chosen attributes: {key_attributes}")
        print(f"[INFO] Found {len(missing_attributes_df)} nodes missing values for those attributes.")
    else:
        print("[INFO] No valid target attributes selected. Exiting workflow.")
        driver.close()
        return

    # Step 4: Preprocess Data for AI Model
    print("\n=== Step 4: Preprocess Data ===")
    preprocessed_features, labels = preprocess_data(nodes_df, relationships_df, key_attributes)
    if isinstance(preprocessed_features, dict):
        for target, features in preprocessed_features.items():
            print(f"[INFO] Preprocessed features for '{target}' with shape {features.shape}")
        for target, lbl in labels.items():
            print(f"[INFO] Labels for '{target}' with shape {lbl.shape}")
    else:
        print(f"[INFO] Data preprocessed. Feature shape: {preprocessed_features.shape}, Labels shape: {labels.shape}")

    # Step 5: Train AI Models for Missing Attributes
    print("\n=== Step 5: Train AI Models ===")
    trained_models = train_ai_models(preprocessed_features, labels, key_attributes)
    print("[INFO] Models trained.")

    # Step 6: Infer Missing Attributes Using Trained Models
    print("\n=== Step 6: Inference ===")
    inferred_attributes = infer_missing_attributes(trained_models, preprocessed_features, missing_attributes_df)
    print("[INFO] Missing attributes inferred.")

    # Step 7: Generate and Present Inference Report to User
    print("\n=== Step 7: Generate Report ===")
    generate_report(inferred_attributes)
    print("[INFO] Inference report generated.")

    # Step 8: Update Neo4j Database with Inferred Attributes (if user wants)
    print("\n=== Step 8: Update Neo4j Database ===")
    update_neo4j(driver, inferred_attributes, missing_attributes_df)
    print("[INFO] Neo4j updated (if requested by user).")

    # Step 9: Save Trained Models Locally
    print("\n=== Step 9: Save Trained Models ===")
    save_trained_models(trained_models)
    print("[INFO] Trained models saved locally.")

    # Close the driver
    print("\n=== Workflow Complete: Closing Neo4j Driver ===")
    driver.close()

if __name__ == "__main__":
    main()
