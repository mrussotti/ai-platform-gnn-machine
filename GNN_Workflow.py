#!/usr/bin/env python

import pandas as pd
from neo4j import GraphDatabase
import torch
import joblib
import ast
import numpy as np

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator

# For classification fallback
from sklearn.linear_model import LogisticRegression

# xgboost
from xgboost import XGBRegressor

###############################################################################
#                           Custom Transformer
###############################################################################
class TopKCategories(TransformerMixin, BaseEstimator):
    """
    Custom transformer to keep only the top K categories for each categorical feature.
    Categories outside the top K are replaced with 'Other'.
    """
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.top_categories_ = []
    
    def fit(self, X, y=None):
        self.top_categories_ = []
        for i in range(X.shape[1]):
            # Convert everything to str, ensuring no list-like objects remain
            col = X[:, i].astype(str)
            unique, counts = np.unique(col, return_counts=True)
            # Keep top_k categories
            top_k = unique[np.argsort(counts)[-self.top_k:]]
            self.top_categories_.append(set(top_k))
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            col_as_str = X_transformed[:, i].astype(str)
            # Replace anything not in top_k with 'Other'
            X_transformed[:, i] = np.where(
                np.isin(col_as_str, list(self.top_categories_[i])),
                col_as_str,
                'Other'
            )
        return X_transformed

###############################################################################
#                            Neo4j Queries
###############################################################################
def connect_to_neo4j(uri=None, username=None, password=None):
    # Update these credentials or read from environment
    HARDCODED_URI = "neo4j+s://2022b6a4.databases.neo4j.io"
    HARDCODED_USERNAME = "neo4j"
    HARDCODED_PASSWORD = "E_zTYSm4--WJKaaz4OTaaBiP_8Kng2ePCYWuMdRDJXU"
    driver = GraphDatabase.driver(
        HARDCODED_URI, 
        auth=(HARDCODED_USERNAME, HARDCODED_PASSWORD)
    )
    return driver

def _get_nodes(tx):
    query = """
    MATCH (n)
    RETURN n.tmdbId AS id, labels(n) AS labels, properties(n) AS properties
    """
    result = tx.run(query)
    nodes = []
    for record in result:
        node_data = record["properties"]
        node_data["id"] = record["id"]
        node_data["labels"] = record["labels"]
        nodes.append(node_data)
    return nodes

def get_nodes(driver):
    with driver.session() as session:
        return session.execute_read(_get_nodes)

def _get_relationships(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN r.tmdbId AS id, n.tmdbId AS start_id, m.tmdbId AS end_id, type(r) AS type, properties(r) AS properties
    """
    result = tx.run(query)
    relationships = []
    for record in result:
        rel_data = record["properties"]
        rel_data["id"] = record["id"]
        rel_data["start_id"] = record["start_id"]
        rel_data["end_id"] = record["end_id"]
        rel_data["type"] = record["type"]
        relationships.append(rel_data)
    return relationships

def get_relationships(driver):
    with driver.session() as session:
        return session.execute_read(_get_relationships)

def extract_data(driver):
    nodes = get_nodes(driver)
    relationships = get_relationships(driver)
    nodes_df = pd.DataFrame(nodes)
    relationships_df = pd.DataFrame(relationships)
    return nodes_df, relationships_df

###############################################################################
#                   Identify Missing Attributes
###############################################################################
def identify_missing_attributes(nodes_df):
    print("\nAvailable Attributes in nodes_df:")
    for attr in nodes_df.columns:
        print(f"  - {attr}")

    chosen_attribute = input("\nEnter the attribute you want to model/infer (e.g. 'age' or 'status'): ")
    if chosen_attribute not in nodes_df.columns:
        print(f"[WARNING] '{chosen_attribute}' not found in DataFrame.\n")
        return pd.DataFrame(), []

    missing_mask = (nodes_df[chosen_attribute].isna()) | (nodes_df[chosen_attribute] == "")
    missing_attributes_df = nodes_df[missing_mask].copy()
    key_attributes = [chosen_attribute]
    return missing_attributes_df, key_attributes

###############################################################################
#                 FIT the Preprocessing Pipeline (TRAINING)
###############################################################################
def fit_preprocessing_pipeline(nodes_df, target_attribute, verbose=True, max_pca_components=150):
    """
    1) Filter training rows (duplicates, missing target).
    2) Extract y_train before dropping the target column.
    3) Manually handle 'embedding' with PCA -> store pca_object.
    4) Force embedding_pca columns to float.
    5) Clean up categorical columns (turn any lists into joined strings).
    6) Fit ColumnTransformer pipeline and return.
    """
    training_df = nodes_df.dropna(subset=[target_attribute]).copy()
    
    # Remove duplicates
    if training_df.duplicated(subset=['tmdbId']).any():
        dup_count = training_df.duplicated(subset=['tmdbId']).sum()
        if verbose:
            print(f"[WARNING] Found {dup_count} duplicated 'tmdbId's in training data.")
        training_df = training_df.drop_duplicates(subset=['tmdbId'])

    # Convert target to float if possible
    is_numeric_target = False
    try:
        training_df[target_attribute] = training_df[target_attribute].astype(float)
        is_numeric_target = True
    except ValueError:
        pass

    # If the target is 'year' we could do out-of-range filtering (example code)
    # if target_attribute == 'year' and is_numeric_target:
    #     ...

    # Extract labels (y_train)
    if is_numeric_target:
        y_train = training_df[target_attribute].values.astype(float)
    else:
        y_train = training_df[target_attribute].values

    # Drop the target col + a few known non-informative
    drop_cols = ['id', 'labels', target_attribute]
    drop_cols = [c for c in drop_cols if c in training_df.columns]
    training_df = training_df.drop(columns=drop_cols, errors='ignore')

    # -----------------------------
    # Handle embedding with PCA
    # -----------------------------
    embedding_cols = [col for col in training_df.columns if col.lower() == 'embedding']
    pca_object = None
    if embedding_cols:
        emb_col = embedding_cols[0]
        
        # Convert string -> list if needed
        if training_df[emb_col].dtype == object:
            training_df[emb_col] = training_df[emb_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        expected_len = 1536
        # Check which rows have a valid embedding length
        mask_correct = training_df[emb_col].apply(
            lambda x: (isinstance(x, list) or isinstance(x, np.ndarray)) and len(x) == expected_len
        )

        # Replace invalid embeddings with zero vectors
        training_df.loc[~mask_correct, emb_col] = training_df.loc[~mask_correct, emb_col].apply(
            lambda x: np.zeros(expected_len, dtype=np.float32)
        )

        # Ensure entire column is float arrays
        training_df[emb_col] = training_df[emb_col].apply(
            lambda arr: np.array(arr, dtype=np.float32)
        )

        # Now stack
        embeddings = np.vstack(training_df[emb_col].values)  # shape: (n_samples, 1536) float32

        # Dynamically clamp PCA n_components
        n_samples, n_features = embeddings.shape
        max_valid_components = min(n_samples, n_features, max_pca_components)

        if max_valid_components < 2:
            # Not enough dimension or samples to do PCA
            print("[WARNING] Not enough data to perform PCA. Skipping embedding PCA.")
            # Optionally remove the embedding col altogether
            training_df.drop(columns=[emb_col], inplace=True)
        else:
            # Perform PCA
            pca_object = PCA(n_components=max_valid_components)
            reduced = pca_object.fit_transform(embeddings)

            # Create columns for the reduced embedding
            pca_cols = [f"{emb_col}_pca_{i}" for i in range(max_valid_components)]
            reduced_df = pd.DataFrame(reduced, columns=pca_cols)
            # Force numeric
            for c in reduced_df.columns:
                reduced_df[c] = reduced_df[c].astype(float)

            # Concat back to training_df
            training_df = pd.concat(
                [training_df.drop(columns=[emb_col]).reset_index(drop=True),
                 reduced_df.reset_index(drop=True)],
                axis=1
            )

    # Identify cat / num / bool columns
    cat_cols = training_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = training_df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = training_df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        cat_cols += bool_cols
        num_cols = [c for c in num_cols if c not in bool_cols]

    # Flatten any list-likes in categorical columns
    for col in cat_cols:
        training_df[col] = training_df[col].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
        # Force string
        training_df[col] = training_df[col].astype(str)

    # Exclude certain ID columns from cat
    exclude_id_cols = ['tmdbId', 'imdbId', 'name', 'userId']
    cat_cols = [c for c in cat_cols if c not in exclude_id_cols]

    # Drop cat cols that are entirely missing or "nan"
    for c in cat_cols:
        if training_df[c].replace("nan", np.nan).isna().all():
            training_df.drop(columns=[c], inplace=True)

    # Re-check cat_cols if we dropped some
    cat_cols = [c for c in cat_cols if c in training_df.columns]

    # Build pipelines
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('topk', TopKCategories(top_k=100)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    X_train = preprocessor.fit_transform(training_df)

    if verbose:
        print(f"[INFO] Fitted preprocessing pipeline. Final X_train shape: {X_train.shape}")

    return preprocessor, training_df, X_train, y_train, pca_object

###############################################################################
#                 TRANSFORM New Data (INFERENCE)
###############################################################################
def transform_data(df, preprocessor, pca_object, verbose=True, max_pca_components=150):
    df_trans = df.copy()

    # Drop 'id','labels'
    drop_cols = []
    if 'id' in df_trans.columns:
        drop_cols.append('id')
    if 'labels' in df_trans.columns:
        drop_cols.append('labels')
    df_trans.drop(columns=drop_cols, inplace=True, errors='ignore')

    # If there's an 'embedding' column, replicate the PCA
    embedding_cols = [c for c in df_trans.columns if c.lower() == 'embedding']
    if pca_object and embedding_cols:
        emb_col = embedding_cols[0]
        if df_trans[emb_col].dtype == object:
            df_trans[emb_col] = df_trans[emb_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        expected_len = 1536
        # Fix invalid embeddings
        mask_correct = df_trans[emb_col].apply(
            lambda x: (isinstance(x, list) or isinstance(x, np.ndarray)) and len(x) == expected_len
        )
        df_trans.loc[~mask_correct, emb_col] = df_trans.loc[~mask_correct, emb_col].apply(
            lambda x: np.zeros(expected_len, dtype=np.float32)
        )

        # Ensure arrays
        df_trans[emb_col] = df_trans[emb_col].apply(
            lambda arr: np.array(arr, dtype=np.float32)
        )

        embeddings = np.vstack(df_trans[emb_col].values)

        # Use the same n_components as pca_object
        reduced = pca_object.transform(embeddings)
        n_components = reduced.shape[1]
        pca_cols = [f"{emb_col}_pca_{i}" for i in range(n_components)]
        reduced_df = pd.DataFrame(reduced, columns=pca_cols)

        # Force numeric
        for c in reduced_df.columns:
            reduced_df[c] = reduced_df[c].astype(float)

        df_trans = pd.concat([
            df_trans.drop(columns=[emb_col]).reset_index(drop=True),
            reduced_df.reset_index(drop=True)
        ], axis=1)
    else:
        # If we have an embedding col but no pca_object, just drop it
        for col in embedding_cols:
            df_trans.drop(columns=[col], inplace=True, errors='ignore')

    # Convert cat columns to string, flatten any list-likes
    cat_cols = df_trans.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df_trans[col] = df_trans[col].apply(
            lambda x: ",".join(x) if isinstance(x, list) else x
        )
        df_trans[col] = df_trans[col].astype(str)

    X_new = preprocessor.transform(df_trans)
    if hasattr(X_new, 'toarray'):
        X_new = X_new.toarray()

    if verbose:
        print(f"[INFO] Transformed new data. Shape: {X_new.shape}")
    return X_new

###############################################################################
#                          Train AI Models (with hyperparam tuning)
###############################################################################
def train_ai_models(X, y, target_attribute, verbose=True):
    """
    - If numeric, we scale the target and use XGBRegressor with basic hyperparam tuning.
    - If categorical, we use LogisticRegression.
    """
    if np.issubdtype(y.dtype, np.floating):
        # Scale target
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).ravel()

        # Prepare train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_scaled,
            test_size=0.2,
            random_state=42
        )

        # Basic param distributions for XGBoost
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }

        xgb_model = XGBRegressor(random_state=42)

        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=42,
            verbose=1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        if verbose:
            print(f"[INFO] Best XGB Params: {search.best_params_}")

        # Final fit
        best_model.fit(X_train, y_train)

        # Evaluate MSE on validation
        y_pred_scaled = best_model.predict(X_val)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1,1)).ravel()

        mse = mean_squared_error(y_val_orig, y_pred)
        if verbose:
            print(f"Trained XGBRegressor for '{target_attribute}' with MSE: {mse:.4f}")

        model_info = {
            'model': best_model,
            'y_scaler': y_scaler
        }
        return model_info

    else:
        # Classification fallback
        logreg = LogisticRegression(max_iter=1000)
        X_train_, X_val_, y_train_, y_val_ = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        logreg.fit(X_train_, y_train_)
        preds = logreg.predict(X_val_)
        acc = accuracy_score(y_val_, preds)
        if verbose:
            print(f"Trained Logistic Regression for '{target_attribute}' with Accuracy: {acc:.4f}")
        return logreg

###############################################################################
#                      Inference on Missing Attributes
###############################################################################
def infer_missing_attributes(model_info, X_missing, is_numeric=True):
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
        y_scaler = model_info['y_scaler']
        y_pred_scaled = model.predict(X_missing)
        predictions = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    else:
        # Classification
        model = model_info
        predictions = model.predict(X_missing)
    return predictions

###############################################################################
#                          Generate Report
###############################################################################
def generate_report(inferred_values, target_attr):
    df = pd.DataFrame(inferred_values, columns=[f"inferred_{target_attr}"])
    filename = f"inferred_{target_attr}_report.csv"
    df.to_csv(filename, index=False)
    print(f"[INFO] Generated report at '{filename}'.")

###############################################################################
#                     Update Neo4j with Inferred Values
###############################################################################
def update_neo4j(driver, missing_attributes_df, inferred_values, target_attribute, batch_size=1000):
    node_ids = missing_attributes_df['tmdbId'].tolist()

    update_choice = input(
        f"\nDo you want to update the Neo4j database with the inferred '{target_attribute}' values? (yes/no): "
    ).strip().lower()
    if update_choice not in ['yes', 'y']:
        print("[INFO] User opted not to update the Neo4j database.")
        return

    with driver.session() as session:
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i+batch_size]
            batch_predictions = inferred_values[i:i+batch_size]
            session.write_transaction(
                _update_batch,
                batch_ids,
                batch_predictions,
                target_attribute
            )
    print(f"[INFO] Updated Neo4j database with inferred '{target_attribute}' in batches.")

def _update_batch(tx, batch_ids, batch_predictions, target):
    for node_id, prediction in zip(batch_ids, batch_predictions):
        if isinstance(prediction, float):
            query = f"""
            MATCH (n)
            WHERE n.tmdbId = $id
            SET n.{target} = $value
            """
            params = {'id': node_id, 'value': float(prediction)}
        else:
            query = f"""
            MATCH (n)
            WHERE n.tmdbId = $id
            SET n.{target} = $value
            """
            params = {'id': node_id, 'value': str(prediction)}
        tx.run(query, params)

###############################################################################
#                        Save Trained Models
###############################################################################
def save_trained_models(model_info, target_attribute):
    if isinstance(model_info, dict) and 'model' in model_info:
        joblib.dump(model_info['model'], f"trained_model_{target_attribute}.joblib")
        joblib.dump(model_info['y_scaler'], f"trained_model_{target_attribute}_scaler.joblib")
        print(f"[INFO] Saved model & scaler for '{target_attribute}'.")
    else:
        joblib.dump(model_info, f"trained_model_{target_attribute}.joblib")
        print(f"[INFO] Saved trained model for '{target_attribute}'.")

###############################################################################
#                           Main Workflow
###############################################################################
def main():
    print("=== Step 1: Connect to Neo4j Graph Database ===")
    driver = connect_to_neo4j()
    print("[INFO] Neo4j driver created.")

    print("\n=== Step 2: Extract Nodes and Relationships Data ===")
    nodes_df, relationships_df = extract_data(driver)
    print(f"[INFO] Retrieved {len(nodes_df)} nodes and {len(relationships_df)} relationships.")

    print("\n=== Step 3: Identify Missing Attributes ===")
    missing_attributes_df, key_attributes = identify_missing_attributes(nodes_df)
    if not key_attributes:
        print("[INFO] No valid target attributes selected. Exiting.")
        driver.close()
        return

    target_attr = key_attributes[0]
    print(f"[INFO] Chosen attribute: {target_attr}")
    print(f"[INFO] Found {len(missing_attributes_df)} nodes missing '{target_attr}'.")

    print("\n=== Step 4: Fit Preprocessing Pipeline ===")
    preprocessor, training_df, X_train, y_train, pca_object = fit_preprocessing_pipeline(
        nodes_df,
        target_attr,
        verbose=True,
        max_pca_components=150  # You may adjust this down if your dataset is very small
    )
    print(f"[INFO] Completed pipeline fitting. Training set shape: {X_train.shape}")
    print(f"[INFO] Training labels shape: {y_train.shape}")

    # Train model
    print("\n=== Step 5: Train AI Model ===")
    model_info = train_ai_models(X_train, y_train, target_attr, verbose=True)
    print("[INFO] Model trained.")

    # Inference
    print("\n=== Step 6: Inference ===")
    # For those missing the target, fill with a placeholder so shape matches
    placeholder_df = nodes_df.copy()
    placeholder_df.loc[missing_attributes_df.index, target_attr] = 2000
    X_missing_full = transform_data(
        placeholder_df,
        preprocessor,
        pca_object,
        verbose=True,
        max_pca_components=150
    )

    missing_indices = missing_attributes_df.index
    X_missing_subset = X_missing_full[missing_indices, :]
    is_numeric = np.issubdtype(y_train.dtype, np.floating)
    inferred_vals = infer_missing_attributes(model_info, X_missing_subset, is_numeric)
    print("[INFO] Inferred missing attributes.")

    # Generate report
    print("\n=== Step 7: Generate Report ===")
    generate_report(inferred_vals, target_attr)

    # Update Neo4j
    print("\n=== Step 8: Update Neo4j Database ===")
    update_neo4j(driver, missing_attributes_df, inferred_vals, target_attr)

    # Save models
    print("\n=== Step 9: Save Trained Model ===")
    save_trained_models(model_info, target_attr)

    print("\n=== Workflow Complete ===")
    driver.close()


if __name__ == "__main__":
    main()
