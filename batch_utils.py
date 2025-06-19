# batch_utils.py
import os, traceback
from typing import Dict

import pandas as pd
from extraction_utils import extract_all_911_call_data
from neo4j_utils      import connect_to_neo4j, save_911_call_to_neo4j

def process_csv_file(params: Dict) -> Dict:
    file_name   = params.get("file_name", "911_dataset3.csv")
    batch_size  = params.get("batch_size", 10)
    limit       = params.get("limit_records", False)
    max_records = params.get("max_records", 50)
    upload      = params.get("save_to_neo4j", False)

    if not os.path.exists(file_name):
        return {"status": "error", "message": f"{file_name} not found"}

    df = None
    for enc in ("utf-8", "latin1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(file_name, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        return {"status": "error", "message": "Failed reading CSV"}

    if "TEXT" not in df.columns:
        return {"status": "error", "message": "Column TEXT missing"}

    df = df[df["TEXT"].notna() & (df["TEXT"].str.strip() != "")]
    if limit:
        df = df.head(max_records)

    results, errors = [], []
    driver = connect_to_neo4j() if upload else None

    for idx, row in df.iterrows():
        try:
            transcript = row["TEXT"]
            # pass the entire row (as dict) into extraction
            metadata = {k: row[k] for k in row.index if k != "TEXT"}
            cd = extract_all_911_call_data(transcript, metadata)
            cd["row_index"] = int(idx)

            if upload and driver:
                try:
                    save_911_call_to_neo4j(cd, driver)
                except Exception as ne:
                    errors.append({"row": idx, "neo4j_error": str(ne)})

            results.append(cd)

        except Exception as e:
            traceback.print_exc()
            errors.append({"row": idx, "extraction_error": str(e)})

        if len(results) % batch_size == 0:
            print(f"Processed {len(results)}/{len(df)}")

    if driver:
        driver.close()

    return {
        "status":        "success",
        "processed":     len(results),
        "errors":        len(errors),
        "results":       results[:batch_size],
        "error_details": errors,
    }
