import json, os, traceback
from typing import Dict, List

import pandas as pd
from extraction_utils import extract_all_911_call_data


def process_csv_file(params: Dict) -> Dict:
    """
    Re‑implementation of your old /process_all_transcripts endpoint,
    but returned as a plain dict so the Flask route stays thin.
    """

    file_name     = params.get("file_name", "911_dataset3.csv")
    batch_size    = params.get("batch_size", 10)
    limit         = params.get("limit_records", False)
    max_records   = params.get("max_records", 50)

    if not os.path.exists(file_name):
        return {"status": "error",
                "message": f"{file_name} not found"}

    # --- load CSV --------------------------------------------------------
    df = None
    for enc in ("utf-8", "latin1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(file_name, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        return {"status": "error",
                "message": "Failed reading CSV with common encodings"}

    if "TEXT" not in df.columns:
        return {"status": "error",
                "message": "Column TEXT missing"}

    df = df[df["TEXT"].notna() & (df["TEXT"].str.strip() != "")]
    if limit:
        df = df.head(max_records)

    results, errors = [], []
    total = len(df)
    for idx, row in df.iterrows():
        try:
            cd = extract_all_911_call_data(row["TEXT"])
            cd["row_index"] = int(idx)
            cd["metadata"]  = {k: (str(v) if pd.notna(v) else "")
                               for k, v in row.items() if k != "TEXT"}
            results.append(cd)
        except Exception as e:
            traceback.print_exc()
            errors.append({"index": int(idx), "error": str(e)})

        if len(results) % batch_size == 0:
            print(f"Processed {len(results)}/{total}")

    return {"status": "success",
            "processed": len(results),
            "errors": len(errors),
            "results": results[:3],   # sample preview
            "error_details": errors}
