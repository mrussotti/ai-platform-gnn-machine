# app.py

from __future__ import annotations
import os
import base64
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

from extraction_utils import extract_all_911_call_data
from similarity import CallRecordComparer, DataManager
from neo4j_utils import connect_to_neo4j, save_911_call_to_neo4j

app = Flask(__name__)
CORS(app)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

BASE_DIR  = Path(__file__).resolve().parent
PULL_FILE = BASE_DIR / "data_pull.json"

cmpr  = CallRecordComparer()
store = DataManager()


def _process_single_call(transcript: str) -> Dict[str, Any]:
    """
    Core logic extracted from /process_911_call, with current-time stamping.
    """
    try:
        # 1) Extract raw call data
        call_data = extract_all_911_call_data(transcript)

        # 1a) Override any timestamps with current time
        now_iso = datetime.utcnow().isoformat()
        call_data["call"]["timestamp"] = now_iso
        if "location" in call_data and isinstance(call_data["location"], dict):
            call_data["location"]["time"] = now_iso

        # 2) Build flat record for dedupe
        desc = call_data["descriptions"][0]
        flat = {
            "transcript": desc["transcript"],
            "summary":    call_data["call"].get("summary", "")
        }

        # 3) Similarity match
        prev = store.load(str(PULL_FILE))
        best_scores, best_pk = cmpr.find_best_match(flat, prev)
        call_data["call"]["incident_pk"] = best_pk

        # 4) Persist to Neo4j
        neo = None
        real_pk = None
        call_status = "success"
        try:
            driver = connect_to_neo4j()
            neo    = save_911_call_to_neo4j(call_data, driver)
            real_pk = neo.get("incident_pk")
            if real_pk is None:
                raise RuntimeError("Failed to obtain incident_pk from Neo4j save.")
        except Exception as e:
            call_status = "error"
            neo = {"status": "error", "message": str(e)}
            log.error("Neo4j error: %s", e)
            traceback.print_exc()
        finally:
            try:
                driver.close()
            except Exception:
                pass

        # 5) Append to data_pull.json
        flat_record = {**flat, "incident_pk": real_pk}
        prev.append(flat_record)
        store.save(prev, str(PULL_FILE))

        # 6) Build response
        result: Dict[str, Any] = {
            "status":          call_status,
            "similarity_test": {"scores": best_scores, "matched_incident_pk": best_pk},
            "neo4j_nodes":     neo,
            "call_data":       call_data,
            "flat_record":     flat_record,
        }
        if call_status == "error":
            result["error_message"] = neo.get("message")
        return result

    except Exception as e:
        # Catch-all for unexpected pipeline errors
        error_msg = str(e)
        log.error("Error processing single call: %s", error_msg)
        traceback.print_exc()
        return {
            "status":        "error",
            "error_message": error_msg
        }


@app.post("/process_911_call")
def process_911_call():
    raw_b64 = request.json.get("transcript_b64", "")
    try:
        transcript = (
            base64.b64decode(raw_b64)
                  .decode("utf-8", errors="replace")
                  .strip()
        )
    except Exception:
        return jsonify({"error": "bad base-64"}), 400

    if not transcript:
        return jsonify({"error": "empty transcript"}), 400

    result = _process_single_call(transcript)
    return jsonify(result)


@app.post("/process_all_transcripts")
def process_all_transcripts():
    """
    Accepts a JSON array of rows, each with a 'TEXT' field for the transcript.
    Delegates each row to the single-call logic, returns list of results.
    """
    rows: List[Dict[str, Any]] = request.get_json(force=True) or []
    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        transcript = row.get("TEXT", "")
        if not transcript:
            results.append({
                "row_index":     idx,
                "status":        "error",
                "error_message": "empty transcript"
            })
            continue

        try:
            single_result = _process_single_call(transcript)
            single_result["row_index"] = idx
            results.append(single_result)
        except Exception as e:
            log.error("Unexpected error on row %d: %s", idx, e)
            traceback.print_exc()
            results.append({
                "row_index":     idx,
                "status":        "error",
                "error_message": str(e)
            })

    return jsonify(results)


if __name__ == "__main__":
    # ensure data_pull.json exists
    if not PULL_FILE.exists():
        PULL_FILE.write_text("[]", encoding="utf-8")
    app.run(debug=True)
