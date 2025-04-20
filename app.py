"""
Main Flask entry‑point.
Only wiring + response shaping; all heavy logic lives in helpers.
"""
from __future__ import annotations

import os, base64, json, logging, time
from pathlib import Path
from typing import Dict, Any, List

from flask      import Flask, jsonify, request
from flask_cors import CORS

# ── local modules ──────────────────────────────────────────────────────
from neo4j_utils      import connect_to_neo4j, extract_training_data, save_911_call_to_neo4j
from model_utils      import (preprocess_training_data, train_and_evaluate_encodings,
                              train_and_save_model, load_count_model)
from extraction_utils import (extract_all_911_call_data, preprocess_transcript,
                              analyze_transcript_quality)
from similarity       import CallRecordComparer, DataManager
from batch_utils      import process_csv_file

# ─────────────────────────  Flask & logging  ──────────────────────────
app = Flask(__name__)
CORS(app)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # may be overridden in __main__

# ──────────────────────────  Globals  ─────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_PULL      = BASE_DIR / "data_pull.json"
comparer: CallRecordComparer = CallRecordComparer()
data_mgr: DataManager        = DataManager()
# =====================================================================


# ==========================  MODEL TRAINING  ==========================
@app.post("/train_model")
def train_model():
    driver = connect_to_neo4j()
    df, le = preprocess_training_data(extract_training_data(driver))
    res    = train_and_evaluate_encodings(df, le)
    driver.close()
    return jsonify(res)


@app.post("/train_and_save")
def train_and_save():
    driver = connect_to_neo4j()
    df, le = preprocess_training_data(extract_training_data(driver))
    msg    = train_and_save_model(df, le)
    driver.close()
    return jsonify({"message": msg})


# ===========================  QUICK PREDICT  ==========================
@app.post("/predict")
def predict():
    t = request.json.get("transcript", "").strip()
    if not t:
        return jsonify({"error": "No transcript"}), 400

    clf, vec, le = load_count_model()
    cleaned      = preprocess_transcript(t).lower()
    lbl          = clf.predict(vec.transform([cleaned]))[0]
    nature       = le.inverse_transform([lbl])[0]
    return jsonify({"predicted_nature": nature})


# =====================  SINGLE CALL + SIMILARITY  =====================
@app.post("/process_911_call")
def process_911_call():
    # ---------- 1. decode -------------------------------------------------
    raw_b64 = request.json.get("transcript_b64", "")
    try:
        transcript = base64.b64decode(raw_b64).decode("utf-8", errors="replace").strip()
    except Exception:
        return jsonify({"error": "Bad base‑64 transcript"}), 400

    if not transcript:
        return jsonify({"error": "No transcript"}), 400

    log.info("⇢ /process_911_call – len=%d", len(transcript))

    # ---------- 2. LLM extraction + QA -----------------------------------
    quality   = analyze_transcript_quality(preprocess_transcript(transcript))
    call_data = extract_all_911_call_data(transcript)

    # ---------- 3. Flatten record for similarity engine ------------------
    def _safe(d: Dict[str, Any], *keys: str, default: str="") -> str:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        return cur or default

    flat_record: Dict[str, Any] = {
        "transcript": call_data.get("incident", {}).get("transcript", transcript),
        "summary"   : _safe(call_data, "emergency_details", "incident", "description"),
        "metadata"  : {
            "clean_address_EMS": _safe(call_data, "emergency_details", "address", "corrected"),
            "start"            : call_data.get("incident", {}).get("timestamp", "")
        }
    }

    # ---------- 4. Similarity pass ---------------------------------------
    prev_records: List[Dict] = data_mgr.load_data_pull(str(DATA_PULL))
    similar: List[Dict]      = []

    for prev in prev_records:
        scores = comparer.compare_json_records(flat_record, prev)
        # keep anything that has decent address or transcript similarity
        keep = (scores[4] > 0.50) or (scores[1] > 0.30)
        if keep:
            similar.append({
                "scores": {
                    "same_zip"          : bool(scores[0]),
                    "address_jaccard"   : round(scores[1], 2),
                    "within_1hr"        : bool(scores[2]),
                    "transcript_jaccard": round(scores[3], 2),
                    "transcript_tfidf"  : round(scores[4], 2),
                    "summary_jaccard"   : round(scores[5], 2),
                    "summary_tfidf"     : round(scores[6], 2)
                },
                "match_summary" : prev.get("summary", ""),
                "match_address" : _safe(prev, "metadata", "clean_address_EMS")
            })

    # append the new record and persist the pull
    prev_records.append(flat_record)
    data_mgr.save_data(prev_records, str(DATA_PULL))

    # ---------- 5. response ---------------------------------------------
    return jsonify({
        "status"         : "success",
        "quality"        : quality,
        "call_data"      : call_data,
        "similar_matches": similar,
        "warnings"       : []
    })


# ===========================  BULK CSV  ===============================
@app.post("/process_all_transcripts")
def process_all_transcripts():
    return jsonify(process_csv_file(request.json or {}))


# ===========================  NEO4J TEST ===============================
@app.get("/test_neo4j_connection")
def test_neo4j_connection():
    try:
        driver = connect_to_neo4j()
        with driver.session() as ses:
            msg = ses.run("RETURN 'Connection OK' AS m").single()["m"]
        driver.close()
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================  MAIN  ==================================
if __name__ == "__main__":
    # honour LOG_LEVEL env‑var (DEBUG / INFO / WARNING / ERROR)
    lvl_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    lvl      = logging.getLevelName(lvl_name)
    log.setLevel(lvl)

    # quick sanity on persistence file
    if not DATA_PULL.exists():
        DATA_PULL.write_text("[]", encoding="utf-8")

    app.run(debug=True)
