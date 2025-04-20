"""
similarity.py
~~~~~~~~~~~~~
Tools for detecting near‑duplicate 911‑call records.

Four main classes
-----------------
CallDataExtractor   – pull address / ZIP / call‑time out of record["metadata"].
SimilarityAnalyzer  – Jaccard / TF‑IDF / time‑window helpers.
CallRecordComparer  – bundle of similarity rules + thresholds.
DataManager         – tiny JSON load/save helper used by the Flask service.
"""
from __future__ import annotations

import json, re, logging
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
AddressData      = Tuple[str, Optional[str], Optional[datetime]]
SimilarityScores = Tuple[bool, float, bool, float, float, float, float]
JSONData         = Dict[str, Any]

# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class CallDataExtractor:
    """Extract address, ZIP and datetime from metadata."""
    @staticmethod
    def extract_address_zip_time(json_data: JSONData) -> AddressData:
        meta = json_data.get("metadata", {})
        address = meta.get("clean_address_EMS", "")

        m = re.search(r"\b\d{5}\b", address)
        zip_code = m.group(0) if m else None

        start_time_str = meta.get("start", "")
        call_time: Optional[datetime] = None
        if start_time_str:
            for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%y %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    call_time = datetime.strptime(start_time_str, fmt)
                    break
                except ValueError:
                    continue

        return address, zip_code, call_time

# ---------------------------------------------------------------------------
# Low‑level similarity helpers
# ---------------------------------------------------------------------------
class SimilarityAnalyzer:
    @staticmethod
    def is_time_within_one_hour(t1: Optional[datetime], t2: Optional[datetime]) -> bool:
        if t1 is None or t2 is None:
            return False
        return abs((t1 - t2).total_seconds()) <= 3600

    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        a, b = set(s1.lower().split()), set(s2.lower().split())
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    @staticmethod
    def tfidf_similarity(txt1: str, txt2: str) -> float:
        if not txt1 or not txt2:
            return 0.0
        try:
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform([txt1, txt2])
            return cosine_similarity(mat[0:1], mat[1:2])[0][0]
        except Exception as e:
            logger.error("TF‑IDF similarity error: %s", e)
            return 0.0

# ---------------------------------------------------------------------------
# Record‑level comparer
# ---------------------------------------------------------------------------
class CallRecordComparer:
    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.extractor = CallDataExtractor()
        self.analyzer  = SimilarityAnalyzer()

        self.thresholds = {
            "address_jaccard"   : 0.50,
            "transcript_jaccard": 0.50,
            "transcript_tfidf"  : 0.80,
            "summary_jaccard"   : 0.50,
            "summary_tfidf"     : 0.60,
        }
        if thresholds:
            self.thresholds.update(thresholds)

    # ---- pairwise comparison ---------------------------------------------
    def compare_json_records(self, d1: JSONData, d2: JSONData) -> SimilarityScores:
        addr1, zip1, t1 = self.extractor.extract_address_zip_time(d1)
        addr2, zip2, t2 = self.extractor.extract_address_zip_time(d2)

        same_zip           = bool(zip1 and zip2 and zip1 == zip2)
        address_jaccard    = self.analyzer.jaccard_similarity(addr1, addr2)
        time_within_hour   = self.analyzer.is_time_within_one_hour(t1, t2)

        tr1, tr2 = d1.get("transcript", ""), d2.get("transcript", "")
        sm1, sm2 = d1.get("summary",    ""), d2.get("summary",    "")

        transcript_jaccard = self.analyzer.jaccard_similarity(tr1, tr2)
        transcript_tfidf   = self.analyzer.tfidf_similarity(tr1, tr2)
        summary_jaccard    = self.analyzer.jaccard_similarity(sm1, sm2)
        summary_tfidf      = self.analyzer.tfidf_similarity(sm1, sm2)

        return (same_zip, address_jaccard, time_within_hour,
                transcript_jaccard, transcript_tfidf,
                summary_jaccard, summary_tfidf)

    # ---- final decision ---------------------------------------------------
    def is_similar(self, scores: SimilarityScores) -> bool:
        same_zip, addr_j, within_hr, tr_j, tr_tf, sm_j, sm_tf = scores
        return ((same_zip and within_hr)
                and addr_j > self.thresholds["address_jaccard"]
                and (tr_j > self.thresholds["transcript_jaccard"]
                     or tr_tf > self.thresholds["transcript_tfidf"])
                and (sm_j > self.thresholds["summary_jaccard"]
                     or sm_tf > self.thresholds["summary_tfidf"]))

# ---------------------------------------------------------------------------
# Data I/O helper
# ---------------------------------------------------------------------------
class DataManager:
    @staticmethod
    def load_data_pull(path: str) -> List[JSONData]:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return json.loads(data) if isinstance(data, str) else data
        except FileNotFoundError:
            logger.info("No data‑pull file yet: %s", path)
            return []
        except Exception as e:
            logger.error("Failed loading %s: %s", path, e)
            return []

    @staticmethod
    def save_data(data: Union[List, Dict], path: str, indent: int = 4) -> bool:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception as e:
            logger.error("Failed saving %s: %s", path, e)
            return False
