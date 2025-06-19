# similarity.py
"""
similarity.py
~~~~~~~~~~~~~
Near-duplicate-detection helpers for 911-call records *plus*
a light wrapper that also returns the primary-key of the best
matching incident so the caller can re-use it in Neo4j.

Changes – 2025-04-21
▪ compare_json_records unchanged (still computes basic scores, although we
  no longer use address/time in deciding “similarity”)
▪ find_best_match() → returns (scores, incident_pk | None)
▪ each flattened record persisted in data_pull.json must now include
  "incident_pk" (str) once it is known.
"""

from __future__ import annotations
import json, re, logging
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────  logging  ──────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

# ───────────────────────────  type aliases  ─────────────────────────
AddressData      = Tuple[str, Optional[str], Optional[datetime]]
SimilarityScores = Tuple[bool, float, bool, float, float, float, float]
JSONData         = Dict[str, Any]


# ══════════════════════════  low-level helpers  ══════════════════════
class CallDataExtractor:
    @staticmethod
    def extract_address_zip_time(d: JSONData) -> AddressData:
        meta = d.get("metadata", {})
        addr = meta.get("clean_address_EMS", "")
        m = re.search(r"\b\d{5}\b", addr)
        zip_code = m.group(0) if m else None

        start = meta.get("start", "")
        call_dt: Optional[datetime] = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M"):
            try:
                call_dt = datetime.strptime(start, fmt)
                break
            except ValueError:
                continue
        return addr, zip_code, call_dt


class SimilarityAnalyzer:
    @staticmethod
    def jaccard(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        s1, s2 = set(a.lower().split()), set(b.lower().split())
        return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

    @staticmethod
    def tfidf(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform([a, b])
        return cosine_similarity(mat[0:1], mat[1:2])[0][0]

    @staticmethod
    def within_1hr(t1: Optional[datetime], t2: Optional[datetime]) -> bool:
        if not (t1 and t2):
            return False
        return abs((t1 - t2).total_seconds()) <= 3600


# ══════════════════════════  public comparer  ════════════════════════
class CallRecordComparer:
    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.extractor = CallDataExtractor()
        self.sim       = SimilarityAnalyzer()
        # We match purely on “merged transcript+summary” text:
        self.thr = {
            "merged_j":  0.40,   # merged‐text Jaccard threshold
            "merged_tf": 0.50,   # merged‐text TF-IDF threshold
        }
        if thresholds:
            self.thr.update(thresholds)

    # ---------- pairwise vector of similarity scores -----------------
    def compare_json_records(self, d1: JSONData, d2: JSONData) -> SimilarityScores:
        """
        Returns (same_zip, addr_j, within_hr, tr_j, tr_tf, sm_j, sm_tf),
        but these sub‐scores are just informational; we use only merged metrics for matching.
        """
        addr1, zip1, t1 = self.extractor.extract_address_zip_time(d1)
        addr2, zip2, t2 = self.extractor.extract_address_zip_time(d2)

        same_zip  = bool(zip1 and zip2 and zip1 == zip2)
        addr_j    = self.sim.jaccard(addr1, addr2)
        within_hr = self.sim.within_1hr(t1, t2)

        tr1, tr2  = d1.get("transcript", ""), d2.get("transcript", "")
        sm1, sm2  = d1.get("summary",    ""), d2.get("summary",    "")

        tr_j  = self.sim.jaccard(tr1, tr2)
        tr_tf = self.sim.tfidf(tr1, tr2)
        sm_j  = self.sim.jaccard(sm1, sm2)
        sm_tf = self.sim.tfidf(sm1, sm2)

        return (same_zip, addr_j, within_hr, tr_j, tr_tf, sm_j, sm_tf)

    # ---------- hard decision ----------------------------------------
    def is_similar(self, d1: JSONData, d2: JSONData) -> bool:
        """
        Compare two records by concatenating transcript + summary,
        then checking Jaccard/TF-IDF against lowered thresholds.
        """
        text1 = f"{d1.get('transcript','')} {d1.get('summary','')}"
        text2 = f"{d2.get('transcript','')} {d2.get('summary','')}"

        m_j  = self.sim.jaccard(text1, text2)
        m_tf = self.sim.tfidf(text1, text2)

        return (m_tf > self.thr["merged_tf"]) or (m_j > self.thr["merged_j"])

    # ---------- convenience wrapper ----------------------------------
    def find_best_match(
        self, new_rec: JSONData, prev: List[JSONData]
    ) -> Tuple[Optional[SimilarityScores], Optional[str]]:
        """
        Return (SimilarityScores, incident_pk) for the best match in prev,
        else (None, None). We pick the prev-record with highest merged TF-IDF.
        """
        best: Tuple[SimilarityScores, str] | None = None

        for rec in prev:
            # 1) get sub-scores (not used for matching logic directly)
            scores = self.compare_json_records(new_rec, rec)

            # 2) check if “similar” by merged transcript+summary
            if not self.is_similar(new_rec, rec):
                continue

            # 3) retrieve stored PK (treat "" or missing as None)
            stored_pk = rec.get("incident_pk")
            if not stored_pk:
                stored_pk = None

            # 4) recompute merged TF-IDF for ordering the “best” match
            merged1 = f"{new_rec.get('transcript','')} {new_rec.get('summary','')}"
            merged2 = f"{rec.get('transcript','')} {rec.get('summary','')}"
            merged_tf = self.sim.tfidf(merged1, merged2)

            # 5) choose the record with highest merged TF-IDF
            if not best or merged_tf > best[0][4]:
                # here best[0][4] is effectively “best merged TF-IDF so far”
                # but we still store the full 7-tuple in best[0]
                best = (scores, stored_pk)

        return best if best else (None, None)


# ══════════════════════════  tiny JSON helper  ════════════════════════
class DataManager:
    @staticmethod
    def load(path: str) -> List[JSONData]:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error("Failed loading %s: %s", path, e)
            return []

    @staticmethod
    def save(data: List[JSONData], path: str):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed saving %s: %s", path, e)
