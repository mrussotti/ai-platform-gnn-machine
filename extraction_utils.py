"""
extraction_utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All 911‑call extraction helpers.

Changes on 2025‑04‑19
---------------------
* Properly read LOG_LEVEL from os.environ
* Always return a dict from `extract_all_911_call_data`
* Extra logging + dump bad LLM replies to disk for inspection
"""
from __future__ import annotations

import os
import re
import json
import pathlib
import logging
import traceback
from datetime import datetime
from typing import Dict, Any

from openai import OpenAI
import httpx                                  # only used for clearer type hints

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
_log_fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
lvl      = logging.DEBUG if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO
logging.basicConfig(level=lvl, format=_log_fmt)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  DeepSeek client
# ------------------------------------------------------------------ #
_CLIENT = OpenAI(
    api_key="sk-d0a34cbfde64466eb6e7c7b07f12e2c9",
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(timeout=60)
)

_SYSTEM_PROMPT = (
    "You are an expert emergency call analyzer. "
    "Return ONLY valid JSON that matches the schema provided. "
    "If you are unsure, do your best but never return free‑text outside JSON."
)


# ------------------------------------------------------------------ #
#  Public helpers
# ------------------------------------------------------------------ #
def extract_all_911_call_data(transcript: str) -> Dict[str, Any]:
    """
    Call DeepSeek, attempt to parse a JSON reply, and always return
    a *dict* – never a raw string.
    """
    user_prompt = f"Extract all structured information from this 911 transcript:\n\n{transcript}\n"
    logger.info("DeepSeek request (%d chars)", len(user_prompt))

    try:
        res = _CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user",   "content": user_prompt}],
            temperature=0,
            stream=False
        )
        content = res.choices[0].message.content.strip()
        # strip Markdown code‑fence if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        data = json.loads(content)

    # ---------- JSON error ------------------------------------------------
    except json.JSONDecodeError as e:
        dump_path = pathlib.Path("last_deepseek_raw.txt").absolute()
        dump_path.write_text(content, encoding="utf‑8", errors="replace")
        logger.error(
            "JSON decode error (%s) – raw reply saved to %s",
            e, dump_path
        )
        data = fallback_response(transcript)    # graceful fallback

    # ---------- Any other error -------------------------------------------
    except Exception:
        logger.error("DeepSeek extraction failed – falling back")
        traceback.print_exc()
        data = fallback_response(transcript)

    # ---------- Sanity pads (unchanged logic) ------------------------------
    data.setdefault("calls",     [{"summary": "911 emergency call", "timestamp": ""}])
    data.setdefault("location",  {"address": "", "type": "", "features": "", "time": ""})
    data.setdefault("persons",   [])
    data.setdefault("incident",  {})            # ensure key exists
    data["incident"].setdefault("transcript", transcript)

    return data


def fallback_response(transcript: str) -> Dict[str, Any]:
    """Return a minimal, well‑formed dict when extraction fails."""
    return {
        "incident": {
            "summary":    "Failed to extract",
            "timestamp":  "",
            "transcript": transcript,
            "nature":     "",
            "severity":   "",
            "hazards":    ""
        },
        "calls": [{
            "summary":   "911 emergency call",
            "timestamp": ""
        }],
        "persons":  [],
        "location": {
            "address":  "",
            "type":     "",
            "features": "",
            "time":     ""
        }
    }


# ------------------------------------------------------------------ #
#  Transcript QA helpers (unchanged)
# ------------------------------------------------------------------ #
def preprocess_transcript(t: str) -> str:
    t = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", t)
    return re.sub(r"\s+", " ", t).strip()


def analyze_transcript_quality(t: str) -> Dict[str, Any]:
    words = t.split()
    kw    = {"emergency", "help", "911", "accident", "injured", "medical",
             "fire", "police", "ambulance", "bleeding", "unconscious"}
    found = [w for w in kw if w in t.lower()]

    warn = []
    if len(words) < 20:
        warn.append("Transcript very short")
    if t.count("\n") > len(words) / 10:
        warn.append("Unusual formatting")
    if not found:
        warn.append("No emergency keywords")

    return {
        "word_count":               len(words),
        "warnings":                 warn,
        "is_suitable":              len(warn) == 0,
        "emergency_keywords_found": found
    }
