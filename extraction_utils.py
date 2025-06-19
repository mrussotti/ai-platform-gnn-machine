# extraction_utils.py
from __future__ import annotations
import os, re, json, pathlib, logging, traceback
from datetime import datetime
from typing import Dict, Any

from openai import OpenAI
import httpx

_log_fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
lvl      = logging.DEBUG if os.environ.get("LOG_LEVEL","").upper()=="DEBUG" else logging.INFO
logging.basicConfig(level=lvl, format=_log_fmt)
logger = logging.getLogger(__name__)

_CLIENT = OpenAI(
    api_key="sk-d0a34cbfde64466eb6e7c7b07f12e2c9",
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(timeout=60)
)

_SYSTEM_PROMPT = (
    "You are an expert emergency call analyzer. "
    "Return ONLY valid JSON matching the schema exactly, with no extra fields."
)


def extract_all_911_call_data(transcript: str) -> Dict[str, Any]:
    """
    Calls DeepSeek and returns exactly this structure:

    {
      "call": {
        "summary":    str,
        "timestamp":  str,
        "transcript": str,
        "severity":   str,
        "hazards":    str
      },
      "location": {
        "address":  str,
        "type":     str,
        "features": str,
        "time":     str
      },
      "descriptions": [
        {"transcript": str}
      ],
      "persons": [
        {
          "name":         str,
          "phone":        str,
          "role":         str,
          "relationship": str,
          "conditions":   str,
          "age":          str,
          "sex":          str
        }
      ]
    }
    """
    # define schema to pass to model
    schema = {
        "call": {
            "summary":    "string",
            "timestamp":  "string",
            "transcript": "string",
            "severity":   "string",
            "hazards":    "string",
        },
        "location": {
            "address":  "string",
            "type":     "string",
            "features": "string",
            "time":     "string",
        },
        "descriptions": [
            {"transcript": "string"}
        ],
        "persons": [
            {
                "name":         "string",
                "phone":        "string",
                "role":         "string",
                "relationship": "string",
                "conditions":   "string",
                "age":          "string",
                "sex":          "string"
            }
        ]
    }

    user_prompt = (
        "Extract the JSON exactly as per this schema (no extra keys): "
        + json.dumps(schema)
        + f"\n\nTRANSCRIPT:\n{transcript}\n"
    )

    try:
        resp = _CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt}
            ],
            temperature=0,
            stream=False
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        raw = json.loads(content)
    except Exception:
        logger.exception("DeepSeek extraction failed, using fallback")
        raw = _fallback_response(transcript)

    # normalize call
    c = raw.get("call", {}) or {}
    call = {
        "summary":    c.get("summary", "911 emergency call"),
        "timestamp":  c.get("timestamp", ""),
        "transcript": c.get("transcript", transcript),
        "severity":   c.get("severity", ""),
        "hazards":    c.get("hazards", "")
    }

    # normalize location
    l = raw.get("location", {}) or {}
    location = {
        "address":  l.get("address", ""),
        "type":     l.get("type", ""),
        "features": l.get("features", ""),
        "time":     l.get("time", "")
    }

    # normalize descriptions
    descs = raw.get("descriptions")
    if not isinstance(descs, list) or not descs:
        descriptions = [{"transcript": ""}]
    else:
        descriptions = [{"transcript": d.get("transcript", "")} for d in descs if isinstance(d, dict)]
        if not descriptions:
            descriptions = [{"transcript": ""}]

    # normalize persons
    persons = []
    for p in raw.get("persons", []):
        if isinstance(p, dict):
            persons.append({
                "name":         p.get("name", ""),
                "phone":        p.get("phone", ""),
                "role":         p.get("role", ""),
                "relationship": p.get("relationship", ""),
                "conditions":   p.get("conditions", ""),
                "age":          str(p.get("age", "")),
                "sex":          p.get("sex", "")
            })

    return {
        "call":         call,
        "location":     location,
        "descriptions": descriptions,
        "persons":      persons
    }


def _fallback_response(transcript: str) -> Dict[str, Any]:
    """Minimal stub matching new schema when DeepSeek fails."""
    return {
        "call": {
            "summary":    "Failed to extract",
            "timestamp":  "",
            "transcript": transcript,
            "severity":   "",
            "hazards":    ""
        },
        "location":     {"address": "", "type": "", "features": "", "time": ""},
        "descriptions": [{"transcript": ""}],
        "persons":      []
    }

def preprocess_transcript(t: str) -> str:
    t = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", t)
    return re.sub(r"\s+", " ", t).strip()

