# neo4j_utils.py
from __future__ import annotations

import uuid
from typing import Dict, Any, Optional

from neo4j import GraphDatabase, Transaction

_URI      = ""
_USERNAME = ""
_PASSWORD = ""


def connect_to_neo4j():
    drv = GraphDatabase.driver(_URI, auth=(_USERNAME, _PASSWORD))
    with drv:
        drv.verify_connectivity()
    return drv


def _incident_by_pk(tx: Transaction, pk: str) -> Optional[int]:
    rec = tx.run("MATCH (i:Incident {pk:$pk}) RETURN id(i) AS id", pk=pk).single()
    return rec["id"] if rec else None


def get_or_create_incident(tx: Transaction,
                           incident: Dict[str, Any],
                           location: Dict[str, Any]) -> str:
    """
    Either find an existing Incident by its pk or create a new one.
    Always RETURN the Incident’s `pk` string.
    """
    # 1) If caller provided incident["pk"], see if it already exists:
    if pk := incident.get("pk"):
        rec = tx.run(
            "MATCH (i:Incident {pk:$pk}) RETURN i.pk AS pk",
            pk=pk
        ).single()
        if rec:
            return rec["pk"]

    # 2) Heuristic match on address + ±1h (if address & timestamp are non-empty):
    addr = (location.get("address") or "").strip()
    ts   = incident.get("timestamp", "").strip()
    if addr and ts:
        rec = tx.run(
            """
            MATCH (i:Incident)
            WHERE i.address = $addr
              AND i.timestamp <> '' AND $ts <> ''
              AND abs(
                    datetime(i.timestamp).epochSeconds -
                    datetime($ts).epochSeconds
                  ) < 3600
            RETURN i.pk AS pk
            """,
            addr=addr, ts=ts
        ).single()
        if rec:
            return rec["pk"]

    # 3) Otherwise, CREATE a brand-new Incident with a new UUID pk:
    new_pk = incident.get("pk") or str(uuid.uuid4())
    tx.run(
        """
        CREATE (i:Incident {
            pk: $pk,
            address:   $addr,
            timestamp: $ts,
            hazards:   $haz,
            severity:  $sev
        })
        """,
        pk=new_pk,
        addr=addr,
        ts=ts,
        haz=incident.get("hazards", ""),
        sev=incident.get("severity", "")
    )
    return new_pk


def save_911_call_to_neo4j(call_data: dict, driver):
    """
    1) Get-or-create the Incident (returns its `pk` string).
    2) Create a CallDetail node, link it to that Incident.
    3) Create a Location node and link to CallDetail.
    4) Create any Description nodes and link them to CallDetail.
    5) Create any Person nodes and link them to CallDetail.
    Returns: { "incident_pk": <the pk>, "call_detail": <internalID>, ... }
    """
    with driver.session() as ses:
        def _tx(tx: Transaction, data: dict):
            # A) Build the properties for Incident using call_data["call"] & call_data["location"]
            c = data["call"]
            incident_props = {
                # If you already have best_pk (maybe from similarity?), use it; else new one
                "pk":       c.get("incident_pk") or None,
                "hazards":  c.get("hazards", ""),
                "severity": c.get("severity", ""),
                "timestamp": c.get("timestamp", ""),
            }
            # location also gives us address/time for heuristic matching
            location_props = data["location"]

            # 1) Get-or-create Incident node, return its pk string
            incident_pk = get_or_create_incident(tx, incident_props, location_props)

            # 2) Create CallDetail node
            cd_props = {
                "pk":         str(uuid.uuid4()),
                "summary":    c.get("summary", ""),
                "timestamp":  c.get("timestamp", ""),
                "severity":   c.get("severity", ""),
                "hazards":    c.get("hazards", ""),
                "transcript": c.get("transcript", "")
            }
            cd_id = tx.run(
                "CREATE (c:CallDetail $p) RETURN id(c) AS id",
                p=cd_props
            ).single()["id"]

            # 3) Link CallDetail -> Incident
            tx.run(
                "MATCH (i:Incident {pk:$ipk}) "
                "MATCH (c:CallDetail) WHERE id(c)=$cid "
                "CREATE (c)-[:ABOUT]->(i)",
                ipk=incident_pk, cid=cd_id
            )

            # 4) Create Location and link to CallDetail
            l = data["location"]
            loc_props = {
                "address":  l.get("address", ""),
                "type":     l.get("type", ""),
                "features": l.get("features", ""),
                "time":     l.get("time", "")
            }
            loc_id = tx.run(
                "CREATE (l:Location $p) RETURN id(l) AS id",
                p=loc_props
            ).single()["id"]
            tx.run(
                "MATCH (l:Location) WHERE id(l)=$lid "
                "MATCH (c:CallDetail) WHERE id(c)=$cid "
                "CREATE (l)-[:AT]->(c)",
                lid=loc_id, cid=cd_id
            )

            # 5) Create each Description node and link to CallDetail
            desc_ids = []
            for d in data.get("descriptions", []):
                did = tx.run(
                    "CREATE (d:Description $p) RETURN id(d) AS id",
                    p={"transcript": d.get("transcript", "")}
                ).single()["id"]
                desc_ids.append(did)
                tx.run(
                    "MATCH (d:Description) WHERE id(d)=$did "
                    "MATCH (c:CallDetail) WHERE id(c)=$cid "
                    "CREATE (d)-[:DESCRIBES]->(c)",
                    did=did, cid=cd_id
                )

            # 6) Create each Person node and link to CallDetail
            person_ids = []
            for p in data.get("persons", []):
                pid = tx.run(
                    "CREATE (pr:Person $p) RETURN id(pr) AS id",
                    p={
                        "name":         p.get("name", ""),
                        "phone":        p.get("phone", ""),
                        "role":         p.get("role", ""),
                        "relationship": p.get("relationship", ""),
                        "conditions":   p.get("conditions", ""),
                        "age":          p.get("age", ""),
                        "sex":          p.get("sex", "")
                    }
                ).single()["id"]
                person_ids.append(pid)
                tx.run(
                    "MATCH (pr:Person) WHERE id(pr)=$pid "
                    "MATCH (c:CallDetail) WHERE id(c)=$cid "
                    "CREATE (pr)-[:INVOLVES]->(c)",
                    pid=pid, cid=cd_id
                )

            # Return the Incident’s pk and any node‐IDs you want
            return {
                "incident_pk":  incident_pk,
                "call_detail":  cd_id,
                "location":     loc_id,
                "descriptions": desc_ids,
                "persons":      person_ids
            }

        return ses.execute_write(_tx, call_data)
