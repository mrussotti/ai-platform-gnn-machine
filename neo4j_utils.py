import pandas as pd
from neo4j import GraphDatabase

# --------------------------------------------------------------------- #
#  Connection helpers
# --------------------------------------------------------------------- #

_URI      = "neo4j+ssc://2b9a3029.databases.neo4j.io"
_USERNAME = "neo4j"
_PASSWORD = "4GGm6B1aeQdjWyFxs7MpVCtqc8xZU0aueO_kqeAwXto"


def connect_to_neo4j():
    """Return a verified Neo4j driver instance."""
    driver = GraphDatabase.driver(_URI, auth=(_USERNAME, _PASSWORD))
    with driver:
        driver.verify_connectivity()
    return driver


# --------------------------------------------------------------------- #
#  Training‑data pull
# --------------------------------------------------------------------- #

def _get_incident_transcripts(tx):
    q = """
        MATCH (i:Incident)-[:CONTAINS]->(t:Transcript)
        RETURN i.nature AS nature, t.TEXT AS transcript
    """
    return [rec.data() for rec in tx.run(q)]


def extract_training_data(driver) -> pd.DataFrame:
    with driver.session() as ses:
        rows = ses.execute_read(_get_incident_transcripts)
    df = pd.DataFrame(rows).dropna(subset=["nature", "transcript"])
    return df


# --------------------------------------------------------------------- #
#  Graph persistence
# --------------------------------------------------------------------- #
def save_911_call_to_neo4j(call_data: dict, driver) -> dict:
    """Identical logic, just lifted out of the old script."""
    try:
        with driver.session() as ses:

            def _tx(tx, d):
                # ---------- Incident ----------
                inc_res = tx.run(
                    """
                    CREATE (i:Incident {
                        summary:$summary, timestamp:$timestamp,
                        nature:$nature, severity:$severity,
                        hazards:$hazards, transcript:$transcript
                    })
                    RETURN id(i) AS inc_id
                    """,
                    **d["incident"]
                ).single()
                inc_id = inc_res["inc_id"]

                # ---------- Calls -------------
                call_ids = []
                for c in d["calls"]:
                    cid = tx.run(
                        "CREATE (c:Call {summary:$summary,timestamp:$timestamp}) "
                        "RETURN id(c) AS cid", **c
                    ).single()["cid"]
                    call_ids.append(cid)
                    tx.run("MATCH (c),(i) WHERE id(c)=$cid AND id(i)=$iid "
                           "CREATE (c)-[:ABOUT]->(i)", cid=cid, iid=inc_id)

                # ---------- Location ----------
                loc_id = tx.run(
                    "CREATE (l:Location {address:$address,type:$type,"
                    "features:$features,time:$time}) RETURN id(l) AS lid",
                    **d["location"]
                ).single()["lid"]
                tx.run("MATCH (i),(l) WHERE id(i)=$iid AND id(l)=$lid "
                       "CREATE (i)-[:AT]->(l)", iid=inc_id, lid=loc_id)

                # ---------- Persons -----------
                person_ids = []
                for p in d["persons"]:
                    if not any(p.values()):
                        continue
                    pid = tx.run(
                        "CREATE (p:Person {name:$name,phone:$phone,role:$role,"
                        "relationship:$relationship,conditions:$conditions,"
                        "age:$age,sex:$sex}) RETURN id(p) AS pid", **p
                    ).single()["pid"]
                    person_ids.append(pid)
                    tx.run("MATCH (p),(i) WHERE id(p)=$pid AND id(i)=$iid "
                           "CREATE (p)-[:INVOLVED_IN]->(i)", pid=pid, iid=inc_id)
                    if p["role"].lower() == "caller" and call_ids:
                        tx.run("MATCH (p),(c) WHERE id(p)=$pid AND id(c)=$cid "
                               "CREATE (p)-[:MADE]->(c)", pid=pid, cid=call_ids[0])

                return {"incident": inc_id, "calls": call_ids,
                        "location": loc_id, "persons": person_ids}

            ids = ses.execute_write(_tx, call_data)

        return {"status": "success",
                "message": "Saved to Neo4j",
                "node_ids": ids}
    except Exception as e:
        return {"status": "error", "message": str(e)}
