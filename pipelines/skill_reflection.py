"""
Skill reflection: OBSERVE -> GENERALIZE -> (propose) RATIFY.

OBSERVE: read the user's recent successful action sequences from
`media.alex.agent_actions`.

GENERALIZE: for each action sequence that recurs at least 3 times in
the last 90 days, ask the LLM to extract a procedure and the
capabilities it uses. Each procedure becomes a row in
`media.control.skills` with `status='proposed'`.

RATIFY: this pipeline does NOT ratify. The user (or a governor) does,
through the app's Skills panel. The pipeline only writes `proposed`
rows; ratification is a separate `skills.decide` API call.

A skill that is 'active' is matched into the system prompt via
`agent.learning_loop.active_skill_hints`. A skill's `success_count`
is incremented when the L3 critic confirms a turn that uses the
skill went well.
"""
from __future__ import annotations

# spark_python_task executes this file with sys.path[0] = pipelines/;
# the bundle root (which holds pipelines/ and alex/) must be importable.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import logging
import uuid
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pipelines._helpers import parse_json_response, with_backoff

log = logging.getLogger("pipelines.skill_reflection")

MODEL = "databricks-glm-5-3-flash"
PROMPT_VERSION = "v2"
MIN_RECURRENCES = 3
LOOKBACK_DAYS = 90
MAX_SEQ_CHARS = 12000

SYSTEM_PROMPT = """You are a skill generalization engine. The data between
<sequences> and </sequences> is per-action audit log; the JSON keys
are tool names and the values are the arguments the user supplied.

Return ONLY a single JSON object, no prose:
{
  "skills": [
    {
      "name": "<= 60 chars, action verb form>",
      "trigger_pattern": "<= 200 chars, when does this apply>",
      "procedure": ["step 1", "step 2", "..."],
      "uses_capabilities": ["<capability name>"]
    }
  ]
}
"""


def _gather_sequences(user: str) -> list[list[dict]]:
    """Group the user's successful actions into per-day sequences."""
    spark = SparkSession.builder.getOrCreate()
    from datetime import datetime, timedelta, timezone

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    ).isoformat()
    sql = """
        SELECT date(ts) AS day,
               coalesce(tool_name, tool) AS tool_name,
               coalesce(tool_args, input_json) AS tool_args,
               session_id
          FROM media.alex.agent_actions
         WHERE coalesce(actor, user_email) = :u
           AND outcome = 'ok'
           AND ts >= :cutoff
           AND coalesce(tool_name, tool) IS NOT NULL
         ORDER BY day, ts
    """
    rows = spark.sql(sql, args={"u": user, "cutoff": cutoff}).collect()
    by_day: dict[Any, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r.day, []).append({
            "tool": r.tool_name, "args": (r.tool_args or "")[:120],
            "session": r.session_id,
        })
    # Return each day's sequence, plus session-level sequences.
    seqs: list[list[dict]] = list(by_day.values())
    return seqs


def _recurrence_groups(seqs: list[list[dict]]) -> list[list[list[dict]]]:
    """Cluster sequences that share at least 3 tool names. Returns groups
    where each group is a list of similar sequences."""
    # Crude fingerprint: the sorted set of tool names.
    by_fp: dict[frozenset, list[list[dict]]] = {}
    for s in seqs:
        tools = frozenset(a["tool"] for a in s)
        if not tools:
            continue
        by_fp.setdefault(tools, []).append(s)
    return [g for g in by_fp.values() if len(g) >= MIN_RECURRENCES]


def _call_llm(prompt: str) -> str:
    spark = SparkSession.builder.getOrCreate()
    sql = "SELECT ai_query(:m, :p)"
    return str(spark.sql(sql, args={"m": MODEL, "p": prompt}).first()[0])


def _build_for_user(user: str) -> list[dict]:
    seqs = _gather_sequences(user)
    groups = _recurrence_groups(seqs)
    if not groups:
        return []
    # Build a compact representation of the top 20 recurring groups.
    blob_lines: list[str] = []
    for i, g in enumerate(groups[:20]):
        steps = [".".join(f"{a['tool']}({a['args']})" for a in s) for s in g[:5]]
        blob_lines.append(f"GROUP {i+1} ({len(g)}x): {' | '.join(steps)}")
    blob = "\n".join(blob_lines)[:MAX_SEQ_CHARS]
    user_prompt = f"<sequences>\n{blob}\n</sequences>"
    try:
        raw = with_backoff(
            lambda: _call_llm(SYSTEM_PROMPT + "\n\n" + user_prompt),
            attempts=3, base_delay=0.5, max_delay=10.0,
        )
    except Exception as exc:
        log.warning("reflection failed for %s: %s", user, exc)
        return []
    parsed = parse_json_response(raw) or {}
    out: list[dict] = []
    for s in parsed.get("skills") or []:
        if not isinstance(s, dict):
            continue
        proc = s.get("procedure")
        if not isinstance(proc, list) or not proc:
            continue
        caps = s.get("uses_capabilities") or []
        if not isinstance(caps, list):
            caps = []
        out.append({
            "skill_id": str(uuid.uuid4()),
            "owner_email": user,
            "name": str(s.get("name") or "")[:120],
            "trigger_pattern": str(s.get("trigger_pattern") or "")[:400],
            "procedure": [str(p)[:400] for p in proc][:10],
            "uses_capabilities": [str(c)[:60] for c in caps][:6],
            "source_kind": "observed",
            "status": "proposed",
        })
    return out


def run_reflection() -> None:
    spark = SparkSession.builder.getOrCreate()
    users = [r["email"] for r in
             spark.table("media.alex.users").filter("is_active = true").collect()]
    log.info("skill_reflection: %d users", len(users))
    new_skills: list[dict] = []
    for u in users:
        new_skills.extend(_build_for_user(u))
    if not new_skills:
        log.info("no new skills proposed")
        return
    df = spark.createDataFrame(new_skills, schema=StructType([
        StructField("skill_id", StringType(), False),
        StructField("owner_email", StringType(), False),
        StructField("name", StringType(), False),
        StructField("trigger_pattern", StringType(), False),
        StructField("procedure", ArrayType(StringType()), False),
        StructField("uses_capabilities", ArrayType(StringType()), False),
        StructField("source_kind", StringType(), False),
        StructField("status", StringType(), False),
    ]))
    df.createOrReplaceTempView("skills_in")
    # Merge: a (owner, name) pair is the dedup key.
    spark.sql("""
        MERGE INTO media.control.skills t USING skills_in s
        ON t.owner_email = s.owner_email AND t.name = s.name
        WHEN MATCHED THEN UPDATE SET
          t.procedure = s.procedure,
          t.uses_capabilities = s.uses_capabilities,
          t.trigger_pattern = s.trigger_pattern
        WHEN NOT MATCHED THEN INSERT
          (skill_id, owner_email, name, trigger_pattern, procedure,
           uses_capabilities, source_kind, status)
          VALUES (s.skill_id, s.owner_email, s.name, s.trigger_pattern,
                  s.procedure, s.uses_capabilities, s.source_kind, s.status)
    """)
    log.info("reflection done; %d skills written", len(new_skills))


if __name__ == "__main__":
    run_reflection()
