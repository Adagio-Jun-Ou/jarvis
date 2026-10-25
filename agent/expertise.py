"""
Expertise retrieval: merge 'company way' (instance library) with
'industry way' (Vector Search on curated corpora). The agent prefers
the company way; the industry way is a fallback.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.expertise")


PARTNER_PROMPT = (
    "When answering, prefer the company's approved procedures and voice "
    "over general industry practice. If both apply, lead with the company "
    "way; note the industry contrast only if it adds clarity."
)


def merge_retrieval(
    company_hits: list[dict], craft_hits: list[dict], *, company_weight: float = 0.7
) -> list[dict]:
    """Merge two retrieval lists by interleaving with weights. The
    company list goes first. Each item is a dict with at least a 'text'
    or 'content' field."""
    if not craft_hits:
        return list(company_hits or [])
    if not company_hits:
        return list(craft_hits)
    out: list[dict] = []
    # Round-robin with weights.
    n_c = max(1, int(round(len(company_hits) * company_weight)))
    n_i = max(1, int(round(len(craft_hits) * (1 - company_weight))))
    i = j = 0
    while i < len(company_hits) or j < len(craft_hits):
        for _ in range(n_c):
            if i < len(company_hits):
                out.append(company_hits[i])
                i += 1
        for _ in range(n_i):
            if j < len(craft_hits):
                out.append(craft_hits[j])
                j += 1
    return out


def to_company_first_prompt(merged: list[dict]) -> str:
    """Render the merged list as a prompt fragment that the model can
    consume. The header tells the model which is which."""
    if not merged:
        return ""
    parts: list[str] = ["RETRIEVED MATERIAL (company-approved first, then industry):"]
    for i, h in enumerate(merged):
        kind = h.get("kind") or "industry"
        text = h.get("text") or h.get("content") or h.get("page_content") or ""
        parts.append(f"[{i + 1} | {kind}] {text[:400]}")
    return "\n".join(parts)
