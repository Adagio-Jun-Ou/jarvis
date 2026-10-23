# Harness Design Rationale — why Alex's agent loop is shaped the way it is

Companion to `07_HARNESS_AND_SDK.md` (the how). This is the *why*: the design
decisions behind the harness, and what we deliberately rejected from each
reference system.

## The core tension
An enterprise AI employee must be simultaneously (a) capable and improving,
(b) fully governed, and (c) fast and cheap at runtime. Most harnesses optimise
one at the cost of the others. Alex's answer is to make governance *structural*
(so capability doesn't erode safety) and to push all expensive work off the
hot path (so governance doesn't erode speed).

## Decision log

**1. Learning is proposed, never self-granted.**
Hermes shows the value of a learning loop — skills from experience, a model of
the user that compounds. We adopt the loop but invert its authority model: a
learned skill is a *proposal* that can only chain already-approved capabilities
and must be human-ratified. Rationale: in a media enterprise, an agent that can
teach itself new *authority* is a liability; an agent that teaches itself new
*efficiency within fixed authority* is pure upside.

**2. Review is triggered, not omnipresent.**
Codex-style always-on step review is too harsh for a plain Python VM and taxes
every action. We gate review on *consequence* (external/irreversible tools,
in-flight constitutional procedures, rights/money, elevated risk), and when it
fires we use a small model with a strict fixed preset. Rationale: computation
on an isolated no-egress VM is already contained by isolation; only
*consequence* needs a second opinion. This keeps p50 latency low while keeping
the "check during execution" property.

**3. Thinking is a channel, not context.**
OpenCode-style visible reasoning is valuable for trust, but naively feeding raw
chain-of-thought back into later turns rots context and inflates tokens. We
stream raw thinking to the UI and archive it, but only an ≤80-token decision
digest re-enters context. Rationale: the human gets full transparency; the
model gets a clean, compact working memory.

**4. Open weights where thinking must be shown.**
We route execution-planning to open-weight reasoners (Hermes/DeepSeek/Qwen)
when "show reasoning" is on, because surfacing a closed model's hidden reasoning
as if it were faithful is misleading. Rationale: inspectable thinking should be
*actually* inspectable.

**5. The SDK encodes governance in the type system.**
Rather than a governance layer bolted beside the SDK, capabilities/tools/
workflows validate at build: an undeclared capability fails at `deploy()`.
Rationale: the cheapest place to catch a governance error is the developer's
terminal, not production.

## What we rejected
- Marketplace/plugin tools (unvetted capability surface) — all tools are UC
  functions or SDK-declared + governor-approved.
- Always-on autonomous action — every external effect is human-gated by the
  constitution.
- Laptop-bound agents — Alex runs on Databricks compute with UC identity, so
  governance and data locality hold.
- Raw-thought context accumulation — replaced by decision digests.

See `07_HARNESS_AND_SDK.md` for the implementation and `00_DEBUG_AND_BUILD_GUIDE.md`
for how it all deploys.
