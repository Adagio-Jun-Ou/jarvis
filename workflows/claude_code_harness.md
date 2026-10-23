# Complex workflow building via Claude Code (governed)

For problems too gnarly for a single agent turn — "build a nightly pipeline
that pulls closed-caption files, aligns them to air logs, and files
discrepancies as tasks" — Alex escalates to a **Claude Code session on its
own VM** (Agent Bricks supports the Claude Code SDK as a harness).

Flow:
1. Alex drafts a WORKPLAN.md in the user's workspace (goal, constraints,
   which approved capabilities the result may use) and asks the user to approve.
2. On approval, a job starts `claude-code` headless on the agent VM with cwd =
   the user's work/ directory. No network egress: packages come from the
   workspace's approved index mirror; data access only via the same UC tools.
3. Claude Code iterates — writes the n8n workflow JSON / pipeline code, runs
   tests in the sandbox, self-reviews.
4. Output lands in outputs/: the workflow definition + a review report.
   Importing it into n8n / the workflow catalog is a governor action —
   the AI builds workflows, humans approve them into the catalog.

Token discipline: the Claude Code session is bounded by a max-turn and
max-token budget from media.control.model_registry, and runs on the fast/
frontier mix it chooses under that budget.
