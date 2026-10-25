

# ---- the cacheable prefix -------------------------------------------------
# Provider prompt caching holds exactly as far as the first changed byte.
# T0 must therefore be byte-identical across turns of a session: skill
# hints (keyed to the message) and the previous turn's reasoning digest
# used to ride in it, so the persona + constitution + workspace were
# re-paid on every single turn.


def test_t0_is_byte_stable_across_different_messages(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config

    config._settings = None
    from agent.context_engine import load_context

    def ctx(msg, hints, digest):
        return load_context(
            system_prompt="You are Alex.",
            user_email="demo@alex.dev",
            session_id="s-cache",
            last_user_msg=msg,
            sql=lambda *a, **k: [],
            memory_recall=lambda u, limit=8: [],
            kg_query=lambda u, t: [],
            constitution_block="THE LAW: be governed.",
            workspace_block="WORKSPACE: 3 files.",
            skill_hints=hints,
            reasoning_block=digest,
        )

    first = ctx("what is the Atlas budget?", "SKILL: budget-review", "")
    second = ctx("draft the vermeil email", "SKILL: email-drafting",
                 "DIGEST: last turn compared rate cards")

    assert first["T0"] == second["T0"], (
        "T0 changed between turns — the prompt cache re-pays the whole "
        "prefix every single message"
    )
    # The volatile content still reaches the prompt, in the session slice.
    assert "SKILL: email-drafting" in second["T2"]
    assert "DIGEST: last turn" in second["T2"]
    assert "SKILL" not in second["T0"]
