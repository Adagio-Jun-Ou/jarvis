-- ===========================================================================
-- grants.sql — least-privilege grants for every principal.
--
-- Split from permissions.sql so the DDL job can run it LAST: a GRANT
-- references tables created across app_schema.sql, control_plane.sql,
-- taskboard.sql, oversight.sql and the rest, and a grant on a table
-- that does not exist yet fails the whole script. Table DDL first,
-- grants once everything exists.
--
-- The properties enforced here are documented in permissions.sql.
-- ===========================================================================


-- Agent SP: only EXECUTE on stored procs (in tools.sql), SELECT/INSERT on
-- the limited operational tables it needs, NEVER MODIFY/UPDATE/DELETE on any
-- ledger, NEVER on any volume.
GRANT USAGE ON SCHEMA media.alex     TO sp-alex-agent;
GRANT USAGE ON SCHEMA media.control  TO sp-alex-agent;
GRANT USAGE ON SCHEMA media.content  TO sp-alex-agent;
GRANT SELECT, INSERT ON media.alex.agent_actions         TO sp-alex-agent;
GRANT SELECT         ON media.alex.users                 TO sp-alex-agent;
GRANT SELECT, INSERT, UPDATE ON media.alex.tasks          TO sp-alex-agent;
GRANT SELECT, INSERT ON media.alex.task_comments         TO sp-alex-agent;
GRANT SELECT         ON media.alex.task_shares            TO sp-alex-agent;
GRANT SELECT         ON media.alex.email_drafts           TO sp-alex-agent;
GRANT SELECT, INSERT ON media.alex.standing_order_files  TO sp-alex-agent;
GRANT SELECT         ON media.content.assets              TO sp-alex-agent;
GRANT SELECT         ON media.content.transcripts         TO sp-alex-agent;
GRANT SELECT         ON media.content.captions            TO sp-alex-agent;
GRANT SELECT         ON media.control.standing_orders     TO sp-alex-agent;
GRANT SELECT         ON media.control.delegations         TO sp-alex-agent;
GRANT SELECT         ON media.control.delegable_capabilities TO sp-alex-agent;
GRANT SELECT         ON media.control.supervision         TO sp-alex-agent;
GRANT SELECT         ON media.control.user_roles          TO sp-alex-agent;
GRANT SELECT         ON media.control.best_practices      TO sp-alex-agent;
GRANT SELECT         ON media.control.mcp_servers         TO sp-alex-agent;
GRANT SELECT         ON media.control.user_preferences    TO sp-alex-agent;
GRANT INSERT         ON media.control.supervision_findings TO sp-alex-agent;
GRANT SELECT         ON media.control.workspace_policy    TO sp-alex-agent;
GRANT SELECT, INSERT, UPDATE ON media.control.mcp_tool_pins TO sp-alex-agent;
GRANT SELECT, INSERT ON media.control.standing_order_runs TO sp-alex-agent;
GRANT INSERT         ON media.control.knowledge_access_log TO sp-alex-agent;
GRANT SELECT         ON media.control.approved_files      TO sp-alex-agent;
GRANT SELECT         ON media.control.instance_library    TO sp-alex-agent;
GRANT SELECT         ON media.control.user_knowledge_chunks TO sp-alex-agent;
GRANT SELECT         ON media.control.tool_integrations   TO sp-alex-agent;
GRANT SELECT         ON media.control.workflow_catalog    TO sp-alex-agent;
GRANT SELECT         ON media.control.employee_acl        TO sp-alex-agent;
GRANT SELECT         ON media.control.instance_acl        TO sp-alex-agent;
GRANT SELECT         ON media.control.skills              TO sp-alex-agent;
GRANT SELECT         ON media.control.thinking_traces     TO sp-alex-agent;
GRANT SELECT         ON media.control.constitution         TO sp-alex-agent;

-- The board's plan/acceptance surfaces, with the write asymmetry
-- expressed AT THE GRANT LAYER, not only in application code:
--   * the plan is the agent's â€” full write (set_plan replaces rows);
--   * acceptance criteria are the humans' â€” the agent may only UPDATE
--     a verdict (met/unmet), never INSERT or DELETE a criterion. An
--     agent that could edit its own definition of done would grade its
--     own homework, and this grant makes that impossible even if the
--     application check regresses.
--   * task_deps and task_proposals get NO agent grant on purpose:
--     dependencies and the email->task gate are human decisions.
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.task_plans      TO sp-alex-agent;
GRANT SELECT, UPDATE                 ON media.control.task_acceptance TO sp-alex-agent;
-- The agent may ASK a question, never resolve one: no UPDATE, so the
-- `answer` column is writable only through the app — a person.
GRANT SELECT, INSERT                 ON media.control.task_questions  TO sp-alex-agent;
GRANT SELECT                         ON media.alex.task_board_view    TO sp-alex-agent;
GRANT SELECT                         ON media.alex.inbox_view         TO sp-alex-agent;

-- App SP: needs to read & write on behalf of users. Volumes are
-- granted ONLY to the app SP, never to the human group â€” every volume
-- access goes through the app's OBO call, which applies the user filter.
GRANT USAGE ON SCHEMA media.alex     TO sp-alex-app;
GRANT USAGE ON SCHEMA media.control  TO sp-alex-app;
GRANT USAGE ON SCHEMA media.content  TO sp-alex-app;
GRANT USAGE ON SCHEMA media.audience TO sp-alex-app;
GRANT USAGE ON SCHEMA media.ops      TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.users         TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.tasks         TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.task_comments TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.task_shares   TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.email_drafts  TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.standing_order_files TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.content.assets     TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.content.transcripts TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.content.captions   TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.alex.agent_actions  TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.browser_sessions TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.browser_history  TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.delegations TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.supervision TO sp-alex-app;
GRANT SELECT         ON media.control.user_roles          TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.best_practices      TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.mcp_servers         TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.user_preferences TO sp-alex-app;
GRANT SELECT, INSERT ON media.control.supervision_findings TO sp-alex-app;
GRANT SELECT         ON media.control.workspace_policy    TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.mcp_tool_pins       TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.standing_order_runs TO sp-alex-app;
GRANT SELECT, INSERT ON media.control.knowledge_access_log TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.forget_requests TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.skills     TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.user_knowledge_chunks TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.workspaces      TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.libraries       TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.api_specs       TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.content.raw_files    TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.content.transcripts_blob TO sp-alex-app;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.browser_evidence TO sp-alex-app;

-- The chat session store. One storage model, two backends: in
-- production these tables ARE the session system, touched on every
-- chat turn. DELETE covers session deletion and turn pruning.
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.sessions      TO sp-alex-app;
GRANT SELECT, INSERT, DELETE         ON media.control.session_turns TO sp-alex-app;

-- The task board's control tables (governance/taskboard.sql). The app
-- serves every board API on these; DELETE only where replacement
-- semantics live (set_plan / set_criteria replace their rows; a dep
-- edge is removed, never updated). task_messages is the thread â€”
-- append-only, so no UPDATE/DELETE from anyone.
GRANT SELECT, INSERT, UPDATE         ON media.control.task_crew        TO sp-alex-app;
GRANT SELECT, INSERT                 ON media.control.task_messages    TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE         ON media.control.task_questions   TO sp-alex-app;
-- Per-turn run accounting: the app writes a row as each working turn
-- ends and the oversight console reads them back. Append-only ledger.
GRANT SELECT, INSERT                 ON media.ops.run_metrics           TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE         ON media.control.task_runs        TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.task_plans       TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.task_acceptance  TO sp-alex-app;
GRANT SELECT, INSERT, DELETE         ON media.control.task_deps        TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE         ON media.control.task_proposals   TO sp-alex-app;

-- Read models built by the jobs, read by the app and the agent tools.
GRANT SELECT ON media.alex.inbox_view      TO sp-alex-app;
GRANT SELECT ON media.alex.task_board_view TO sp-alex-app;

-- Completed runs are checkpointed into the memory manifest by the app
-- (taskboard._remember_run), and the transparency panel reads it.
GRANT SELECT, INSERT, UPDATE ON media.control.vector_memory_mirror TO sp-alex-app;
GRANT SELECT, INSERT, UPDATE ON media.control.kg_mirror            TO sp-alex-app;

-- Jobs SP: writes to operational and ledgers, never reads memory / KG.
GRANT USAGE ON SCHEMA media.alex     TO sp-alex-jobs;
GRANT USAGE ON SCHEMA media.control  TO sp-alex-jobs;
GRANT USAGE ON SCHEMA media.content  TO sp-alex-jobs;
GRANT USAGE ON SCHEMA media.audience TO sp-alex-jobs;
GRANT USAGE ON SCHEMA media.ops      TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.users         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.tasks         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.task_comments TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.email_drafts  TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.standing_order_files TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.content.assets     TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.content.transcripts TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.content.rights       TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.content.captions   TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.approved_files      TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.instance_library    TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.delegations         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.supervision         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.user_roles          TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.best_practices      TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.standing_order_runs TO sp-alex-jobs;
GRANT SELECT, INSERT ON media.control.knowledge_access_log               TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.forget_requests     TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.user_knowledge_chunks TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.skills                TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.taught_routines      TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.suggestion_feedback  TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.standing_order_briefs TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.employee_acl         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.instance_acl         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.delegable_capabilities TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.consolidation_queue  TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.vector_memory_mirror TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.kg_mirror             TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.morning_briefs       TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.emails                  TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.email_summaries         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.email_intel             TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.email_intel_quarantine  TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.email_ingest_state      TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.transcripts_chunks     TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.risk_signals         TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.investigations       TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.investigation_access_log TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.exec                          TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.ops.gateway_logs             TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.ops.gateway_logs_payload     TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.ops.agent_reviews             TO sp-alex-jobs;
-- Read models the jobs materialise (inbox_build, task_board_build):
-- MERGE needs the full set, and the anti-join DELETE prunes ghosts.
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.inbox_view              TO sp-alex-jobs;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.alex.task_board_view         TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.workspaces      TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.libraries       TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.api_specs       TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.content.raw_files    TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.content.transcripts_blob TO sp-alex-jobs;
GRANT READ VOLUME, WRITE VOLUME ON media.alex.browser_evidence TO sp-alex-jobs;

-- Gateway SP: writes to ops tables only.
GRANT USAGE ON SCHEMA media.ops TO sp-alex-gw;
GRANT SELECT, INSERT ON media.ops.gateway_logs          TO sp-alex-gw;
GRANT SELECT, INSERT ON media.ops.gateway_logs_payload  TO sp-alex-gw;

-- Human groups.
GRANT USAGE ON SCHEMA media.alex     TO `alex-users`;
GRANT USAGE ON SCHEMA media.control  TO `alex-users`;
GRANT SELECT ON media.alex.users                          TO `alex-users`;
GRANT SELECT, INSERT, UPDATE ON media.alex.tasks          TO `alex-users`;
GRANT SELECT, INSERT       ON media.alex.task_comments    TO `alex-users`;
GRANT SELECT               ON media.alex.task_shares      TO `alex-users`;
GRANT SELECT, INSERT       ON media.alex.email_drafts     TO `alex-users`;
GRANT SELECT               ON media.content.assets        TO `alex-users`;
GRANT SELECT               ON media.content.transcripts   TO `alex-users`;
GRANT SELECT               ON media.content.captions      TO `alex-users`;
-- INSERT-only on forget_requests; the job flips 'applied' to true.
GRANT SELECT, INSERT       ON media.control.forget_requests TO `alex-users`;

-- Governors own the control plane. They can amend, ratify, investigate.
GRANT USAGE ON SCHEMA media.alex     TO `alex-governors`;
GRANT USAGE ON SCHEMA media.control  TO `alex-governors`;
GRANT USAGE ON SCHEMA media.content  TO `alex-governors`;
GRANT USAGE ON SCHEMA media.audience TO `alex-governors`;
GRANT SELECT, INSERT ON media.control.constitution          TO `alex-governors`;
GRANT SELECT, INSERT ON media.control.constitution_amendments TO `alex-governors`;
GRANT SELECT, INSERT ON media.control.constitution_violations TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.employee_acl TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.instance_acl TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.delegable_capabilities TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.delegations TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.supervision TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.user_roles  TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.best_practices TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.mcp_servers   TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.workspace_policy TO `alex-governors`;
GRANT SELECT, UPDATE ON media.control.supervision_findings TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.mcp_tool_pins TO `alex-governors`;
GRANT SELECT ON media.control.standing_order_runs TO `alex-governors`;
GRANT SELECT ON media.control.knowledge_access_log TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.skills     TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.approved_files TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.workflow_catalog TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.tool_integrations TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.taught_routines TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.suggestion_feedback TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.standing_orders TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.standing_order_briefs TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.risk_signals TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.investigations TO `alex-governors`;
GRANT SELECT, INSERT, UPDATE, DELETE ON media.control.investigation_access_log TO `alex-governors`;
