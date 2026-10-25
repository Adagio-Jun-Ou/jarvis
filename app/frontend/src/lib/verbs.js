// Plain-language verbs for agent activity.
//
// Research signal (agent-UX 2026 guidance): activity is written in plain
// language, not tool-call names — "Searching enterprise knowledge" tells
// anyone what is happening; "search_knowledge" tells engineers. The
// execution log keeps the exact tool names (that is its contract); the
// live status line and the activity rail speak human.

const VERBS = {
  search_knowledge: 'Searching enterprise knowledge',
  list_knowledge: 'Listing knowledge sources',
  ingest_document: 'Indexing a document',
  ingest_workspace_file: 'Indexing a workspace file',
  read_inbox: 'Reading the inbox',
  search_email: 'Searching email',
  summarize_email: 'Summarizing an email',
  draft_email_reply: 'Drafting an email reply',
  send_email: 'Sending an email',
  create_task: 'Creating a task',
  update_task_status: 'Updating a task',
  list_my_tasks: 'Checking open tasks',
  plan_task: 'Writing the plan',
  check_plan_step: 'Updating the plan',
  check_acceptance: 'Verifying an acceptance criterion',
  delegate_subagents: 'Delegating to sub-agents',
  remember: 'Saving a memory',
  recall: 'Recalling memories',
  graph_query: 'Querying the knowledge graph',
  graph_neighbors: 'Exploring the knowledge graph',
  add_edge: 'Linking facts in the graph',
  view: 'Reading a file',
  create_file: 'Writing a file',
  str_replace: 'Editing a file',
  undo_file_change: 'Undoing a file change',
  present_files: 'Gathering output files',
  download_file: 'Downloading a file',
  bash: 'Running a command',
  run_python: 'Running Python',
  pip_install: 'Installing a package',
  create_spreadsheet: 'Building a spreadsheet',
  read_spreadsheet: 'Reading a spreadsheet',
  create_presentation: 'Building a presentation',
  create_document: 'Writing a document',
  browser_navigate: 'Opening a page',
  browser_extract: 'Reading a page',
  browser_click: 'Clicking on a page',
  browser_type: 'Typing on a page',
  browser_submit: 'Submitting a form',
  browser_screenshot: 'Capturing a page',
  api_read: 'Reading an internal API',
  api_act: 'Acting through an internal API',
  trigger_workflow: 'Triggering a workflow',
  list_standing_orders: 'Checking standing orders',
  run_standing_order: 'Running a standing order',
  search_practices: 'Consulting best practices',
  run_routine: 'Following a taught routine',
  code_outline: 'Reading code structure',
  code_definition: 'Looking up a definition',
  code_references: 'Finding references',
  code_diagnostics: 'Checking for code problems',
  code_search_symbols: 'Searching code symbols',
};

/** Human verb for one tool name. MCP tools name their server. */
export function verbFor(tool) {
  if (!tool) return 'Working';
  if (VERBS[tool]) return VERBS[tool];
  const mcp = /^mcp__([^_]+(?:_[^_]+)*?)__(.+)$/.exec(tool);
  if (mcp) return `Using ${mcp[1]}: ${mcp[2].replace(/_/g, ' ')}`;
  return tool.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

/** The one-line "what is happening right now" for a streaming turn. */
export function nowVerb(turn) {
  if (!turn) return '';
  const running = (turn.timeline || []).filter(
    (i) => i.kind === 'tool' && i.result == null
  );
  if (running.length) return `${verbFor(running[running.length - 1].tool)}…`;
  const sw = turn.swarm;
  if (sw) {
    const live = (sw.order || []).filter(
      (n) => sw.agents[n]?.status === 'running'
    );
    if (live.length) {
      return `${live.length} sub-agent${live.length === 1 ? '' : 's'} working…`;
    }
  }
  const last = (turn.timeline || [])[turn.timeline.length - 1];
  if (last?.kind === 'context') {
    return last.source === 'compaction'
      ? 'Compressing the conversation…'
      : 'Gathering context…';
  }
  if (turn.reasoning?.length && !turn.content) return 'Thinking…';
  if (turn.content) return 'Writing…';
  return 'Thinking…';
}
