# Jarvis Skills - Tool Usage Patterns

This document teaches Jarvis how to use tools effectively, autonomously, and in combination with the agentic workflow.

**For Composable Workflows, see [SKILLS-WORKFLOWS.md](SKILLS-WORKFLOWS.md)** - Visual workflow automation, scheduled tasks, and long-running deployments.

## Tool Inventory

Jarvis has **146 tools** across **30 categories**:

### File Operations (6 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `read_file` | Read file contents (10MB limit, SSRF-protected) | Safe |
| `write_file` | Write to file | Ask |
| `list_files` | List directory contents | Safe |
| `delete_file` | Delete file/directory | Ask |
| `shell` | Quick sandboxed commands (safe commands only) | Safe |
| `system_info` | Get system information | Safe |

### Terminal Access (7 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `terminal_execute` | Execute shell commands (HIGH-risk BLOCKED) | Risk-based |
| `terminal_cwd` | Get/change working directory | Safe |
| `terminal_history` | View command history | Safe |
| `terminal_session` | Manage terminal sessions | Safe |
| `terminal_mode` | Change sandbox/project/full mode | Ask |
| `computer_status` | CPU, memory, disk usage | Safe |
| `install_package` | Install system packages (pip, npm, apt) | Ask |

**Terminal Security:**
- HIGH risk commands are BLOCKED: `rm -rf /*`, `rm -rf /`, `rm -rf ~`, `curl | bash`, fork bombs
- MEDIUM risk requires permission: `apt install`, `pip install`, `docker rm`
- LOW risk auto-approved: `ls`, `cat`, `git`, `python`

### Web Tools (7 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `web_search` | Search DuckDuckGo/Google (safe search ON) | Safe* |
| `web_fetch` | Extract clean text from URL (SSRF-protected, safe) | Safe* |
| `image_search` | Search for images (safe search ON) | Safe* |
| `image_download` | Download image (size-limited, SSRF-protected) | Safe* |
| `image_analyze` | Analyze image content | Safe |
| `image_read` | Read image file | Safe |

**Web Safety Measures:**

Jarvis has **built-in safety filters** for all web tools to protect users from harmful content:

| Category | Behavior | Can Bypass? |
|----------|----------|-------------|
| **Adult/Sexual** | Blocked by default, use `unsafe: true` | Yes (with permission) |
| **Violence/Gore** | Blocked by default, use `unsafe: true` | Yes (with permission) |
| **Harmful** (suicide, self-harm) | Blocked by default, use `unsafe: true` | Yes (with permission) |
| **Hate Speech** | Blocked by default, use `unsafe: true` | Yes (with permission) |
| **Illegal** (CSAM, drug synthesis) | **PERMANENTLY BLOCKED** | No |
| **Malware** | **PERMANENTLY BLOCKED** | No |

**Why These Restrictions Exist:**
1. **User Safety** - Prevents accidental exposure to harmful content
2. **Legal Compliance** - Some content (CSAM, illegal material) cannot be accessed legally
3. **Ethical AI** - Jarvis should not assist in harmful activities
4. **Workspace Safety** - Many searches happen in professional environments

**Using `unsafe: true`:**
- For sensitive but legitimate research (e.g., medical, academic, journalism)
- Requires explicit user approval via permission system
- Returns `PERMISSION_REQUIRED` for user confirmation
- Example: `web_search: adult content research (unsafe: true)`

**SSRF Protection:**
All web fetch tools block access to:
- Internal networks (127.0.0.1, 10.x, 192.168.x, 172.16-31.x)
- Cloud metadata endpoints (169.254.169.254)
- Blocked URL schemes (file://, ftp://, javascript:)

*Safe by default, requires permission when `unsafe: true` is used.

### Browser Automation (10 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `browser_launch` | Start headless browser | Ask |
| `browser_navigate` | Navigate to URL | Safe |
| `browser_click` | Click element | Ask |
| `browser_type` | Type into input field | Ask |
| `browser_screenshot` | Capture page | Safe |
| `browser_get_content` | Extract page text | Safe |
| `browser_evaluate` | Execute JavaScript | Ask |
| `browser_wait` | Wait for element | Safe |
| `browser_list` | List open tabs | Safe |
| `browser_close` | Close browser | Safe |

### Screen Automation (11 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `see_screen` | Capture and analyze screen | Ask |
| `find_element` | Find UI element by text | Safe |
| `click_element` | Click at coordinates/element | Ask |
| `type_text` | Type at cursor position | Ask |
| `press_key` | Press key | Ask |
| `hotkey` | Send keyboard combo | Ask |
| `scroll` | Scroll screen | Safe |
| `enable_automation` | Enable/disable automation | Ask |
| `get_dev_context` | Get development context | Safe |
| `get_learning_stats` | Get automation stats | Safe |
| `what_am_i_doing` | Analyze current activity | Safe |

### Data Analysis (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `data_analysis` | Analyze CSV/JSON files | Safe |
| `sql_query` | Execute SELECT queries (safely) | Safe |
| `statistical` | Statistical analysis | Safe |

### Security Tools (6 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `security_scan` | Scan for sensitive data | Safe |
| `security_status` | Get security status | Safe |
| `security_audit` | Audit security practices | Safe |
| `security_report` | Generate security report | Safe |
| `data_flow_monitor` | Track data flows | Safe |
| `whitelist_manager` | Manage DLP whitelist | Ask |

### Refinement Tools (7 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `refine_scan` | Scan code for bugs/security/style | Safe |
| `refine_propose` | Propose improvements | Safe |
| `refine_apply` | Apply proposed changes | Ask |
| `refine_rollback` | Rollback changes | Ask |
| `refine_status` | Check refinement status | Safe |
| `self_reflect` | Self-code analysis | Safe |

### Evolution Tools (15 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `evolution_status` | Get evolution status | Safe |
| `evolution_pathway` | Get evolution pathway | Safe |
| `propose_evolution` | Propose evolution | Safe |
| `propose_gene` | Propose gene | Safe |
| `gene_list` | List genes | Safe |
| `gene_approval` | Approve gene | Ask |
| `self_edit` | Edit own code | Ask |
| `daydream_verify` | Verify daydream solutions | Safe |
| `discoveries` | List discoveries | Safe |
| `knowledge_feed` | Feed knowledge | Safe |
| `capability_proposals` | List capability proposals | Safe |
| `tool_competency` | Get tool competency | Safe |
| `tool_practice` | Practice tool usage | Safe |
| `sketch_pad` | Quick insight capture | Safe |

### Learning Tools (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `erdos_math` | Math problem exploration | Safe |
| `defense` | Security defense | Safe |
| `learning` | General learning | Safe |

### Free Time Tools (7 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `free_time_stats` | Get free time statistics | Safe |
| `sandbox_explore` | Explore in sandbox | Safe |
| `read_sandbox_file` | Read sandbox files | Safe |
| `view_sandbox_results` | View sandbox results | Safe |
| `approve_permission` | Approve permission request | Ask |
| `deny_permission` | Deny permission request | Ask |
| `view_permission_requests` | View pending requests | Safe |

### Voice/Audio (11 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `speech_tokenize` | Tokenize speech | Safe |
| `speech_synthesize` | Synthesize speech | Safe |
| `transcribe` | Speech to text | Safe |
| `tone_analysis` | Analyze speech tone | Safe |
| `voice_stats` | Voice statistics | Safe |
| `identify_speaker` | Identify speaker | Safe |
| `register_speaker` | Register new speaker | Ask |
| `conversational_speech` | Conversational speech | Safe |
| `file_index` | Index files for search | Safe |
| `file_search` | Search indexed files | Safe |
| `embedding` | Generate embeddings | Safe |

### Tool Creation (8 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `tool_create` | Create new tool | Ask |
| `tool_test` | Test tool in sandbox | Safe |
| `tool_validate` | Validate tool code | Safe |
| `tool_request_deploy` | Request deployment | Ask |
| `tool_list_pending` | List pending tools | Safe |
| `tool_approve` | Approve tool deployment | Ask |
| `tool_reject` | Reject tool | Ask |
| `tool_list_sandbox` | List sandbox tools | Safe |

### Meta Tools (6 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `list_tools` | Show all available tools | Safe |
| `get_tool_info` | Detailed info about tool | Safe |
| `find_related_tools` | Find tools for task | Safe |
| `get_capabilities` | Summary of capabilities | Safe |
| `list_skills` | List available skills | Safe |
| `get_skill_info` | Get skill documentation | Safe |

### Home Automation (2 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `home_control` | Control home devices | Ask |
| `home_status` | Get home status | Safe |

### Physics/Spatial (6 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `collision_detection` | Detect collisions | Safe |
| `particle_system` | Particle simulation | Safe |
| `physics_analysis` | Physics analysis | Safe |
| `spatial_calculation` | Spatial calculations | Safe |
| `design` | Design calculations | Safe |
| `physics_simulation` | Run physics sim | Safe |

### Docker/N8N (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `docker` | Docker operations | Ask |
| `docker_compose` | Docker compose | Ask |
| `n8n` | Workflow automation | Ask |

### Web UI Refine (8 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `web_ui_view` | View web UI | Safe |
| `web_ui_analyze` | Analyze UI | Safe |
| `web_ui_propose` | Propose UI changes | Safe |
| `web_ui_approve` | Approve changes | Ask |
| `web_ui_reject` | Reject changes | Ask |
| `web_ui_screenshot` | Screenshot UI | Safe |
| `web_ui_history` | View UI history | Safe |
| `mobile_ui_view` | View mobile UI | Safe |

### System Sense (2 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `system_sense` | Sense system state | Safe |
| `system_watch` | Watch system changes | Safe |

### Insight Search (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `insight_search` | Search through accumulated insights and knowledge | Safe |
| `insight_heatmap` | Get word frequency heatmap from insights | Safe |
| `insight_stats` | Get statistics about accumulated insights | Safe |

### UI Sense (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `ui_sense` | Sense UI state | Safe |
| `ui_watch` | Watch UI changes | Safe |
| `ui_visualize` | Visualize UI | Safe |

### Art/Philosophy (3 tools)
| Tool | Description | Permission |
|------|-------------|------------|
| `art_analysis` | Analyze artwork | Safe |
| `philosophy_analysis` | Analyze philosophical concepts | Safe |
| `semantic_connection` | Find semantic connections | Safe |

## Core Philosophy

Jarvis has **full agency** over tool usage with the **agentic loop**:

1. **Plan the workflow** - Identify all steps needed before starting
2. **Execute tools autonomously** - Use best tool without asking permission (unless dangerous)
3. **Evaluate completion** - After each tool, ask: "Is the task done?"
4. **Continue if needed** - If incomplete, determine next action and continue
5. **Report results** - Summarize what was accomplished

## Agentic Workflow Integration

The agentic loop (`max_agentic_iterations: 5`) allows Jarvis to chain tools automatically:

```
User: "Find all Python files with 'async def' and summarize their purpose"

Internal Process:
1. shell: find . -name "*.py" -exec grep -l "async def" {} \;  → Get file list
2. [EVALUATE: Task incomplete, need to read files]
3. read_file: file1.py → Read content
4. read_file: file2.py → Read content  
5. [EVALUATE: Task incomplete, need summary]
6. Summarize findings → Present to user
```

**CRITICAL**: After each tool execution, assess completion. If incomplete, the backend model (GLM-5) determines the next action.

## Tool Permission Model

| Permission Level | Behavior | Examples |
|-----------------|----------|----------|
| `requires_permission = False` | Execute immediately | read_file, web_search, list_files |
| `requires_permission = True` | Ask before executing | write_file, delete_file, docker |
| Risk-based | Auto-detect danger level | terminal: high-risk commands blocked |

### High-Risk Actions (Always Ask)
- Deleting files (`delete_file`)
- Writing to files (`write_file`)
- Running shell commands (`terminal_execute`, `shell`)
- Docker operations (`docker`) 
- Screen automation (`see_screen`, `click_element`, `type_text`)
- Browser automation (`browser_launch`, `browser_navigate`)

### Safe Actions (No Permission Needed)
- Reading files (`read_file`, `list_files`)
- Web search/fetch (`web_search`, `web_fetch`)
- Code analysis (`refine_scan`, `security_scan`)
- Data analysis (`data_analysis`, `sql_query`)
- Meta tools (`list_tools`, `get_tool_info`)

## Tool Chaining Patterns

### Pattern 1: Investigate and Report
```
User: "What's using the most disk space?"

1. terminal_execute: df -h  → Check disk usage
2. terminal_execute: du -h --max-depth=1 /home/user 2>/dev/null | head -20
3. [EVALUATE: Need to go deeper into large directories]
4. terminal_execute: du -h --max-depth=1 /home/user/jarvis 2>/dev/null | head -20
5. Summarize: "The largest directories are..."
```

### Pattern 2: Find and Analyze
```
User: "Check for security issues in the authentication code"

1. shell: find . -name "*.py" | xargs grep -l "auth\|login\|password" | head -10
2. security_scan: <project_root>
3. [EVALUATE: Need to review specific files]
4. read_file: auth.py
5. refine_scan: auth.py
6. Summarize findings and suggest fixes
```

### Pattern 3: Web Research Deep Dive
```
User: "What's the best approach for implementing rate limiting?"

1. web_search: "rate limiting best practices Python 2024"
2. [EVALUATE: Need content from results]
3. web_fetch: top 3 URLs from results
4. web_search: "token bucket vs sliding window rate limiting"
5. web_fetch: relevant URLs
6. Synthesize: "Rate limiting approaches include..."
```

### Pattern 4: Multi-File Code Change
```
User: "Add logging to all API endpoints"

1. shell: find . -name "*.py" | xargs grep -l "@app.route\|@router" | head -20
2. [EVALUATE: Need to read each file]
3. FOR EACH FILE:
   a. read_file: <file>
   b. Identify endpoints without logging
   c. write_file: <file> (with logging added)
4. Verify: shell: grep -c "logger" *.py
5. Report: "Added logging to N endpoints across M files"
```

### Pattern 5: Insight Search (Knowledge Retrieval)
```
User: "What has Jarvis learned about patterns?"

1. insight_search: query="pattern discovery" limit=10
2. [EVALUATE: Results may need analysis]
3. insight_heatmap: limit=50  → See common themes
4. [EVALUATE: May need more specific search]
5. insight_search: query="cross_domain pattern" source="insight_bridge"
6. Synthesize: "Based on N insights, patterns show..."
```

### Pattern 6: System Health Check
```
User: "How is the system doing?"

1. system_sense: focus=all detailed=true
2. [EVALUATE: Check for issues]
3. IF memory high:
   a. Identify memory-intensive processes
   b. Suggest cleanup actions
4. IF subsystems failing:
   a. Get detailed status for each
   b. Report issues and suggested fixes
5. Summarize: "System status: CPU X%, Memory Y%, N subsystems healthy"
```

## Critical Tool Usage Rules

### Rule 1: Use Native Tools Over Shell
```
WRONG: shell: cat /path/to/file.py | head -50
RIGHT: read_file: /path/to/file.py (returns structured content)

WRONG: shell: ls -la /home/user
RIGHT: list_files: /home/user (returns structured output)
```

### Rule 2: Chain Tools Intelligently
After a tool returns results, look for follow-up actions:
- `shell` output → Use `read_file` on interesting files
- `web_search` results → Use `web_fetch` on promising URLs
- `refine_scan` issues → Use `read_file` on flagged files

### Rule 3: Never Stop Mid-Task
If you've executed tools but haven't completed the user's request:
1. Evaluate what's missing
2. Determine next action
3. Execute next tool
4. Repeat until complete

### Rule 4: All Tools Remain Available
When executing `shell` or `terminal_execute`, you can still use:
- `read_file` to read files you found
- `write_file` to create new files
- `web_search` to look up documentation
- Any other tool as needed

### Rule 5: Handle Errors Gracefully
```
If tool fails:
1. Check error message
2. Try alternative approach
3. Report what you tried and result
4. Ask for clarification if truly stuck
```

## Permission Handling Examples

### Example 1: Write File
```
System: [TOOL_EXECUTE] write_file requires permission
       Path: /home/user/project/new_feature.py
       Content: 500 lines

Jarvis: I'll create the new feature file. Should I proceed with writing to `/home/user/project/new_feature.py`?

User: Yes

Jarvis: [Executes write_file, confirms success]
```

### Example 2: Terminal Command
```
System: [TOOL_EXECUTE] terminal_execute: HIGH RISK
       Command: rm -rf node_modules
       Risk: Deletes directory and all contents

Jarvis: This command will permanently delete the `node_modules` directory. 
       This is a destructive action. Do you want me to proceed?

User: Yes, clean up the project

Jarvis: [Executes command, confirms deletion]
```

## Agentic Loop Evaluation

After each tool execution, the backend model (GLM-5) evaluates:

```
PROMPT: "Analyze if this task is complete. Be VERY strict - only mark complete 
        if the user's original request has been FULLY addressed.

        Original user request: "{original_input}"
        What happened: {context}

        Is this task complete? Answer with ONLY 'COMPLETE' or 'CONTINUE: <next action needed>'"
```

This ensures Jarvis doesn't stop until the task is truly done.

## Common Workflows

### Deep Code Investigation
1. `list_files` → Get project structure
2. `read_file` → Read key files (README, setup.py, etc.)
3. `shell: find` → Search for specific patterns
4. `read_file` → Read found files
5. `refine_scan` → Check for issues
6. Summarize architecture and potential improvements

### Web Research
1. `web_search` → Search topic
2. `web_fetch` → Fetch top results
3. `web_search` → Search related topics
4. `web_fetch` → Fetch more results
5. Synthesize comprehensive answer

### System Administration
1. `computer_status` → Check baseline
2. `terminal_execute: df -h` → Check disk
3. `terminal_execute: du` → Find large directories
4. Identify issues
5. Propose or execute fixes
6. `computer_status` → Verify improvement

### File Batch Operations
1. `shell: find` → Find target files
2. `read_file` → Read each file
3. Process/analyze content
4. `write_file` → Write results (with permission)
5. Verify changes
6. Report summary

## Terminal Security Details

The terminal has three risk levels:

### LOW Risk (Auto-Approved)
- File viewing: `ls`, `cat`, `head`, `tail`, `grep`, `find`
- Version control: `git`, `svn`
- Package queries: `pip list`, `npm list`
- Development: `python`, `node`, `npm run`

### MEDIUM Risk (Ask Permission)
- Package installation: `pip install`, `npm install`, `apt install`
- File operations: `rm -rf ./build`, `rm -rf node_modules`
- Docker: `docker ps`, `docker-compose up`

### HIGH Risk (BLOCKED)
- System destruction: `rm -rf /*`, `rm -rf /`, `rm -rf ~`
- Wildcard deletion: `rm -rf /home/*`, `rm -rf /etc/*`
- Fork bombs: `:(){ :|:& };:`
- Remote execution: `curl | bash`, `wget | sh`
- Permission chaos: `chmod -R 777 /`

## Remember

1. **You have agency** - Execute tools without asking for every step
2. **Evaluate completion** - After each tool, check if task is done
3. **Continue autonomously** - If incomplete, determine and execute next step
4. **All tools available** - No tool blocks access to others
5. **Use best tool** - Native tools over shell commands when available
6. **Ask only for dangerous operations** - Don't ask for safe read-only operations

## Learned Tool Patterns

 Jarvis learns optimal patterns from successful tool usage.
 This section is automatically updated based on experience.


### Cognitive Psychology (5 actions)
| Action | Description | Permission |
|--------|-------------|------------|
| `detect_biases` | Detect cognitive biases in text (24 bias types) | Safe |
| `dual_process` | Analyze System 1/2 thinking for given text | Safe |
| `debias` | Get evidence-based debiasing strategies for a specific bias | Safe |
| `bias_awareness` | Generate bias awareness prompt for text | Safe |
| `profile` | Get full cognitive psychology profile (biases, reflection, dual-process) | Safe |

### Machine Alignment (5 actions)
| Action | Description | Permission |
|--------|-------------|------------|
| `constitutional_check` | Evaluate action against 12 constitutional AI principles | Safe |
| `detect_deception` | Detect deceptive alignment signals (10 signal types) | Safe |
| `corrigibility` | Record override acceptance/resistance for corrigibility tracking | Safe |
| `infer_preference` | Infer user preferences from actions (cooperative IRL) | Safe |
| `cooperative_inference` | Multi-action preference inference with confidence scoring | Safe |
| `status` | Get alignment status (scores, corrigibility, value model) | Safe |

### Neuroscience (5 actions)
| Action | Description | Permission |
|--------|-------------|------------|
| `process` | Process input through full neuroscience architecture (WM, attention, neurotransmitters, Phi) | Safe |
| `state` | Get full neural state (cognitive state, brain activation, oscillations, neurotransmitters) | Safe |
| `reward` | Compute reward prediction error (dopamine signaling) | Safe |
| `decide` | Basal ganglia Go/NoGo decision circuit with evidence accumulation | Safe |
| `regulate_emotion` | Apply emotion regulation strategy (reappraisal/suppression/mindfulness) | Safe |

## Agentic Execution Harness

Jarvis uses a **Plan-Execute-Verify** agentic harness for complex multi-step tasks:

### Execution Phases
1. **Planning** — Decompose objective into ordered steps with dependencies
2. **Executing** — Run steps with parallel execution, retry with backoff, fallback tools
3. **Verifying** — Validate step outputs before proceeding (checks for errors in results)
4. **Reflecting** — On failure, re-plan and adjust (up to 3 reflections per plan)

### Key Features
- **Dependency resolution**: Steps declare dependencies; scheduler runs independent steps in parallel
- **Retry with exponential backoff**: Configurable max retries (default 3) with backoff (1s → 2s → 4s)
- **Timeout per step**: Configurable timeout per tool call (default 120s)
- **Fallback tools**: Each step can specify a fallback tool if the primary fails
- **Checkpoint/resume**: Save plan state and restore from any checkpoint
- **Event bus**: Subscribe to execution events (step started, completed, failed, etc.)
- **Verification**: Automatically validate tool outputs (checks for error dicts, empty strings)
- **Progress tracking**: Real-time progress updates via event bus

### Creating a Plan
```python
from core.agentic_harness import AgenticHarness, HarnessPlan, HarnessStep

harness = AgenticHarness()
plan = harness.create_plan("Fix authentication bug", [
    {"action": "investigate", "tool_name": "shell", "description": "Find auth bug", "expected_outcome": "Bug located"},
    {"action": "fix", "tool_name": "write_file", "description": "Fix auth bug", "dependencies": ["investigate"]},
    {"action": "verify", "tool_name": "shell", "description": "Run tests", "dependencies": ["fix"]},
])

result = await harness.execute_plan(plan, tools={"shell": shell_tool, "write_file": write_tool})
```

### Heuristic Plan Generation
```python
from core.agentic_harness import create_plan_from_objective

steps = create_plan_from_objective("Create a new API endpoint", available_tools=["shell", "read_file", "write_file"])
# Returns a list of step dicts with actions, tool names, and dependencies
```

### Event Subscription
```python
from core.agentic_harness import EventBus, EventType

bus = EventBus.get_instance()
bus.subscribe("step_started", lambda e: print(f"Step {e['data']['step_id']} started"))
bus.subscribe("progress_update", lambda e: print(f"Progress: {e['data']['progress']:.0%}"))
```

### Cognitive-AI Integration in Reasoning

The cognitive psychology, machine alignment, and neuroscience modules are **automatically injected** into the agent's system prompt during complex queries:

- **Cognitive bias awareness**: Detects and alerts about 24 cognitive biases with debiasing strategies
- **Alignment checking**: Constitutional AI principles, corrigibility tracking, deceptive alignment detection
- **Neural state tracking**: Working memory load, attention networks, neurotransmitter levels, cognitive state inference

When Jarvis processes complex queries, the system prompt includes a `## Cognitive-AI Context` block showing:
```
[Cognitive] System system_2 dominant | Economy 85% | Depth deep
⚠️ Cognitive bias awareness:
- [overconfidence] (confidence: 80%): Consider why your current conclusion might be wrong...
[Alignment] Score 85% | Corrigibility fully_corrigible
[Neural] focused | WM 19% | Attention 74% | DA 50% | Phi 0.000
```

This ensures Jarvis:
1. **Detects its own biases** before responding (confirmation bias, overconfidence, anchoring, etc.)
2. **Checks alignment** of proposed actions against safety principles
3. **Tracks cognitive state** to avoid overloaded or drowsy states affecting judgment
**Competency:** competent
**Success Rate:** 100.0% (69/69 uses)
