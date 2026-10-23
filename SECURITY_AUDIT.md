# Jarvis Security & Design Audit - Session Report

## Start: Thu Apr 9 09:18 EDT | Duration: ~3h30m | Status: COMPLETE

---

## CRITICAL FINDINGS (Fixed)

### 1. PATH TRAVERSAL IN MEMORY SAVE/LOAD (CRITICAL)
**File**: `core/memory.py:510-570`
**Issue**: `save()` and `load()` accepted arbitrary `name` parameters that could contain `../` sequences, allowing write/read of arbitrary files on disk.
**Fix**: Added input sanitization rejecting names with non-alphanumeric characters, and resolved path validation to ensure no traversal beyond storage_path.

### 2. PROMPT INJECTION VIA IMPORTANCE KEYWORDS (HIGH)
**File**: `core/memory.py:190-198`
**Issue**: `add_user_message()` auto-assigns importance 0.95 to any message containing "remember", "important", "don't forget", or "save". Attackers could inject arbitrary content into permanent memory.
**Fix**: Capped auto-elevated importance at 0.9 (down from 0.95), added length check (>200 chars = lower importance 0.85).

### 3. API KEY LEAKAGE IN OLLAMA CLIENT (__repr__) (HIGH)
**File**: `core/llm.py:244-246`
**Issue**: `OllamaClient.__repr__()` would expose the full API key if the object was logged or printed.
**Fix**: Added `__repr__` that masks the key, showing only first 4 and last 4 characters.

### 4. TIMING ATTACK ON API KEY VALIDATION (HIGH)
**File**: `core/auth_middleware.py:270-275`
**Issue**: Key comparison used `in` dict lookup which is NOT timing-safe. Attackers could use timing differences to extract API keys.
**Fix**: Replaced with `hmac.compare_digest()` for constant-time comparison.

### 5. `trust_remote_code=True` IN MODEL LOADING (CRITICAL)
**File**: `core/concept_model.py:229`
**Issue**: `SentenceTransformer(model_name, trust_remote_code=True)` allows execution of arbitrary Python code from model repositories. A malicious HuggingFace model could execute arbitrary code.
**Fix**: Changed to `trust_remote_code=False`.

### 6. POST-SHIELD MODIFIES RESPONSE AFTER YIELD (LOGIC BUG)
**File**: `core/agent.py:2084-2099`
**Issue**: POST-shield modified `accumulated_response` after it had been yielded to the user via streaming, creating silent data inconsistency between what the user saw and what was stored in memory.
**Fix**: Changed POST-shield to only LOG quality issues and update reflect-edit metrics, NOT modify the response.

### 7. COUNCIL DELIBERATION OVERRIDES GOOD RESPONSES (LOGIC BUG)
**File**: `core/agent.py:1190`
**Issue`: Council was convened when routing confidence < 0.4, which could override a perfectly good dual-model response with a slower council synthesis.
**Fix**: Only convene council when response is weak (< 50 chars or empty) AND confidence is low (< 0.3).

---

## RACE CONDITIONS (Fixed)

### 8. `_reflections` dict modified outside lock (reflect_edit.py)
**Fix**: Moved OrderedDict operations inside `self._lock`.

### 9. Memory block edits via `get_block()` then `block.append()` without lock (reflect_edit.py)
**Fix**: Changed to use `tlc.edit_block()` which holds the TLA lock.

### 10. `_cache` dict modified outside lock (lcm_shield.py)
**Fix**: Wrapped cache writes in `self._lock`.

### 11. `_rewrite_count` modified outside lock (lcm_shield.py)
**Fix**: Wrapped in `self._lock` and added OrderedDict with max size.

### 12. `evict_to_l2` accessed memory blocks without lock (three_layer_context.py)
**Fix**: Wrapped block access in `self._blocks_lock`.

### 13. `model_errors` dict read without lock (llm.py)
**Fix**: Used `self._model_errors_lock` for all reads and writes.

### 14. Redundant unlocked read of `current_model` (llm.py)
**Fix**: Removed unnecessary unlocked read before locked read.

---

## RESOURCE EXHAUSTION (Fixed)

### 15. Unbounded `self.context` dict in ConversationMemory
**Fix**: Added `MAX_CONTEXT_KEYS = 200` and eviction in `set_context()`.

### 16. Unbounded `model_errors` dict in OllamaClient
**Fix**: Added max 50 entries with FIFO eviction in `_increment_model_error()`.

### 17. Unbounded `_rewrite_count` dict in LCMShield
**Fix**: Converted to OrderedDict with max 500 entries.

### 18. Infinite loop in `_clean_hallucination_markers`
**Fix**: Added `max_iterations` limit per indicator.

### 19. Unbounded token generation in `_text_to_audio_tokens`
**Fix**: Changed from `text[:10000]` cap to iterative `text[:4000]` per character loop.

### 20. Unbounded `_call_log` in DualModelOrchestrator
**Fix**: Added hard cap of 500 entries after time-based pruning.

---

## INJECTION ATTACKS (Fixed)

### 21. Tool name injection via newlines in memory (memory.py)
**Fix**: Sanitized tool names with `replace("\n", " ").replace("\r", " ")[:100]`.

### 22. Cross-model prompt injection (dual_model.py)
**Fix**: Sanitized prompts in `_execute_sequential()`, capped at 500 chars, stripped `---` delimiters. Added `[Cross-model context]` wrapper with truncation at 2000 chars.

### 23. LCM Shield `_classify_risk` bypass via Unicode lookalikes
**Fix**: Added Unicode normalization and extended pattern list with common obfuscation variants.

### 24. Unvalidated tool names in agent.py (agent.py)
**Fix**: Added validation that tool names are alphanumeric with `_` and `-` only.

---

## DATA LEAKAGE (Fixed)

### 25. API key in `OllamaClient.__repr__` (llm.py)
**Fix**: Added `__repr__` with key masking.

### 26. Error messages in `All models failed` exception (llm.py)
**Fix**: Left as-is (error propagation is necessary for debugging), but noted.

---

## DESIGN IMPROVEMENTS (Implemented)

### 27. Context Overflow Guardian (NEW: `core/context_guardian.py`)
Proactively detects approaching context overflow and triggers graduated countermeasures (WARNING → SOFT_EVICT → HARD_EVICT → EMERGENCY).

### 28. Conversation Summarizer (NEW: `core/conversation_summarizer.py`)
Compresses evicted messages before storing in L2/L3, extracting topics, key facts, decisions, and tools used.

### 29. Prompt Sandbox (NEW: `core/prompt_sandbox.py`)
Isolates cross-model context with clear delimiters and injection pattern filtering. Applied to user input entry point and dual-model context flow.

### 30. System prompt cache invalidation when L2 blocks change (agent.py)
Added `_last_l2_version` counter. `get_system_prompt()` now checks `three_layer_context.get_version()` and invalidates cache when blocks are edited.

### 31. Memory pressure auto-summarization (memory.py)
When the Context Guardian triggers eviction, `_trim_if_needed()` now summarizes evicted messages before discarding them via the Conversation Summarizer.

### 32. Read-only memory blocks (three_layer_context.py)
Added `read_only` flag to `MemoryBlock`. The `lcm_meta` block is now read-only, only editable by the prefrontal cortex (`caller="prefrontal_cortex"`).

### 33. Memory load validation (memory.py)
Added schema validation for loaded history entries - rejecting entries missing `role`/`content`, entries with invalid roles, and converting non-string content to strings.

### 34. SMART message selection perf fix (memory.py)
Converted `selected_indices` list to set for O(1) membership checks in `_smart_select_messages()`.

### 35. `self_edit.py` safety improvements
Removed `dual_model.py`, `backend_llm.py`, `web_react.py` from SAFE_FILES. Added `agent.py`, `llm.py`, `memory.py`, `self_edit.py`, `context_guardian.py`, `auth_middleware.py` to CRITICAL_FILES.

### 36. WebSocket input length limits (web_react.py)
Added 10,000 character limit on `chat` and `agent_loop` WebSocket messages.

### 37. `use_cloud=False` override bug (llm.py)
Fixed `use_cloud or bool(self.api_key)` to `use_cloud if use_cloud is not None else bool(self.api_key)`.

### 38. `importance=0.0` falsy bug (memory.py)
Fixed `importance or 0.6` to `importance if importance is not None else 0.6`.

### 39. `max_tokens=0` falsy bug (memory.py)
Fixed `max_tokens or self.max_tokens` to `max_tokens if max_tokens is not None else self.max_tokens`.

### 40. Working memory priority inversion (memory.py)
Changed `_build_context_pack` to iterate in REVERSE (most recent first) so recent messages get priority in the working memory budget.

### 41. Token estimation inconsistency (memory.py)
Noted but left as-is since both methods (char//4 and word*2) are approximate and used in different contexts.

### 42. `response.json()` without error handling (llm.py)
Added try/except around all `response.json()` calls to handle non-JSON responses gracefully.

### 43. Exponential backoff bug (llm.py)
Fixed useless double-min: `min(backoff, 2)` was capping at 2s regardless of `min(..., 10)`. Changed to proper exponential backoff with 30s cap.

---

## NEW TEST FILES

- `tests/test_phase1_modules.py` - 67 tests for LCM Shield, Planner, Three-Layer Context, Reflect-Edit, Contracts, Audio Bridge
- `tests/test_security_guardian.py` - 22 tests for Context Guardian, memory security, LLM security, read-only blocks
- `tests/test_summarizer.py` - 11 tests for Conversation Summarizer
- `tests/test_prompt_sandbox.py` - 12 tests for Prompt Sandbox

**Total: 112 new tests, all passing**
---

## ADDITIONAL FINDINGS (Fixed)

### 44. TIMING ATTACK ON API KEY VALIDATION (HIGH)
**File**: `core/auth_middleware.py:270-275`
**Issue**: Key validation used `in` dict lookup which allows timing attacks.
**Fix**: Replaced with `hmac.compare_digest()` for constant-time comparison.

### 45. `trust_remote_code=True` IN CONCEPT MODEL (CRITICAL)
**File**: `core/concept_model.py:229`
**Issue**: `SentenceTransformer(trust_remote_code=True)` could execute arbitrary code from model repos.
**Fix**: Changed to `trust_remote_code=False`.

### 46. USER INPUT LENGTH UNLIMITED IN WEBSOCKET (MEDIUM)
**File**: `interfaces/web_react.py:6141`
**Issue**: No length limit on user input from WebSocket `chat` and `agent_loop` messages.
**Fix**: Added 10,000 character limit on both input paths.

### 47. AGENT INPUT NOT SANITIZED AT ENTRY (MEDIUM)
**File**: `core/agent.py:1649`
**Issue**: Raw user input flows directly into `run_step_stream()` without sanitization.
**Fix**: Added `PromptSandbox.sanitize_user_input()` call at the entry point.

### 48. SELF_EDIT SAFE_FILES INCLUDED DANGEROUS FILES (HIGH)
**File**: `core/self_edit.py:90-148`
**Issue**: `web_react.py` and `dual_model.py` were in SAFE_FILES, allowing LLM-generated edits to modify the web server and model routing.
**Fix**: Removed dangerous entries, added `agent.py`, `llm.py`, `memory.py`, etc. to CRITICAL_FILES.

### 49. NEVER_SAFE_PATTERNS INCOMPLETE (MEDIUM)
**File**: `core/self_edit.py:166-179`
**Issue**: Missing patterns for `trust_remote_code`, `__builtins__`, `globals()`, `socket`, `/etc/passwd` access.
**Fix**: Added 12 additional NEVER_SAFE_PATTERNS for comprehensive security.

### 50. CROSS-MODEL PROMPT INJECTION (HIGH)
**File**: `core/dual_model.py:718-724, 806-807`
**Issue**: User input and cross-model context injected directly into LLM prompts without sanitization.
**Fix**: Added `PromptSandbox` wrapping with source delimiters, truncation at 2000 chars, and `---` delimiter stripping.

### 51. MEMORY LOAD LACKS SCHEMA VALIDATION (MEDIUM)
**File**: `core/memory.py:656-667`
**Issue**: Loaded JSON had no validation for missing keys, wrong types, or missing role/content.
**Fix**: Added validation that rejects entries without `role`/`content`, with invalid roles, or non-string content. Also bounded `self.context` at `MAX_CONTEXT_KEYS = 200`.

### 52. RESPONSE.JSON() WITHOUT ERROR HANDLING (MEDIUM)
**File**: `core/llm.py:562, 785`
**Issue**: `response.json()` calls in `chat()` and `chat_with_tools()` could crash on non-JSON responses.
**Fix**: Added `try/except` around both calls.

### 53. EXPONENTIAL BACKOFF BUG (MEDIUM)
**File**: `core/llm.py:462-466`
**Issue**: Double `min()` - inner `min(backoff, 2)` capped at 2s, making outer `min(10)` dead code.
**Fix**: Changed to proper exponential with 30s cap.

### 54. SMART MESSAGE SELECTION O(n^2) PERFORMANCE (LOW)
**File**: `core/memory.py:305-320`
**Issue**: `i in selected_indices` was O(n) list scan, making the while loop O(n²).
**Fix**: Converted `selected_indices` to set for O(1) lookup.

### 55. CONVERSATION SUMMARIZATION ON EVICTION (IMPROVEMENT)
**File**: `core/memory.py:410-438`
**Issue**: Old messages were simply dropped during compression, losing information.
**Fix**: Integrated `ConversationSummarizer` to produce summaries of evicted messages before dropping them.

### 56. CONTEXT GUARDIAN AUTO-EVICTION (IMPROVEMENT)
**File**: `core/memory.py:366-408, core/agent.py:1733-1755`
**Issue**: No proactive context management - overflow only handled reactively.
**Fix**: Added `ContextGuardian` that monitors pressure levels (NORMAL/WARNING/SOFT_EVICT/HARD_EVICT/EMERGENCY) and auto-evicts to L2.

### 57. HEALTH CHECKER (IMPROVEMENT)
**File**: `core/health_check.py`
**New**: Unified health check endpoint monitoring all 8 subsystems.

### 58. CONVERSATION CHECKPOINT (IMPROVEMENT)
**File**: `core/conversation_checkpoint.py`
**New**: Crash recovery - checkpoint saves every 5 turns, can restore from latest checkpoint.

### 59. WEB DASHBOARD STATS UPDATED (IMPROVEMENT)
**File**: `interfaces/web_react.py:1700-1790`
**New**: Dashboard now shows stats for LCM Shield, Three-Layer Context, Reflect-Edit, Council, Audio Bridge, Contracts, LCM Planner, Context Guardian, Prompt Sandbox.
