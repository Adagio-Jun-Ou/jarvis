# Code Review Report: Stage 1 & Stage 2 Implementation Files

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 12 |
| HIGH | 25 |
| MEDIUM | 22 |
| MISSING | 4 |

---

## CRITICAL Issues (Must Fix Immediately)

### core/live_state.py

**L176: Session Counter Logic Bug**
```python
# Line 176 - sessions_count incremented every update_from_agent call
self._data.narrative.sessions_count += 1
```
- **Issue**: Counter increments on every `update_from_agent()` call, not once per session
- **Fix**: Move counter increment to `start_session()` method

**L228-232: Race Condition in Discovery Update**
```python
# Lines 228-232 - check outside lock, assignment inside
now = time.time()
if now - self._last_discovery_update < self._discovery_update_interval:
    return
self._last_discovery_update = now
```
- **Issue**: Time check happens outside lock, race condition between check and update
- **Fix**: Move entire check inside the lock

---

### core/context_enrichment.py

**L428: Missing Import - NameError**
```python
# Line 428 - MemoryType not imported
recent = um.get_recent(MemoryType.INSIGHT, limit=limit)
```
- **Issue**: `MemoryType` is used but never imported
- **Fix**: Add `from core.unified_memory import MemoryType` at top

**L252-253: Race Condition in Cache Access**
```python
# Lines 247-253 - cache accessed outside lock
if use_cache:
    with _cache_lock:
        if time.time() - _last_enrichment_time < _cache_ttl:
            return _enrichment_cache.copy()
```
- **Issue**: Cache copy returned after lock release, but TTL check was inside lock
- **Fix**: Return copy while holding lock

---

### core/emotional_state.py

**L42: Enum Member Name Mismatch**
```python
# Line 42 - URGE should be URGENT
URGE = "urgent"
```
- **Issue**: `URGE` enum member name doesn't match the pattern (other moods use full words like FRUSTRATED, CURIOUS)
- **Fix**: Rename to `URGENT = "urgent"`

---

### core/user_model.py

**L359-371: Data Corruption from Truncation**
```python
# Line 359 - preference key truncated to 30 chars
key = f"{pref_type}_{match[:30]}"
# ...
self._data.preferences[key].confidence = min(1.0, ...)
```
- **Issue**: If two different preferences hash to same truncated key, confidence increases but value is wrong
- **Fix**: Use full key or hash for uniqueness, store full value separately

---

### core/goal_tracker.py

**L236-238: Similarity Check Too Loose**
```python
# Lines 236-238 - substring matching could merge different goals
if desc_lower in goal.description.lower() or goal.description.lower() in desc_lower:
    return goal
```
- **Issue**: Goals like "build a website" and "build a car" would incorrectly merge due to "build" substring
- **Fix**: Use word-level overlap or embedding similarity

---

### core/proactive.py

**L107-109: Urgent Users Never Get Help**
```python
# Lines 107-109
if user_mood in ["frustrated", "angry", "urgent"]:
    return False, None
```
- **Issue**: Users in urgent need are specifically excluded from proactive help
- **Fix**: For "urgent", return `True, "immediate_help"` instead

**L103-105: First Suggestion Delayed**
```python
# Line 103
if time.time() - self._last_suggestion_time < min_suggestion_interval:
```
- **Issue**: `_last_suggestion_time` initialized to 0.0, so first check waits 2 minutes from Unix epoch
- **Fix**: Initialize to `time.time()` in `__init__`

---

### core/self_reflection.py

**L129-132: Schema Versioning Missing**
```python
# Lines 129-132
self._mistakes = [
    Mistake(**m) for m in data.get("mistakes", [])
]
```
- **Issue**: If `Mistake` dataclass fields change, old saved data won't load correctly
- **Fix**: Add schema version field and migration logic

---

### core/personality.py

**L86-94: Silent Trait Loss**
```python
# Lines 86-94
try:
    trait_type = TraitType(t_data["name"])
    traits[name] = PersonalityTrait(...)
except (ValueError, KeyError):
    continue  # Silent skip
```
- **Issue**: Invalid trait values silently skipped, losing user's personality data
- **Fix**: Log warning and use default value

---

## HIGH Priority Issues

### core/live_state.py
- **L372**: `thinking_about` truncated to 100 chars without word boundary check
- **L237-243**: Import failures logged at DEBUG level should be WARNING

### core/context_enrichment.py
- **L568-572**: Silent exception swallowing hides failures from callers
- **L680-692**: `unified_memory_search` doesn't isolate failures per system

### core/background_bridge.py
- **L546-554**: Sentence splitting with `.split(". ")` breaks on abbreviations
- **L197**: Timestamp check outside lock

### core/narrative_thread.py
- **L217**: Session ID collision possible with MD5 of timestamp
- **L348**: Disk I/O inside lock blocks other threads

### core/emotional_state.py
- **L263-276**: Partial state update on exception leaves inconsistent state
- **L195-198**: No bounds on `MAX_HISTORY` enforcement during updates

### core/user_model.py
- **L404-415**: Emoji regex doesn't cover all Unicode emoji ranges
- **L356-371**: Preference deduplication missing

### core/goal_tracker.py
- **L404-418**: `_check_condition_met` only checks "completed" status
- **L281-288**: Duplicate tasks not prevented

### core/proactive.py
- **L293-300**: Potential memory growth in topic/file tracking
- **L302-306**: Dismiss counter not used to prevent re-suggestions

### core/self_reflection.py
- **L227-242**: Satisfaction signals list unbounded
- **L314-334**: Wrong turn index used for mistake identification
- **L469-485**: Confusing negation logic in quality tracking

### core/personality.py
- **L228-241**: Disk I/O inside lock
- **L357-378**: Unbounded adaptation values

---

## MEDIUM Priority Issues

### core/live_state.py
- **L480-485**: Singleton pattern could be improved with module-level initialization
- **L328-331**: Inefficient list operations for `recent_progress`

### core/context_enrichment.py
- **L366-438**: All debug-level logging doesn't differentiate failure types
- **L498-501**: Relevance calculation is simplistic

### core/background_bridge.py
- **L85-183**: Silent exception swallowing should WARN at minimum

### core/narrative_thread.py
- **L276-282**: Topic extraction includes noise words
- **L371-388**: Empty milestones edge case not handled

### core/emotional_state.py
- **L113-159**: Regex should be precompiled
- **L353-385**: Fixed word thresholds don't adapt to language

### core/user_model.py
- **L476-481**: Interest sorting ties not handled
- **L650**: Interest prediction incentivizes spam

### core/goal_tracker.py
- **L158-159**: Hardcoded file extensions
- **L371-402**: Simplistic suggestion logic

### core/proactive.py
- **L183-204**: Hardcoded break threshold
- **L325-328**: Counter import inside method

### core/self_reflection.py
- **L273-275**: Counter import inside method
- **L243**: Score calculation should average, not sum
- **L412-417**: Timezone not considered

### core/personality.py
- **L164-170**: Inconsistent mood adjustments
- **L306-333**: Context validation missing

---

## MISSING Features (Imported but Not Connected)

### 1. personality.py - Never Used
- **Location**: `core/agent.py:46`
- **Issue**: `get_personality_system()` imported but never called
- **Impact**: Personality traits don't influence responses
- **Fix**: Call in `get_system_prompt()` to add personality hints

### 2. proactive.py - Suggestions Never Generated
- **Location**: `core/agent.py:1165`
- **Issue**: `proactive_engine.update_activity()` called but `should_suggest()` never called
- **Impact**: Proactive suggestions never surface to user
- **Fix**: Check for suggestion in `run_step()` and append to response

### 3. self_reflection.py - Insights Never Used for Context
- **Location**: `core/agent.py:1239`
- **Issue**: Reflection recorded but `get_reflection_context()` never called
- **Impact**: Agent doesn't improve based on past mistakes
- **Fix**: Add reflection context to `get_system_prompt()`

### 4. All Context in build_rich_context Not Used
- **Location**: `core/agent.py:760-787`
- **Issue**: `build_rich_context()` imported but only live_state and narrative added separately
- **Impact**: User model, goals, proactive context all missing from prompts
- **Fix**: Replace manual context assembly with `build_rich_context()` call

---

## Integration Gaps in core/agent.py

### L363-386: Silent Initialization Failures
```python
# If any subsystem fails to import, entire live_state set to None
except Exception as e:
    logger.warning(f"Failed to initialize live state: {e}")
    self._live_state = None
```
- **Issue**: Failure kills all context features silently
- **Fix**: Continue with partial initialization

### L1139-1178: Context Updates Not Utilized
- Emotional state analyzed but not used for response generation
- User model updated but not queried for personalization
- Goals extracted but never cross-referenced with suggestions

### L771-787: Manual Context Assembly
```python
# Lines 771-787 - manual assembly instead of build_rich_context
self._live_state.update_from_agent(self)
self._live_state.update_from_subsystems()
self._live_state.update_discoveries()
live_context = self._live_state.get_context_string()
# ... manual additions
```
- **Issue**: Duplicates logic in `build_rich_context()` which includes all modules
- **Fix**: Use `build_rich_context(self, user_input)` instead

---

## Recommended Fix Priority

1. **Immediate (Today)**: Fix NameError in context_enrichment.py L428
2. **This Week**: Fix race conditions and session counter bugs
3. **Next Sprint**: Connect missing integrations (personality, proactive, reflection)
4. **Ongoing**: Address MEDIUM issues during refactoring