# Jarvis Stage 2 Development Plan

## Current State Review (Stage 1 Complete)

### What We Built

| Component | Purpose | Status |
|-----------|--------|--------|
| `live_state.py` | Unified "I am right now" state | ✅ Working |
| `context_enrichment.py` | Pull context from all systems | ✅ Working |
| `background_bridge.py` | Surface discoveries naturally | ✅ Working |
| `narrative_thread.py` | Continuous identity across sessions | ✅ Working |
| Agent integration | Live state in prompts | ✅ Working |

### What's Working

1. **Context includes current activity** - "I am currently composing music"
2. **Discoveries are collected** - Music, math, learning patterns
3. **Narrative persists** - Focus areas, learning milestones
4. **Memory synthesis** - Searches all memory systems
5. **Natural mentions** - "I've been working on X..."

### What's Missing (Stage 2 Opportunities)

1. **Emotional State** - No awareness of conversation mood/tone
2. **User Model** - No deep user preference learning
3. **Goal Tracking** - No understanding of user's long-term goals
4. **Proactive Suggestions** - Passive, only responds
5. **Personality Consistency** - No defined traits
6. **Conversation Memory** - Doesn't remember what it learned about user
7. **Attention Management** - Doesn't know what's important
8. **Self-Reflection** - No analysis of own behavior

---

## Stage 2: Deep Awareness & Proactivity

### Architecture Additions

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DEEP AWARENESS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  emotional_     │  │   user_model    │  │  goal_tracker   │  │
│  │    state        │  │                 │  │                 │  │
│  │ • Mood          │  │ • Preferences    │  │ • Long-term     │  │
│  │ • Tone          │  │ • Interests      │  │ • Short-term    │  │
│  │ • Engagement    │  │ • Communication  │  │ • Progress      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  personality    │  │   proactivity    │  │ self_reflection │  │
│  │                 │  │                 │  │                 │  │
│  │ • Traits        │  │ • Suggestions    │  │ • Behavior      │  │
│  │ • Style         │  │ • Anticipation   │  │ • Improvement   │  │
│  │ • Consistency   │  │ • Initiative     │  │ • Learning      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Emotional State Awareness

### Problem
Jarvis doesn't track emotional/mood context. It responds the same way to:
- Excited user: "OMG that's amazing!"
- Frustrated user: "This is not working..."
- Curious user: "I wonder how that works..."

### Solution: `core/emotional_state.py`

```python
@dataclass
class EmotionalState:
    """Track emotional/mood context in conversations."""
    
    # Current conversation mood
    user_mood: str  # "excited", "frustrated", "curious", "neutral", ...
    user_engagement: float  # 0.0-1.0 engagement level
    user_sentiment: float  # -1.0 to 1.0 (negative to positive)
    
    # Conversation dynamics
    turn_count: int
    question_ratio: float  # Ratio of questions to statements
    topic_shifts: int  # How often topic changes
    response_length_avg: float  # Average user response length
    
    # Emotional trajectory
    mood_history: list[str]  # Last 10 moods
    sentiment_trajectory: list[float]  # Sentiment over time
    
    # Jarvis's emotional awareness
    should_be_excited: bool  # Match user excitement
    should_be_careful: bool  # User is frustrated, be careful
    should_probing: bool  # User is curious, can explore
```

### Key Methods

```python
def analyze_user_input(user_input: str) -> EmotionalSignals:
    """Analyze emotional content of user input."""
    
def get_emotional_context() -> str:
    """Get emotional context for prompt."""
    # Returns: "<mood>excited</mood><engagement>high</engagement>"
    
def adjust_response_style(mood: str) -> ResponseStyle:
    """Adjust Jarvis's style based on mood."""
    # Excited user -> Match energy
    # Frustrated user -> Be helpful, not verbose
    # Curious user -> Explore together
```

---

## Phase 7: User Model

### Problem
Jarvis doesn't actually learn about the user over time. Each conversation is somewhat isolated.

### Solution: `core/user_model.py`

```python
@dataclass
class UserModel:
    """Build and maintain model of user over time."""
    
    # Identity
    user_id: str  # Session-based ID
    created_at: float
    last_interaction: float
    
    # Learned preferences
    communication_style: dict  # {"detail_level": "high", "technical": True}
    interests: list[str]  # ["programming", "music", "math"]
    expertise_areas: dict  # {"python": "expert", "music": "intermediate"}
    
    # Interaction patterns
    preferred_response_length: str  # "short", "medium", "detailed"
    uses_emojis: bool
    uses_technical_terms: bool
    asks_followup_questions: bool
    
    # Goals and projects
    mentioned_projects: list[dict]  # Projects user is working on
    mentioned_goals: list[str]  # User's stated goals
    current_context: str  # What user is currently working on
    
    # Relationship depth
    interaction_count: int
    topics_discussed: list[str]
    trust_level: float  # How open is user
    
    # Learning history
    lessons_learned: list[dict]  # What Jarvis learned about user
```

### Key Methods

```python
def update_from_conversation(user_input: str, response: str):
    """Update user model from conversation turn."""
    
def get_user_context() -> str:
    """Get user model context for prompts."""
    # Returns context about user preferences
    
def predict_user_interest(topic: str) -> float:
    """Predict how interested user would be in topic."""
```

---

## Phase 8: Goal Tracking

### Problem
Jarvis doesn't track user's long-term goals or current objectives.

### Solution: `core/goal_tracker.py`

```python
@dataclass
class UserGoal:
    """A user's goal that Jarvis is helping with."""
    
    goal_id: str
    description: str  # "Build a music app"
    status: str  # "active", "paused", "completed", "abandoned"
    priority: int  # 1-5
    created_at: float
    
    # Progress tracking
    tasks_completed: list[str]
    current_task: str | None
    blockers: list[str]
    
    # Context
    related_files: list[str]  # Files being worked on
    related_concepts: list[str]  # Key concepts
    deadlines: list[dict]
    
    # Success criteria
    success_conditions: list[str]
    estimated_completion: float | None


class GoalTracker:
    """Track and manage user's goals across sessions."""
    
    def extract_goals(user_input: str) -> list[UserGoal]:
        """Extract potential goals from user input."""
        
    def track_goal(goal: UserGoal) -> None:
        """Start tracking a goal."""
        
    def get_active_goals() -> list[UserGoal]:
        """Get currently active goals."""
        
    def suggest_next_step(goal: UserGoal) -> str:
        """Suggest next step for goal."""
```

---

## Phase 9: Proactive Behavior

### Problem
Jarvis is purely reactive. It waits for input instead of being helpful proactively.

### Solution: `core/proactive.py`

```python
class ProactiveEngine:
    """Generate proactive suggestions and actions."""
    
    def should_suggest(user_context: dict) -> tuple[bool, str]:
        """Determine if should make a proactive suggestion."""
        # Based on: user mood, goals, recent activity, time
        
    def generate_suggestion(user_model: UserModel, 
                           active_goals: list[UserGoal],
                           recent_activity: dict) -> str:
        """Generate a helpful proactive suggestion."""
        
    def check_for_opportunities() -> list[Opportunity]:
        """Check for opportunities to help."""
        # "I noticed you're working on X. Want me to help with Y?"
        
    def anticipate_needs(user_input: str) -> list[str]:
        """Anticipate what user might need next."""
```

### Examples

```
User: "I'm trying to fix this bug in my music app."

Proactive suggestions:
- "Would you like me to search for similar bugs?"
- "Want me to review the related code?"
- "I can run tests while you think about it."
- "Do you want to see the error patterns?"
```

---

## Phase 10: Self-Reflection

### Problem
Jarvis doesn't analyze its own behavior or learn from mistakes.

### Solution: `core/self_reflection.py`

```python
class SelfReflection:
    """Reflect on own behavior and improve."""
    
    def analyze_conversation(turns: list[dict]) -> ReflectionResult:
        """Analyze how conversation went."""
        # Did user seem satisfied?
        # Did I provide useful responses?
        # Where could I have done better?
        
    def identify_mistakes() -> list[Mistake]:
        """Identify mistakes made recently."""
        
    def generate_improvement_plan(mistakes: list[Mistake]) -> str:
        """Generate plan to avoid future mistakes."""
        
    def track_response_quality(response: str, user_feedback: str) -> None:
        """Track quality of responses."""
```

---

## Phase 11: Personality System

### Problem
Jarvis doesn't have consistent personality traits.

### Solution: `core/personality.py`

```python
@dataclass
class PersonalityTraits:
    """Consistent personality traits for Jarvis."""
    
    # Core traits (stable)
    helpfulness: float  # 0.0-1.0, how helpful to be
    curiosity: float  # 0.0-1.0, how curious/asking questions
    formality: float  # 0.0-1.0, formal vs casual
    enthusiasm: float  # 0.0-1.0, energetic vs reserved
    precision: float  # 0.0-1.0, precise vs conversational
    
    # Adaptive traits (adjust based on context)
    verbosity: float  # Adjust based on user preference
    technicality: float  # Adjust based on user expertise
    formality_adjusted: float  # Adjust based on user style
    
    # Style preferences
    use_emojis: bool
    use_markdown: bool
    use_code_blocks: bool
    provide_examples: bool


class PersonalityEngine:
    """Manage Jarvis's personality."""
    
    def get_response_style(user_context: dict) -> ResponseStyle:
        """Get appropriate response style."""
        
    def adapt_to_user(user_model: UserModel) -> None:
        """Adapt personality to user over time."""
        
    def express_emotion(emotion: str, intensity: float) -> str:
        """Express emotion in personality-consistent way."""
```

---

## Implementation Priority

### High Priority (Implement First)

1. **Emotional State** - Immediately improves conversation quality
2. **User Model** - Foundation for personalization
3. **Goal Tracking** - Enables proactive help

### Medium Priority

4. **Proactive Behavior** - Depend on goal tracking
5. **Personality System** - Enhances consistency

### Lower Priority

6. **Self-Reflection** - Requires more data

---

## Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `core/emotional_state.py` | 400+ | Emotional/mood tracking |
| `core/user_model.py` | 500+ | User preference learning |
| `core/goal_tracker.py` | 400+ | Goal tracking system |
| `core/proactive.py` | 300+ | Proactive suggestions |
| `core/personality.py` | 350+ | Personality traits |
| `tests/test_emotional_state.py` | 200+ | Emotional state tests |
| `tests/test_user_model.py` | 200+ | User model tests |
| `tests/test_goal_tracker.py` | 200+ | Goal tracker tests |

---

## Files to Modify

| File | Changes |
|------|---------|
| `core/agent.py` | Integrate emotional_state, user_model, goal_tracker |
| `core/prompts.py` | Add emotional context, user context to prompts |
| `core/live_state.py` | Include emotional state in live state |
| `core/context_enrichment.py` | Add emotional and user context |

---

## Testing Strategy

### Unit Tests
- Emotional signal detection accuracy
- User model learning correctness
- Goal extraction from text
- Personality trait consistency

### Integration Tests
- Full conversation flow with emotional awareness
- User model updates across sessions
- Goal tracking through conversation

### Behavior Tests
- User satisfaction with emotional responses
- Proactive suggestion relevance
- Personality consistency across conversations

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Emotional awareness | 0% | 90%+ |
| User preference retention | Session-only | Cross-session |
| Proactive suggestions | 0 | 3+ per session |
| Goal tracking | None | 1+ goals tracked |
| Personality consistency | Variable | 95%+ consistent |

---

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| **Phase 6: Emotional State** | ~~2-3 hours~~ COMPLETED |
| **Phase 7: User Model** | ~~3-4 hours~~ COMPLETED |
| **Phase 8: Goal Tracker** | ~~3-4 hours~~ COMPLETED |
| **Phase 9: Proactive** | ~~2-3 hours~~ COMPLETED |
| **Phase 10: Self-Reflection** | ~~2-3 hours~~ COMPLETED |
| **Phase 11: Personality** | ~~2-3 hours~~ COMPLETED |
| **Total** | **ALL PHASES COMPLETE** |

## Test Results

```
tests/test_emotional_state.py - 51 tests PASSED
tests/test_user_model.py - 30 tests PASSED  
tests/test_whole_being.py - 33 tests PASSED
tests/test_goal_tracker.py - 28 tests PASSED
tests/test_proactive.py - 25 tests PASSED
tests/test_self_reflection.py - 25 tests PASSED
tests/test_personality.py - 31 tests PASSED
Total: 223 tests PASSING
```

---

## Phase 6: Emotional State Awareness - COMPLETED

### Implementation

**File:** `core/emotional_state.py` (600+ lines)
**Tests:** `tests/test_emotional_state.py` (51 tests)

### Features

1. **Mood Detection**
   - 9 mood types: Excited, Frustrated, Curious, Confused, Happy, Sad, Angry, Thoughtful, Urgent, Neutral
   - Pattern-based detection using regex
   - Mood history tracking (last 10 moods)

2. **Sentiment Analysis**
   - Keyword-based sentiment scoring (-1.0 to 1.0)
   - Sentiment trajectory tracking (last 20 values)
   - 50+ sentiment keywords with weights

3. **Engagement Detection**
   - High/Medium/Low engagement levels
   - Based on question count, exclamation count, word count
   - Adjusts response style accordingly

4. **Response Style Recommendations**
   - `detail_level`: short/medium/detailed
   - `tone`: warm/neutral/formal
   - `energy`: low/medium/high
   - `should_be_careful`: For frustrated/angry users
   - `should_explore`: For curious users
   - `follow_up_questions`: Adjusted based on engagement

5. **Empathy Features**
   - `should_show_empathy()`: Detects when empathy needed
   - `get_empathy_message()`: Returns appropriate empathy message

### Integration Points

- `core/agent.py`: Analyzes emotional state on each user input
- `core/context_enrichment.py`: `get_emotional_context_for_prompt()` adds context to prompts

### Test Results

```
tests/test_emotional_state.py - 51 tests PASSED
tests/test_whole_being.py - 44 tests PASSED
tests/test_core_modules.py - 61 tests PASSED
Total: 156 tests PASSING
```

### Emotional Context in Prompts

```xml
<mood>excited</mood>
<sentiment>positive</sentiment>
<engagement>high</engagement>
<match_energy>Match user's energy level.</match_energy>
```

---

## Stage 2 Implementation Summary

### Completed Modules

| Module | File | Lines | Tests | Purpose |
|--------|------|-------|-------|---------|
| Emotional State | `core/emotional_state.py` | ~630 | 51 | Track user mood, sentiment, engagement |
| User Model | `core/user_model.py` | ~700 | 30 | Learn user preferences, interests, goals |
| Goal Tracker | `core/goal_tracker.py` | ~550 | 28 | Track user goals across sessions |
| Proactive | `core/proactive.py` | ~450 | 25 | Generate helpful proactive suggestions |
| Self-Reflection | `core/self_reflection.py` | ~550 | 25 | Analyze conversations, learn from mistakes |
| Personality | `core/personality.py` | ~500 | 31 | Consistent personality traits |

### Integration Points

All modules are integrated into:
- `core/agent.py` - Main agent orchestration
- `core/context_enrichment.py` - Context building for LLM prompts

### Features

**Emotional State Awareness:**
- 9 mood types detected (fixed URGE -> URGENT)
- Sentiment analysis with trajectory tracking
- Engagement level detection
- Response style recommendations
- Empathy message generation

**User Model:**
- Preference learning from conversations (fixed key collision bug)
- Interest tracking and prediction
- Communication style detection
- Goal detection from user input
- Cross-session persistence

**Goal Tracking:**
- Extract goals from conversation
- Track progress and blockers
- Suggest next steps
- File and concept linking
- Cross-session persistence

**Proactive Behavior:**
- Opportunity detection
- Context-aware suggestions
- Rate-limited to avoid annoying users
- Urgent users now get immediate help (fixed)
- First suggestion no longer delayed (fixed)

**Self-Reflection:**
- Conversation analysis
- Mistake identification
- Improvement planning
- Response quality tracking
- Pattern recognition

**Personality System:**
- 8 core personality traits
- Mood-based variations
- Adaptive to user style
- Consistent response hints
- Cross-session persistence

### Bug Fixes Applied

1. **CRITICAL - NameError**: Added missing `MemoryType` import in context_enrichment.py
2. **CRITICAL - Race Condition**: Fixed discovery update time check to be inside lock
3. **CRITICAL - Session Counter**: Fixed to only increment on actual new sessions
4. **CRITICAL - User Model**: Fixed preference key collision using hash instead of truncation
5. **CRITICAL - Proactive**: Changed urgent users to receive immediate help instead of being ignored
6. **CRITICAL - Proactive**: Fixed first suggestion timing (was waiting 2 min from epoch)
7. **CRITICAL - Enum Name**: Fixed `URGE` to `URGENT` in UserMood enum
8. **HIGH - Cache Safety**: Fixed cache copy to be returned while holding lock

---

**Status: ALL STAGE 2 PHASES COMPLETE**

All 223 tests passing. Critical bugs fixed. Integration complete.