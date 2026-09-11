# Prompt Engineering Architecture for Multi-Agent Schema Mapping

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER / API REQUEST                              │
│                    "Map 'tumor_grade' to FHIR"                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SYSTEM PROMPT                                 │   │
│  │  - Task framing (entity/field/content matching)                 │   │
│  │  - Tool selection guidance                                       │   │
│  │  - Output format specification                                   │   │
│  │  - Domain constraints (FHIR R5, oncology focus)                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CLAUDE AGENT (LLM)                           │   │
│  │  - Reasoning about which tools to use                           │   │
│  │  - Synthesizing tool results                                     │   │
│  │  - Applying domain knowledge                                     │   │
│  │  - Making final mapping decision                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   EMBEDDING TOOLS    │ │   KNOWLEDGE TOOLS    │ │   TERMINOLOGY TOOLS  │
│                      │ │                      │ │                      │
│  biobert_match       │ │  rule_match          │ │  search_snomed       │
│  magneto_match       │ │  lookup_mapping      │ │  search_loinc        │
│  find_similar        │ │  explore_fhir        │ │  search_ontology     │
│                      │ │  search_fhir_fields  │ │                      │
│  [semantic vectors]  │ │  [curated rules]     │ │  [external APIs]     │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CALIBRATION LAYER                                  │
│  - Raw scores → calibrated confidence                                  │
│  - Tier-aware (entity/field/content)                                   │
│  - Context-aware backoff (schema:context → schema:* → *:*)             │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PERSISTENCE LAYER                                  │
│  - FlatMappingDatabase (curated rules)                                 │
│  - FeedbackStore (HITL decisions)                                      │
│  - VectorStore (ChromaDB embeddings)                                   │
│  - Calibrators (pkl files)                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: Prompts as Orchestration, Not Knowledge Store

The prompt's job is to **orchestrate tool use**, not to encode domain knowledge. The knowledge lives in:
- Tools (matchers, lookups)
- Databases (flat mappings, calibrators)
- The LLM itself (pre-trained FHIR/biomedical understanding)

---

## 2. Division of Labor: Knowledge Sources

### 2.1 Knowledge in Prompts (Static, Task-Framing)

**Best for:**
- Task definition and constraints
- Tool selection heuristics
- Output format requirements
- Workflow guidance

**Example from `claude_agent.py`:**
```python
FIELD_MATCHING_PROMPT = """
task: map a source field to the appropriate FHIR R5 resource field path.

THINK FIRST - before using any tools:
1. look at the FIELD NAME - what does it mean semantically?
2. look at the SAMPLE VALUES if provided - what kind of data is this?
...
"""
```

**What to encode:**
- Task decomposition strategy
- When to use which tool
- What constitutes a good answer
- Edge case handling

**What NOT to encode:**
- Specific mappings (use tools instead)
- Exhaustive domain rules (tool layer)
- Data that changes (database layer)

### 2.2 Knowledge in Tool Layer (Dynamic, Authoritative)

**Best for:**
- Curated expert mappings
- Calibrated confidence scores
- External terminology lookups
- Semantic similarity search

| Tool | Knowledge Type | Source |
|------|---------------|--------|
| `rule_match` | Expert mappings | FlatMappingDatabase (2,753 sources) |
| `biobert_match` | Biomedical semantics | BioBERT embeddings |
| `magneto_match` | Schema patterns | GDC→FHIR trained model |
| `lookup_mapping` | Direct lookups | SQLite O(1) access |
| `search_snomed` | Diagnosis codes | External API |
| `search_loinc` | Lab codes | External API |

**Advantage:** Single source of truth. Update the database, all mappings improve.

### 2.3 Knowledge the LLM Brings (Pre-trained, General)

**Claude's built-in knowledge:**
- FHIR R4/R5 resource structure
- Biomedical terminology understanding
- Oncology domain concepts
- Data modeling patterns

**How to leverage:**
```python
# DON'T: Enumerate all FHIR resources in prompt
"Patient, Specimen, Observation, Condition, Procedure..."

# DO: Trust LLM knows FHIR, guide when to explore
"If unsure which FHIR resource, use explore_fhir_resource to check fields"
```

**When LLM knowledge conflicts with tools:**
- Tools are authoritative for YOUR domain (GDC/HTAN patterns)
- LLM is authoritative for general FHIR semantics
- Prompt should establish this hierarchy

### 2.4 Few-Shot Examples (Contextual, Retrieved)

**Static few-shot (in prompt):**
```python
examples:
- "case" -> Patient (high confidence, standard clinical concept)
- "biospecimen" -> Specimen (exact fhir resource match)
```

**Good for:** Establishing output format, showing reasoning patterns

**Dynamic few-shot (retrieved):**
```python
# Use find_similar to get relevant examples
similar = vector_store.find_similar("tumor_grade", k=3)
# Inject into prompt: "Similar mappings in our KB: {similar}"
```

**Good for:** Context-specific guidance, rare patterns

---

## 3. Prompt Engineering Patterns

### 3.1 Pattern: Minimal Prompts + Rich Tools

**Philosophy:** Prompt defines WHAT to do, tools define HOW.

```python
# MINIMAL PROMPT
"""
Map {source} to FHIR. Use available tools to find the best match.
Explain your reasoning. Return confidence 0-1.
"""

# RICH TOOLS (12 tools with detailed descriptions)
- biobert_match: "Best for matching biomedical concepts..."
- rule_match: "94.7% accuracy on field matching..."
```

**Pros:**
- Easy to maintain (update tools, not prompts)
- Tool descriptions guide LLM
- Composable

**Cons:**
- LLM may not choose optimal tool order
- Less control over reasoning process

### 3.2 Pattern: Decision Tree Prompts

**Philosophy:** Encode expert workflow in prompt.

```python
"""
Step 1: Check if exact match exists
  → Use lookup_mapping first
  → If found with high confidence, STOP

Step 2: If no exact match, gather evidence
  → Use rule_match for KB patterns
  → Use biobert_match for semantic similarity
  → Use magneto_match for structural patterns

Step 3: Synthesize results
  → Prefer rule_match if score > 0.9
  → Otherwise, weighted average of all matchers
"""
```

**Pros:**
- Predictable behavior
- Optimizes tool calls
- Encodes expert heuristics

**Cons:**
- Brittle to edge cases
- Harder to maintain
- May override good LLM intuition

### 3.3 Pattern: Think-First Prompts (Current Approach)

**Philosophy:** LLM reasons first, validates with tools.

```python
FIELD_MATCHING_PROMPT = """
THINK FIRST - before using any tools:
1. look at the FIELD NAME - what does it mean semantically?
2. look at the SAMPLE VALUES if provided
3. use YOUR DOMAIN KNOWLEDGE

then use tools to VALIDATE your hypothesis
"""
```

**Pros:**
- Leverages LLM's knowledge
- Tools validate, not dictate
- Handles novel cases well

**Cons:**
- LLM may be wrong initially
- More tokens (thinking + validation)

### 3.4 Avoiding Prompt-Tool Conflicts

**Problem:** Prompt says "A maps to B" but tool returns "A maps to C"

**Solutions:**

1. **Don't encode specific mappings in prompts**
   ```python
   # BAD
   "tumor_grade typically maps to Observation.component"

   # GOOD
   "use rule_match to check existing mappings for tumor_grade"
   ```

2. **Establish authority hierarchy**
   ```python
   """
   Trust order:
   1. rule_match results (curated expert mappings)
   2. Your FHIR knowledge (general patterns)
   3. Embedding similarity (when no rules exist)
   """
   ```

3. **Prompt for conflict resolution**
   ```python
   """
   If tools disagree:
   - Prefer rule_match if confidence > 0.8
   - Explain the disagreement in your response
   - Flag for human review if uncertain
   """
   ```

---

## 4. Extracting Domain Knowledge from Code

### 4.1 Automated Knowledge Extraction

**Script to extract mappings and patterns:**

```python
# extract_knowledge.py
from schema_crush.mappings import FlatMappingDatabase
from collections import Counter

def extract_prompt_knowledge(db: FlatMappingDatabase):
    """Extract patterns for prompt engineering."""

    # 1. Most common mappings per tier
    entity_patterns = Counter()
    field_patterns = Counter()

    for src in db.sources.values():
        dests = db.lookup(src.source)
        for _, dest in dests:
            if src.tier.value == "entity":
                entity_patterns[f"{src.source} → {dest.destination}"] += 1
            elif src.tier.value == "field":
                field_patterns[f"{src.source} → {dest.destination}"] += 1

    # 2. High-confidence patterns (from calibrators)
    # 3. Common contexts
    # 4. Edge cases (low-confidence, multiple destinations)

    return {
        "entity_examples": entity_patterns.most_common(10),
        "field_examples": field_patterns.most_common(10),
    }
```

### 4.2 Knowledge Audit Checklist

When updating prompts, check:

| Source | What to Extract | Update Frequency |
|--------|----------------|------------------|
| `flat_mappings.db` | Common patterns, edge cases | Monthly |
| `calibrators/*.pkl` | Which matchers work for which tiers | After retraining |
| `FeedbackStore` | User corrections, common mistakes | Weekly |
| FHIR spec changes | New resources, deprecated fields | Per FHIR release |

### 4.3 Prompt Template with Dynamic Injection

```python
def build_prompt(tier: str, source: str, context: str = None):
    """Build prompt with dynamic knowledge injection."""

    # Static task framing
    base = PROMPTS[tier]

    # Dynamic: similar mappings from KB
    similar = vector_store.find_similar(source, k=3)
    examples = "\n".join([f"- {s['source']} → {s['target']}" for s in similar])

    # Dynamic: calibrator confidence hints
    calibrator_hint = ""
    if tier == "field":
        calibrator_hint = "rule_match is 94.7% accurate for fields"
    elif tier == "content":
        calibrator_hint = "rule_match is 99.6% accurate for content"

    return f"""
{base}

Similar mappings in knowledge base:
{examples}

Calibration note: {calibrator_hint}

Source to map: {source}
Context: {context or 'unknown'}
"""
```

---

## 5. Addressing Common Gaps

### 5.1 Orchestrated Mapping with Deliberative Mode (`map_field`)

The `map_field` tool provides end-to-end orchestrated mapping with automatic escalation to LLM reasoning when embeddings/rules are uncertain.

**Cascade Architecture:**
```
map_field(source, candidates, tier, threshold=0.7)
    │
    ▼
┌─────────────┐
│ rule_match  │ ── if conf ≥ threshold → RETURN (fast path)
└─────────────┘
    │
    ▼
┌─────────────┐
│   biobert   │ ── if conf ≥ threshold → RETURN
└─────────────┘
    │
    ▼
┌─────────────┐
│   magneto   │ ── if conf ≥ threshold → RETURN
└─────────────┘
    │
    ▼ (all matchers < threshold)
┌─────────────────────────────────────┐
│     DELIBERATIVE MODE (LLM)         │
│  reason_mapping sub-agent           │
│  - Thinks harder about edge cases   │
│  - Returns reasoning + confidence   │
└─────────────────────────────────────┘
    │
    ▼
Compare LLM vs best embedding result
    │
    ├─ LLM confidence > best_embedding → Use LLM result
    │
    └─ Otherwise → Use best embedding (note: "LLM confidence lower")
```

**Response includes full trace:**
```json
{
  "source": "tumor_grade",
  "target": "Observation.component.valueCodeableConcept",
  "confidence": 0.85,
  "matcher": "agent",
  "tier": "field",
  "escalated_to_llm": true,
  "escalation_reason": "All matchers below threshold (0.7). Best was biobert=0.62",
  "cascade": [
    {"matcher": "rule", "target": "...", "confidence": 0.45},
    {"matcher": "biobert", "target": "...", "confidence": 0.62},
    {"matcher": "magneto", "target": "...", "confidence": 0.38},
    {"matcher": "agent", "target": "...", "confidence": 0.85, "reasoning": "..."}
  ]
}
```

**Tool hierarchy:**
- **`map_field`** - Recommended for most use cases (auto-orchestrated)
- **`reason_mapping`** - Explicit LLM reasoning (manual control)
- **`biobert_match` / `magneto_match` / `rule_match`** - Individual matchers (debugging)

### 5.2 Confidence Thresholds (HITL Integration)

The system HAS explicit thresholds via calibrated confidence:

```python
# From calibration system (see orchestrator/calibration.py)
CONFIDENCE_THRESHOLDS = {
    "auto_accept": 0.95,    # High: auto-accept
    "hitl_review": 0.85,    # Medium: requires human review
    "deliberative": 0.70,   # Below this: escalate to LLM reasoning
    "reject": 0.50          # Below this: filtered out
}
```

**Escalation flow:**
```
Tool returns calibrated_score
    │
    ├─ score >= 0.95 → Auto-accept, log to FeedbackStore
    │
    ├─ 0.85 <= score < 0.95 → Route to HITL queue
    │                         (record_feedback tool)
    │
    ├─ 0.70 <= score < 0.85 → Suggest review, may be correct
    │
    └─ score < 0.70 → Escalate to deliberative mode (map_field)
                      or reject if no candidates
```

**Prompt integration:**
```python
"""
After getting tool results:
- If calibrated confidence >= 0.95: Accept and explain
- If confidence 0.85-0.95: Flag for human review
- If confidence 0.70-0.85: Present with caveats
- If confidence < 0.70: Use map_field for LLM reasoning
"""
```

### 5.2 Latency Optimization

**Fast-path pattern:**
```python
"""
TOOL ORDER (optimized for speed):

1. lookup_mapping FIRST (O(1) SQLite lookup)
   → If exact match found, STOP. Don't call expensive matchers.

2. Only if no exact match:
   → Call rule_match, biobert_match, magneto_match IN PARALLEL
   → MCP supports concurrent tool calls

3. Terminology lookups (search_snomed, search_loinc):
   → Only when mapping content values
   → External API, cache results
"""
```

**In practice (MCP server):**
- Claude can call multiple tools in one turn
- Tool results are cached per session
- KnowledgeBase singleton avoids reloading

### 5.3 Prompt Versioning (Critical)

**Implementation approach:**
```python
# In MappingProposal dataclass
@dataclass
class MappingProposal:
    source: str
    target: str
    confidence: float
    matcher: str
    prompt_version: str = "v1.0"  # ADD THIS
    timestamp: datetime = None
```

**Track in FeedbackStore:**
```sql
ALTER TABLE feedback ADD COLUMN prompt_version TEXT DEFAULT 'v1.0';
```

**Version naming:**
```
v1.0 - Initial think-first prompt
v1.1 - Added calibrator hints
v2.0 - Switched to decision-tree for field tier
```

### 5.4 Dynamic Few-Shot Specification

**Parameters:**
```python
def get_few_shot_examples(source: str, tier: str, k: int = 3):
    """
    k: Number of examples (3 is optimal - more adds noise)

    Selection criteria:
    1. Same tier (entity/field/content)
    2. Similar source name (via find_similar)
    3. Same schema if available (GDC examples for GDC source)
    4. High-confidence examples only (from accepted feedback)
    """

    similar = vector_store.find_similar(source, k=k*2)  # Over-fetch

    # Filter to same tier
    same_tier = [s for s in similar if s.get('tier') == tier]

    # Prefer same schema
    same_schema = [s for s in same_tier if s.get('schema') == current_schema]

    return (same_schema or same_tier)[:k]
```

**When context mismatch hurts:**
- GDC uses `case_id`, HTAN uses `HTAN_participant_id`
- Showing GDC examples for HTAN input can confuse
- Solution: Filter by schema, fall back to tier-only if none found

### 5.5 Pattern Commitment: Think-First as Default

**Decision:** Think-First is the DEFAULT pattern.

**Rationale:**
- Tools validate, not dictate
- Handles novel inputs (new schemas beyond GDC/HTAN)
- LLM's FHIR knowledge is strong
- 94.7% rule_match accuracy means validation usually confirms

**When to use Decision Tree instead:**
- High-volume batch processing (predictable, faster)
- Strict compliance requirements (auditable steps)
- When LLM costs are a concern (fewer reasoning tokens)

**Document this in prompt:**
```python
"""
DEFAULT APPROACH: Think-first, validate with tools.

EXCEPTION: For batch processing of 1000+ mappings,
use the decision-tree prompt (see BATCH_MAPPING_PROMPT).
"""
```

### 5.6 Pre-Dispatch Calibrator Hints

**Add to main architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SYSTEM PROMPT                                 │   │
│  │  + CALIBRATOR HINTS (injected dynamically):                     │   │
│  │    "For field tier: rule_match=94.7%, magneto=26.7%"           │   │
│  │    "For content tier: rule_match=99.6%, biobert=69.8%"         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
```

**Implementation:**
```python
CALIBRATOR_HINTS = {
    "entity": "rule_match: 52%, magneto: 4%, biobert: 1% - use multiple tools",
    "field": "rule_match: 94.7% - check rules first, high confidence",
    "content": "rule_match: 99.6% - rules are near-perfect for terminology",
}

def build_prompt(tier: str):
    return f"""
{BASE_PROMPT}

TOOL ACCURACY FOR THIS TIER ({tier}):
{CALIBRATOR_HINTS[tier]}
"""
```

---

## 6. Recommendations for Schema Crush

### Immediate Actions

1. **Keep prompts minimal** - Your tools are well-designed. Let them carry the knowledge.

2. **Add tool hints to descriptions** - Include accuracy stats:
   ```python
   "rule_match: 94.7% accuracy on fields, 99.6% on content. Use first."
   ```

3. **Inject dynamic examples** - Use `find_similar` to add relevant few-shot examples.

### Future Improvements

1. **Prompt versioning** - Track which prompt version produced which mappings.

2. **A/B testing** - Compare minimal vs decision-tree prompts on accuracy.

3. **Feedback-driven prompt updates** - Extract patterns from `FeedbackStore` corrections.

### Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Do This Instead |
|--------------|--------------|-----------------|
| Encoding all mappings in prompt | Stale, conflicts with tools | Use `lookup_mapping` tool |
| Long enumerated lists | Token waste, incomplete | Trust LLM + tool exploration |
| Rigid decision trees | Brittle, misses edge cases | Think-first + validate |
| Ignoring calibration | Wrong confidence signals | Include calibrator hints |

---

## 6. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│ AUTHORITATIVE (Tools/DB)                                        │
│   └─ Curated mappings, calibrated scores, terminology lookups  │
│                                                                 │
│ STRUCTURAL (Prompts)                                            │
│   └─ Task framing, tool selection hints, output format         │
│                                                                 │
│ GENERAL (LLM Pre-training)                                      │
│   └─ FHIR semantics, biomedical concepts, reasoning            │
│                                                                 │
│ CONTEXTUAL (Dynamic Injection)                                  │
│   └─ Similar examples, recent feedback, calibrator stats       │
└─────────────────────────────────────────────────────────────────┘
```

The goal: **Prompts orchestrate, tools inform, LLM reasons, databases persist.**