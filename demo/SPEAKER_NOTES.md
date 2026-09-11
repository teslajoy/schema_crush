# schema_crush speaker notes
## conversational guide for each section

---

## intro (2 min)

**say this:**
> "schema_crush maps biomedical data schemas to FHIR. think of it as a smart translator - you give it GDC or HTAN data, it figures out how to represent that in FHIR format. three levels of mapping: entities (tables), fields (columns), and content (actual values like 'Male' to coded terms)."

**show:** architecture diagram

**key point:** "the magic is combining rule-based lookups with ML embeddings and an LLM that reasons about uncertainty."

---

## what is langchain? (5 min)

**say this:**
> "langchain is a framework for building LLM applications. let me be precise about what it actually is in CS terms."

**show the honest definition:**
```
WHAT LANGCHAIN ACTUALLY IS
==========================

Core language: Python (also TypeScript port). No C underneath.

NOT a queue, NOT a stack, NOT a runtime.

It's just Python classes that:
1. Wrap LLM API calls (ChatAnthropic, ChatOpenAI)
2. Define a "Tool" protocol (function + JSON schema)
3. Handle message serialization (dict -> JSON -> API)

The "chain" is a METAPHOR, not a data structure.
Original meaning: compose functions like Unix pipes
    prompt1 | llm | prompt2 | llm | output

Now: mostly just means "call LLM, maybe call tools, repeat"
```

**show the actual loop - this is the core pattern:**
```
LANGCHAIN TOOL LOOP = while + dispatch + list
=============================================

messages = [system_prompt, user_query]   # list
max_iterations = 10                       # safety limit
iteration = 0

while iteration < max_iterations:
    iteration += 1
    response = llm.call(messages)        # API call
    messages.append(response)            # list append

    if no_tool_calls(response):
        break                            # done, return answer

    for tool in response.tool_calls:
        fn = tools[tool.name]            # dict lookup O(1)
        result = fn(**tool.args)         # execute locally
        messages.append(ToolMessage(result))  # list append

# if iteration == max_iterations: agent got stuck, return best effort

That's it. A while loop, a dict, a list, and a counter for safety.

WHY max_iterations?
===================
- LLM might loop forever (call same tool repeatedly)
- cost control: each iteration = API call = $$$
- timeout safety: 10 iterations × 3 sec = 30 sec max
- in practice: most tasks finish in 1-3 iterations
```

**show simple use cases:**
```
THREE EXAMPLES:
===============

1. CALCULATOR
   user: "what's 234 * 567?"
   LLM outputs: {"tool": "multiply", "args": {"a": 234, "b": 567}}
   we run it -> 132678
   LLM outputs: "the answer is 132,678"
   loop iterations: 1

2. WEATHER + OUTFIT
   user: "what should I wear in NYC today?"
   LLM outputs: {"tool": "get_weather", "args": {"city": "NYC"}}
   we run it -> "45°F, rainy"
   LLM outputs: {"tool": "get_clothing", "args": {"temp": "cold", "weather": "rain"}}
   we run it -> "jacket, umbrella"
   LLM outputs: "wear a jacket and bring an umbrella"
   loop iterations: 2

3. SCHEMA_CRUSH
   user: "map 'tumor_grade' to FHIR"
   LLM outputs: {"tool": "biobert_match", ...}
   we run it -> [{"target": "Condition.stage", "score": 0.78}, ...]
   LLM outputs: {"tool": "magneto_match", ...}
   we run it -> [{"target": "Observation.value", "score": 0.81}, ...]
   LLM outputs: "CHOSEN TARGET: Condition.stage, CONFIDENCE: 0.76"
   loop iterations: 2
```

**key point:** "langchain gives us the plumbing. while loop + dict dispatch + list of messages. that's the whole pattern."

---

## data layer (3 min)

**say this:**
> "all expert knowledge lives in FlatMappingDatabase. normalized tables - just like you'd design a relational database. sources table, destinations table, foreign keys linking them."

> "why O(1)? we build hash indexes on load. lookup('case') hits a dict, not a loop. instant."

**show:** run cells 3a-3e, show the dataframes

**key point:** "this is our ground truth - 500+ expert-curated mappings from GDC and HTAN teams."

---

## matchers (3 min)

**say this:**
> "three matchers, three strategies. rule matcher is just dict lookup - if we've seen it before, 100% confident. biobert understands medical terms - 'adenocarcinoma' and 'cancer' are similar. magneto was trained on schema matching - it knows 'case_id' patterns map to 'Patient.id' patterns."

> "we run all three and combine evidence. rule says 1.0? trust it. rule says 0.0 but biobert and magneto both say 0.9? probably good."

**show:** matcher comparison cell

**key point:** "ensemble approach - each matcher has blindspots, together they're robust."

---

## embeddings math (5 min)

**say this:**
> "you all know this, but let me show how we apply it. biobert encodes text into 768-dim vectors. similar concepts land nearby in that space."

**show the actual math:**
```
COSINE SIMILARITY FORMULA
=========================

         A · B           Σᵢ(aᵢ × bᵢ)
cos(θ) = ───────  =  ─────────────────────
        ||A|| ||B||   √(Σᵢaᵢ²) × √(Σᵢbᵢ²)

where:
  A = embedding of "case"        = [0.23, -0.15, 0.89, ..., 0.04]  (768 dims)
  B = embedding of "Patient"     = [0.21, -0.12, 0.91, ..., 0.06]  (768 dims)
  A · B = dot product            = 0.23×0.21 + (-0.15)×(-0.12) + ...
  ||A|| = L2 norm (magnitude)    = √(0.23² + 0.15² + 0.89² + ...)

GEOMETRIC INTERPRETATION
========================
         ^  B (Patient)
        /|
       / |
      /  |  cos(θ) = 0.95
     /θ  |  θ ≈ 18°
    +----+---> A (case)

  cos(0°)   = 1.0   identical meaning
  cos(90°)  = 0.0   unrelated
  cos(180°) = -1.0  opposite (rare)

WORKED EXAMPLE
==============
"case" embedding (simplified to 4D):     A = [0.8, 0.2, 0.1, 0.5]
"Patient" embedding:                     B = [0.75, 0.25, 0.15, 0.48]

A · B = 0.8×0.75 + 0.2×0.25 + 0.1×0.15 + 0.5×0.48
      = 0.60 + 0.05 + 0.015 + 0.24 = 0.905

||A|| = √(0.64 + 0.04 + 0.01 + 0.25) = √0.94 = 0.97
||B|| = √(0.5625 + 0.0625 + 0.0225 + 0.2304) = √0.878 = 0.937

cos(θ) = 0.905 / (0.97 × 0.937) = 0.905 / 0.909 = 0.996

interpretation: case and Patient are VERY similar (0.996 ≈ 1.0)
```

**say this:**
> "why cosine over euclidean? magnitude invariance. 'specimen' in a title vs 'specimen' in a paragraph - euclidean sees different lengths, cosine sees same direction."

```
WHY COSINE > EUCLIDEAN
======================
euclidean: d = √(Σᵢ(aᵢ - bᵢ)²)   <- sensitive to magnitude

problem:
  short doc "specimen"  -> [0.08, 0.02, 0.01]  (small magnitude)
  long doc "specimen"   -> [0.80, 0.20, 0.10]  (large magnitude)

  euclidean distance = 0.82  <- sees them as DIFFERENT
  cosine similarity  = 1.0   <- sees them as IDENTICAL (same direction)
```

**demo the UMAP:**
> "see how case and patient cluster together? biospecimen and specimen? that's the embedding space doing its job."

**key point:** "embeddings turn semantic similarity into geometric distance. cosine normalizes for length, compares pure direction."

---

## calibration (3 min)

**say this:**
> "raw scores aren't probabilities. biobert might say 0.72, but historically that's correct 95% of the time. calibration learns this mapping from labeled data."

> "isotonic regression - fancy name for 'fit a monotonic curve'. if 0.7 raw means 0.9 calibrated, then 0.8 raw can't mean 0.85 calibrated. has to be monotonic."

> "tier-aware: entity matching is easier (fewer options), content matching is harder. same raw score means different things per tier."

**show:** calibration table

**key point:** "calibration lets us set meaningful thresholds. 'accept if confidence > 0.9' actually means something."

---

## vector store and chromadb (3 min)

**say this:**
> "we have 500+ expert mappings in our database. when a NEW field comes in like 'tumor_grade', how do we find SIMILAR mappings to guide the agent?"

> "answer: semantic search with embeddings. we encode all source fields as vectors, store them in ChromaDB, and query for k-nearest neighbors."

**show this diagram:**
```
VECTOR STORE: SEMANTIC RETRIEVAL
================================

PROBLEM:
  new field: "tumor_grade"
  question: what similar fields have we mapped before?

SOLUTION: k-nearest neighbors in embedding space

1. INDEX TIME (startup):
   for each source in FlatMappingDB:
       embedding = model.encode(source)      # 768-dim vector
       chromadb.add(embedding, metadata)     # store with source->target info

2. QUERY TIME:
   query_emb = model.encode("tumor_grade")
   results = chromadb.query(query_emb, k=5)  # 5 nearest neighbors

   returns:
     tumor_stage -> Condition.stage     (distance: 0.09)
     grade -> Observation.value         (distance: 0.15)
     ajcc_staging -> Condition.stage    (distance: 0.18)


CHROMADB:
=========
- in-memory vector database (no external server needed)
- stores: embedding + metadata dict
- uses ANN (approximate nearest neighbor) for speed
- O(log n) query time with HNSW index
```

**say this:**
> "these similar mappings become FEW-SHOT EXAMPLES for Claude. 'here are similar mappings we've seen before: tumor_stage -> Condition.stage. now map tumor_grade.' the agent learns from patterns."

**key point:** "vector store enables learning from examples, not just raw matcher scores."

---

## what is an agent? + agent patterns (5 min)

**say this:**
> "an agent is an LLM that can take actions. not just generate text - actually DO things. call APIs, run code, query databases."

> "the key ingredients: (1) tools it can use, (2) ability to decide WHICH tool and WHEN, (3) loop until task is done."

**show this diagram:**
```
AGENT vs SIMPLE LLM
===================

simple LLM:
    input -> generate text -> output

agent:
    input -> think -> "I need more info" -> call tool ->
    get result -> think -> "need another tool" -> call tool ->
    get result -> think -> "now I can answer" -> output

the agent DECIDES the workflow at runtime
```

**now demystify "thinks" - this is important:**
```
WHAT "THINKS" ACTUALLY MEANS IN CS TERMS
========================================

There is NO thinking. NO reasoning.
It's next-token prediction that outputs JSON.

llm.call(messages) returns EITHER:
  1. plain text     -> final answer
  2. JSON object    -> tool call request


WHAT ACTUALLY HAPPENS:
======================

1. INPUT: tokens go in
   "map tumor_grade to FHIR" -> [8234, 19847, 62, 5691, ...]

2. TRANSFORMER: matrix multiplication × 96 layers
   for each layer:
       attention = softmax(Q × Kᵀ / √d) × V    # which tokens matter?
       output = FFN(attention)                  # transform

3. OUTPUT: predict next token
   logits = output × W_vocab     # score every possible token
   probs = softmax(logits)       # normalize

   probs: "{" = 0.89, "The" = 0.03, "I" = 0.02 ...
   pick "{" (highest)

4. REPEAT for each token:
   "{" -> "tool" -> ":" -> "biobert" -> "_match" -> ...

   until we get:
   {"tool": "biobert_match", "args": {...}}


THE ANTHROPOMORPHISM TRAP:
==========================

we say              CS reality
--------            ----------
"thinks"        ->  generates tokens
"reasons"       ->  generates tokens
"decides"       ->  outputs JSON vs plain text
"understands"   ->  pattern matches training data
"knows"         ->  has seen similar patterns

"Claude thinks: I need semantic similarity"
MEANS:
"next-token prediction produced tokens forming a biobert_match call"
```

**say this:**
> "the 'intelligence' is frozen in weight matrices from training. at runtime, it's pure linear algebra. no magic - just matrix math that outputs likely next tokens."

> "our ClaudeAgent is an agent. given a mapping task, it decides: should I check rules first? should I use biobert or magneto? should I explore the FHIR schema? it chains these together until confident."

**now show common patterns we use:**
```
AGENT PATTERNS IN SCHEMA_CRUSH
==============================

1. ReAct (Reason + Act)
   ---------------------
   think -> act -> observe -> think -> act -> ...

   Claude: "I need semantic similarity" -> call biobert -> see 0.78 ->
           "let me verify structure"    -> call magneto -> see 0.81 ->
           "they disagree, check few-shot" -> reason -> answer

   key: interleave thinking with tool calls


2. Fast Path / Slow Path
   ----------------------
   input -> [in cache?] --yes--> instant response (FREE)
                        --no---> LLM reasoning ($$$)

   "case" -> found in KB -> return Patient (0ms, $0)
   "tumor_grade" -> not in KB -> LLM reasons (3s, $0.02)

   key: avoid expensive compute when unnecessary


3. Ensemble + Arbitration
   -----------------------
   input -> [rule]    -\
        -> [biobert]  --> LLM weighs evidence --> answer
        -> [magneto]  -/

   multiple signals reduce error. LLM is the arbitrator.

   key: redundancy + intelligent fusion


4. Human-in-the-Loop (HITL)
   -------------------------
   agent -> [confidence < threshold?] -> PAUSE -> human reviews ->
            agent learns from feedback -> continues

   PEARL: reason -> act -> [conf < 0.8?] -> ask human -> learn

   key: know when to escalate to human
```

**say this:**
> "these patterns compose. schema_crush uses ReAct inside the LLM, fast/slow at entry, ensemble of matchers, HITL for uncertain cases. pick patterns that match your uncertainty profile."

**key point:** "agents are LLMs with agency - good patterns help them use that agency wisely."

---

## langchain tools (3 min)

**say this:**
> "@tool decorator does three things: (1) parses function signature into JSON schema, (2) uses docstring as tool description, (3) wraps function for langchain protocol."

**show what actually gets sent to Claude API:**
```
WHAT bind_tools() SENDS TO CLAUDE
=================================

our code:
    llm = ChatAnthropic(model="claude-sonnet")
    llm = llm.bind_tools(MAPPING_TOOLS)

what Claude API receives:
{
  "tools": [{
    "name": "biobert_match",
    "description": "use biobert for biomedical semantic matching...",
    "input_schema": {
      "type": "object",
      "properties": {
        "source": {"type": "string"},
        "candidates": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["source", "candidates"]
    }
  }]
}

Claude responds with tool_use:
{
  "tool_use": [{
    "name": "biobert_match",
    "input": {"source": "tumor_grade", "candidates": ["Condition.stage", "Observation.value"]}
  }]
}

WE execute the tool locally, send result back as ToolMessage
```

> "the LLM reads your docstring to decide when to call the tool. good docstrings = smarter tool selection."

**key point:** "tools are how we give the LLM capabilities. each tool is a skill. design them like APIs."

---

## claude agent (5 min)

**say this:**
> "let me walk through the full cycle. source field comes in. first we check fast path - is this in our knowledge base? if yes, skip LLM entirely. saves money and time."

> "if not, we find similar mappings for few-shot context. 'hey claude, here are similar mappings we've seen before.' then claude starts calling tools."

**show full decision trace:**
```
CLAUDE AGENT: FULL TRACE
========================
agent.propose_mappings("tumor_grade", ["Condition.stage", "Observation.value"])

STEP 1: FAST PATH
-----------------
kb.db.lookup("tumor_grade") -> []   # not found
decision: must use LLM             # fast path failed

STEP 2: FEW-SHOT CONTEXT
------------------------
kb.find_similar("tumor_grade", k=3) ->
  tumor_stage -> Condition.stage (0.91)
  ajcc_staging -> Condition.stage (0.85)
  grade -> Observation.value (0.79)

STEP 3: LLM CALL #1
-------------------
Claude thinks: "similar to tumor_stage which maps to Condition.stage.
               let me verify with biobert."

tool_call: biobert_match("tumor_grade", [...])
result: Condition.stage: 0.78, Observation.value: 0.72

STEP 4: LLM CALL #2
-------------------
Claude thinks: "biobert slightly prefers Condition.stage.
               let me check magneto for structure patterns."

tool_call: magneto_match("tumor_grade", [...])
result: Observation.value: 0.81, Condition.stage: 0.74

STEP 5: FINAL REASONING
-----------------------
Claude thinks: "matchers disagree.
  - biobert (semantic): Condition.stage (0.78)
  - magneto (structure): Observation.value (0.81)
  - few-shot context: similar terms -> Condition.stage

  the biomedical semantics + similar mappings
  outweigh magneto's structure preference."

OUTPUT:
  CHOSEN TARGET: Condition.stage
  CONFIDENCE: 0.76
  REASONING: Combined evidence from semantic matching
             and similar historical mappings.
```

> "the loop: claude calls a tool, we execute it, feed result back, claude decides next step. repeat until claude says 'I'm confident, here's my answer.'"

**demo:** run the live demo cell

**key point:** "fast path handles 93% of known mappings. LLM synthesizes multiple signals for uncertain cases."

---

## langgraph (5 min)

**say this:**
> "langgraph is for when you need more structure than a simple loop. state machines with typed state."

> "think of it like a flowchart that executes. nodes are functions. edges are transitions. you define the graph, langgraph executes it."

**show this diagram:**
```
GRAPH CONCEPTS
==============

node: a function that transforms state
      state_in -> process -> state_out

edge: transition between nodes
      node_a -> node_b

conditional edge: branching based on state
      node_a -> [if x: node_b, else: node_c]

entry point: where execution starts
END: special node that terminates

STATEGRAPH:
    workflow = StateGraph(MyState)
    workflow.add_node("step1", do_step1)
    workflow.add_node("step2", do_step2)
    workflow.add_edge("step1", "step2")
    workflow.set_entry_point("step1")
    compiled = workflow.compile()
```

> "PEARL agent uses this for multi-tier mapping. perceive schema, reason about matches, act on high-confidence, pause for human review on medium-confidence, learn from feedback, bump to next tier."

**key point:** "langgraph = state machines for LLM workflows. great for HITL and complex orchestration."

---

## live demo (5 min)

**say this:**
> "let's see it work end-to-end. I'll map 'tumor_stage' - a field that's NOT in our knowledge base. watch what claude does."

**run the demo, narrate:**
> "fast path missed - no exact match. finding similar mappings... got three examples. now claude's thinking..."
> "see? it called magneto first, then biobert. comparing scores. now it's reasoning about which field fits best."
> "final answer: Condition.stage with 0.92 confidence. the reasoning shows it combined evidence from both matchers."

**key point:** "this is the agentic approach - claude figured out HOW to solve it, not just WHAT the answer is."

---

## UMAP visualization (2 min)

**say this:**
> "let's see where our terms live in embedding space. 768 dimensions projected to 2D. clusters show semantic neighborhoods."

**show:** generated UMAP plot

> "case and patient cluster - they're both about individuals. biospecimen and specimen cluster - both about samples. file and document cluster - both about records."

**key point:** "UMAP is a debugging tool. if matches are wrong, check if concepts cluster correctly."

---

## q&a prep

**common questions:**

Q: "how much does this cost?"
A: "fast path is free. LLM calls ~$0.01-0.03 each. for 1000 fields, maybe $10-30 if all need LLM. but 93% hit fast path, so more like $1-3."

Q: "can we add our own matchers?"
A: "yes, inherit from BaseMatcher, implement match(). then add to MAPPING_TOOLS list."

Q: "what about hallucinations?"
A: "we constrain the LLM. candidate list is fixed. it picks from options we provide, doesn't invent fields. calibration catches overconfidence."

Q: "why claude vs gpt-4?"
A: "tool calling quality. claude follows instructions better for structured output. but architecture is model-agnostic."

Q: "how do we add new mappings?"
A: "add to GDC/HTAN JSON files, rebuild database. or use HITL workflow to capture new mappings from feedback."

---

## closing

**say this:**
> "schema_crush combines the best of both worlds. expert knowledge for known mappings, ML for semantic understanding, LLM for reasoning about uncertainty. the fast path keeps costs down. calibration makes confidence meaningful. HITL catches edge cases."

> "check out the tutorial notebook to run it yourself. WALKTHROUGH.md has all the details. questions?"