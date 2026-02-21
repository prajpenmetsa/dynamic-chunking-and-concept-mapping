"""
LLM-as-Judge Evaluation Framework for Learning Objectives
Evaluates LOs against ABCD, SMART, and Bloom's Taxonomy criteria
with detailed rubrics and consistency checking
"""

import json
import os
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
import requests
from collections import defaultdict
import statistics

# ==================== CONFIGURATION ====================
load_dotenv()

# Judge Configuration
# Using dual judges for methodological rigor:
# 1. Gemini 2.0 Flash (proprietary) - primary judge
# 2. Llama 3.3 70B via Groq (open-source) - validation judge

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Evaluation settings
RATE_LIMIT_DELAY = 3  # Gemini and Groq both have generous free tiers

# Paths
ABCD_INPUT = "../../datasets/slide_based_los_simple_abcd.json"
SMART_INPUT = "../../datasets/slide_based_los_simple_smart.json"
BLOOMS_INPUT = "../../datasets/slide_based_los_simple_blooms.json"
OUTPUT_DIR = "../../datasets/evaluation"

# Evaluation settings
NUM_EVALUATION_RUNS = 3  # Run each evaluation multiple times for consistency


# ==================== API HELPERS ====================

def call_gemini_api(prompt: str, system_prompt: str = None, temperature: float = 0.3) -> Dict:
    """Call Gemini 2.0 Flash API with JSON response parsing."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file!")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Gemini uses different structure for system prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json"
        }
    }
    
    url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from Gemini response structure
            if "candidates" in result and len(result["candidates"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            else:
                raise Exception("No candidates in Gemini response")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"   ⚠️  Rate limit (Gemini). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    
    raise Exception("Max retries exceeded (Gemini)")


def call_groq_api(prompt: str, system_prompt: str = None, temperature: float = 0.3) -> Dict:
    """Call Llama 3.3 70B via Groq API with JSON response parsing."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return json.loads(text)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"   ⚠️  Rate limit (Groq). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    
    raise Exception("Max retries exceeded (Groq)")


def call_judge_api(prompt: str, system_prompt: str = None, temperature: float = 0.3, judge: str = "gemini") -> Dict:
    """Call Ollama Cloud API with JSON response parsing."""
    """Unified API caller that routes to appropriate judge."""
    if judge == "gemini":
        return call_gemini_api(prompt, system_prompt, temperature)
    elif judge == "groq":
        return call_groq_api(prompt, system_prompt, temperature)
    else:
        raise ValueError(f"Unknown judge: {judge}. Must be 'gemini' or 'groq'")


# ==================== RUBRIC DEFINITIONS ====================

ABCD_RUBRIC = """
## ABCD Framework Rubric (1-5 Scale)

**EVALUATION LENS**: Focus on STRUCTURE and COMPLETENESS of the learning objective statement.
ABCD evaluates whether all four components are present and well-specified, NOT the cognitive level.

**HARD CONSTRAINTS**:
- If the LO uses "understand", "know", "learn", or "appreciate" WITHOUT clarifying HOW it will be demonstrated → Behavior score ≤ 2
- If no explicit or strong implicit audience → Audience score ≤ 3
- If no performance standard (explicit or implicit) → Degree score ≤ 2

### A - Audience (Who will learn?)
**5 - Excellent**: Explicitly states the learner (e.g., "Students will...", "Learners will..."). Crystal clear who is learning.
**4 - Good**: Audience is implicit but unambiguous from context (e.g., course syllabus context makes "Analyze..." clearly for students).
**3 - Adequate**: Audience can be inferred but requires assumptions. Some ambiguity exists.
**2 - Poor**: Audience is unclear or could apply to multiple groups (students, teachers, developers).
**1 - Unacceptable**: No identifiable audience. Generic statement that doesn't specify who learns.

### B - Behavior (What will they DO?) - FOCUS: Observable demonstration, not cognitive level
**5 - Excellent**: Uses precise, observable, measurable action verb (analyze, design, implement, evaluate, compare, construct). Behavior is concrete and demonstrable.
**4 - Good**: Uses clear action verb from Bloom's taxonomy mid-levels (apply, explain, demonstrate, classify). Observable and testable.
**3 - Adequate**: Uses weaker action verb that's harder to observe (describe, discuss, summarize). Measurability is questionable.
**2 - Poor**: Uses vague verb (understand, know, learn) without clarifying demonstration method. Hard to observe/measure.
**1 - Unacceptable**: No action verb, or uses purely passive verbs (be exposed to, appreciate, become aware of).

### C - Condition (Under what circumstances?)
**5 - Excellent**: Explicitly states conditions/context (e.g., "After completing the project...", "Using synchronization primitives...", "Given a scheduling scenario...").
**4 - Good**: Conditions are implicit but unambiguous from content (e.g., "Implement a scheduler" clearly implies "using a programming environment").
**3 - Adequate**: Some contextual hints exist but conditions are vague. Unclear when/how behavior occurs.
**2 - Poor**: Missing meaningful conditions. No context for when learning is demonstrated.
**1 - Unacceptable**: Completely absent. No indication of circumstances or context.

### D - Degree (How well?)
**5 - Excellent**: Specific quantified performance standard (e.g., "with 90% accuracy", "identifying at least 3 differences", "handling all 5 edge cases").
**4 - Good**: Implicit but clear standard (e.g., "correctly", "accurately", "all major components"). Reasonably unambiguous.
**3 - Adequate**: Weak implicit standard (e.g., "effectively", "appropriately"). Some ambiguity about success.
**2 - Poor**: Vague standard (e.g., "well", "properly") or minimal criterion. Success criteria unclear.
**1 - Unacceptable**: No standard specified. Impossible to determine what constitutes achievement.

**BINARY CHECKLIST** (Answer YES/NO for each, then assign scores):
1. [ ] Can you identify WHO is learning without making assumptions?
2. [ ] Does the verb describe an OBSERVABLE action (not a mental state)?
3. [ ] Are the CONDITIONS/CONTEXT stated or unambiguously implied?
4. [ ] Is there a STANDARD (explicit or clear implicit) that defines success?
5. [ ] Could you create an exam question/rubric to assess this objectively?
"""

SMART_RUBRIC = """
## SMART Framework Rubric (1-5 Scale)

**EVALUATION LENS**: Focus on CLARITY, ASSESSABILITY, and PRACTICAL FEASIBILITY.
SMART evaluates whether the objective is actionable and useful for course planning, NOT just structural completeness.
DISTINCTION FROM ABCD: While ABCD asks "is behavior observable?", SMART asks "can we measure SUCCESS with concrete criteria?"

**HARD CONSTRAINTS**:
- If the LO uses "understand", "know", "appreciate" → Measurable score ≤ 2 (cannot assess objectively)
- If content is generic ("OS concepts", "programming skills") → Specific score ≤ 2
- If achievement would require >1 semester for target students → Achievable score ≤ 2
- If topic is peripheral to course core → Relevant score ≤ 3

### S - Specific (Is it focused and clear?)
**5 - Excellent**: Names exact concepts, algorithms, or skills (e.g., "Implement the Banker's algorithm", "Compare FCFS vs SJF scheduling"). No ambiguity about WHAT.
**4 - Good**: Specific domain with clear boundaries (e.g., "Analyze deadlock prevention strategies"). Scope is well-defined.
**3 - Adequate**: Somewhat specific but includes vague terms (e.g., "Understand synchronization mechanisms"). Scope is fuzzy.
**2 - Poor**: Very broad or generic (e.g., "Learn about OS", "Study memory"). Unclear what exactly is covered.
**1 - Unacceptable**: Completely vague (e.g., "Gain knowledge", "Explore concepts"). No discernible focus.

### M - Measurable (Can achievement be assessed with concrete criteria?) - DISTINCT FROM ABCD BEHAVIOR
**NOTE**: ABCD Behavior asks "is it observable?"; SMART Measurable asks "can we define SUCCESS criteria?"
**5 - Excellent**: Explicit success metrics (e.g., "correctly implement 3 algorithms", "identify all primitives", "with <5% overhead").
**4 - Good**: Uses verbs that enable rubric-based assessment (analyze, design, implement, compare). Clear grading criteria possible.
**3 - Adequate**: Can be measured but criteria are fuzzy (explain, describe, discuss). Grading would be somewhat subjective.
**2 - Poor**: Difficult to measure objectively. Uses weak verbs (understand, know) or no clear assessment method.
**1 - Unacceptable**: Cannot be measured meaningfully. Purely subjective outcomes or intangible goals.

### A - Achievable (Is it realistic for the target learners?)
**5 - Excellent**: Perfectly scoped for course level. Neither trivial nor impossible. Matches expected cognitive level.
**4 - Good**: Realistic but might be slightly challenging/easy. Generally appropriate for course.
**3 - Adequate**: Questionable difficulty level. Might be too advanced or too basic for stated course.
**2 - Poor**: Likely unrealistic. Too complex or too simple given course context.
**1 - Unacceptable**: Clearly impossible or insulting. Requires prerequisites not in course or is trivially obvious.

### R - Relevant (Does it align with course goals?)
**5 - Excellent**: Directly tied to core course concepts mentioned in course description. Essential learning.
**4 - Good**: Relevant to course domain. Supports main learning goals.
**3 - Adequate**: Tangentially related. Could be relevant but connection is weak.
**2 - Poor**: Questionable relevance. Peripheral topic that's not core to course.
**1 - Unacceptable**: Completely irrelevant. Doesn't belong in this course.

### T - Time-bound (Is timeframe implicit or explicit?)
**5 - Excellent**: Explicit timeframe (e.g., "by end of course", "after module 3", "by week 8"). Clear deadline.
**4 - Good**: Timeframe implicit but unambiguous (e.g., end-of-course expectation is standard for course LOs).
**3 - Adequate**: Timeframe vague or requires inference. Could be interpreted multiple ways.
**2 - Poor**: No clear timeframe. Unclear when achievement is expected.
**1 - Unacceptable**: Timeframe makes no sense, contradicts course structure, or is completely absent.

**BINARY CHECKLIST** (Answer YES/NO for each, then assign scores):
1. [ ] If you gave this to a student, would they know EXACTLY what topic/skill to learn?
2. [ ] Can you write a specific exam question or design a graded assignment for this?
3. [ ] Is this achievable within one semester for students at the stated course level?
4. [ ] Does this topic appear in the course description or align with stated course goals?
5. [ ] Is there a clear (explicit or standard implicit) timeframe for achievement?
"""

BLOOMS_RUBRIC = """
## Bloom's Taxonomy Rubric (1-5 Scale)

**EVALUATION LENS**: Focus on COGNITIVE LEVEL ACCURACY and PEDAGOGICAL PROGRESSION.
Bloom's evaluates whether the verb matches the actual thinking required, NOT just structural quality.

**HARD CONSTRAINTS**:
- If verb is "understand", "know", "learn" → Must be classified as Remember/Understand (low level), NOT higher
- If task requires creating something new but verb is "implement"/"apply" → Flag as misclassified
- If verb says "analyze" but task is just recall/application → Cognitive Demand score ≤ 2

### INDIVIDUAL LO CRITERIA (evaluate each LO separately)

### Overall Taxonomy Alignment
**5 - Excellent**: Uses precise Bloom's verb from correct level. Cognitive demand matches verb perfectly. No ambiguity about intended level.
**4 - Good**: Uses appropriate Bloom's verb. Minor mismatch between verb and cognitive complexity (off by one level). Generally aligned.
**3 - Adequate**: Uses Bloom's-adjacent verb but complexity doesn't match stated level (e.g., says "analyze" but task is "apply").
**2 - Poor**: Uses weak verb (understand, know) or verb significantly mismismatches task complexity (off by 2+ levels).
**1 - Unacceptable**: No Bloom's verb or completely wrong level (e.g., uses "list" for a creation task).

### Level-Specific Criteria

#### Remember (Recall facts, terms, basic concepts)
**Appropriate verbs**: Define, List, Recall, Identify, Label, Name, State, Recognize
**5**: Pure recall with no interpretation required. Clear factual knowledge.
**3**: Mix of recall and understanding. Not pure memorization.
**1**: Requires higher-order thinking, not just recall.

#### Understand (Explain ideas, concepts; summarize; interpret)
**Appropriate verbs**: Explain, Describe, Summarize, Paraphrase, Interpret, Classify, Compare
**5**: Requires explanation in own words or interpretation. Goes beyond recall.
**3**: Borders on recall or application. Not clearly understanding.
**1**: Either pure recall or requires analysis/application.

#### Apply (Use knowledge in new situations; implement procedures)
**Appropriate verbs**: Apply, Implement, Execute, Use, Demonstrate, Solve, Compute
**5**: Clear application of concept/procedure to new scenario. Execution focus.
**3**: Borders on understanding or analysis. Application unclear.
**1**: No clear application; either recall or higher-order analysis.

#### Analyze (Break down, examine relationships, distinguish parts)
**Appropriate verbs**: Analyze, Examine, Compare, Contrast, Differentiate, Distinguish, Investigate
**5**: Requires breaking down into components, finding relationships, examining structure.
**3**: More like application or evaluation. Analysis unclear.
**1**: Lower-level task (recall/understand) mislabeled as analysis.

#### Evaluate (Judge, critique, assess based on criteria)
**Appropriate verbs**: Evaluate, Assess, Critique, Judge, Justify, Argue, Defend
**5**: Requires making judgments, defending positions, critiquing based on standards.
**3**: Borders on analysis or creation. Evaluation unclear.
**1**: No clear evaluative judgment required.

#### Create (Design, construct, develop something new)
**Appropriate verbs**: Design, Develop, Create, Construct, Formulate, Plan, Compose
**5**: Requires producing something original/new. Synthesis of multiple concepts.
**3**: More like apply or evaluate. Not truly creating something new.
**1**: Lower-level task mislabeled as create.

**BINARY CHECKLIST FOR INDIVIDUAL LOs** (Answer YES/NO, then assign scores):
1. [ ] Does the verb accurately represent the cognitive level (e.g., "analyze" for Analyze level)?
2. [ ] Does the actual TASK complexity match what the verb implies?
3. [ ] Can you classify this into a SINGLE Bloom's level without debating?
4. [ ] If the verb is "understand" or "know", is it scored as Remember/Understand (not higher)?

---

### SET-LEVEL CRITERIA (evaluate the complete set of LOs together)
**NOTE**: These criteria ONLY apply when evaluating ALL learning objectives together, NOT individual LOs.

### Progression & Hierarchy (SET-LEVEL ONLY)
**5 - Excellent**: Clear progression from Remember/Understand → Apply → Analyze/Evaluate/Create. Logical scaffolding.
**4 - Good**: Mostly progressive with 1-2 minor sequencing issues. Generally builds up.
**3 - Adequate**: Some progression but noticeable gaps or jumps (e.g., jumps from Remember to Create without Apply/Analyze).
**2 - Poor**: Little progression. Mostly random ordering of cognitive levels.
**1 - Unacceptable**: No progression (all same level) or entirely reversed (starts with Create, ends with Remember).

**SET-LEVEL CHECKLIST**:
1. [ ] Do LOs progress from lower-order (Remember/Understand) to higher-order thinking (Analyze/Create)?
2. [ ] Are lower-level outcomes logical prerequisites for higher-level ones?
3. [ ] Is there reasonable coverage across multiple Bloom's levels (not all Remember or all Create)?
4. [ ] Do later LOs build on concepts introduced in earlier ones?
"""

# ==================== GRANULAR EVALUATION QUESTIONS ====================

ABCD_QUESTIONS = [
    {
        "criterion": "Audience",
        "question": "Is the intended learner explicitly or clearly implicitly stated (e.g., 'Students will...', 'Learners will...', or context makes audience obvious)?",
        "scale": "1=No audience identifiable, 3=Implicit but clear, 5=Explicitly stated"
    },
    {
        "criterion": "Behavior",
        "question": "Does the LO use an observable, measurable action verb (not vague terms like 'understand' or 'know')?",
        "scale": "1=Non-behavioral/passive, 3=Vague action verb, 5=Clear measurable verb"
    },
    {
        "criterion": "Condition",
        "question": "Are the circumstances, context, or conditions under which the behavior will be demonstrated stated or strongly implied?",
        "scale": "1=No context, 3=Vague conditions, 5=Explicit conditions"
    },
    {
        "criterion": "Degree",
        "question": "Is there a performance standard or criterion that defines how well the behavior must be performed?",
        "scale": "1=No standard, 3=Implicit standard, 5=Explicit measurable standard"
    }
]

SMART_QUESTIONS = [
    {
        "criterion": "Specific",
        "question": "Does the LO clearly identify WHAT specific concept, skill, or knowledge will be learned (not generic terms)?",
        "scale": "1=Completely vague, 3=Somewhat specific, 5=Highly specific with exact topics"
    },
    {
        "criterion": "Measurable",
        "question": "Can achievement of this LO be objectively assessed through exams, projects, or assignments?",
        "scale": "1=Cannot be measured, 3=Difficult to measure, 5=Easily measurable with clear criteria"
    },
    {
        "criterion": "Achievable",
        "question": "Is this LO realistic and attainable for students in this course within one semester?",
        "scale": "1=Impossible/trivial, 3=Questionable difficulty, 5=Perfectly scoped"
    },
    {
        "criterion": "Relevant",
        "question": "Is this LO directly tied to core concepts of 'Advanced Operating Systems' course?",
        "scale": "1=Irrelevant, 3=Tangentially related, 5=Core course topic"
    },
    {
        "criterion": "Time-bound",
        "question": "Is there a clear (explicit or implicit) timeframe for when this learning outcome should be achieved?",
        "scale": "1=No timeframe, 3=Vague timeframe, 5=Clear explicit timeframe"
    }
]

BLOOMS_QUESTIONS = [
    {
        "criterion": "Verb_Accuracy",
        "question": "Does the action verb accurately represent the cognitive level according to Bloom's Taxonomy (e.g., 'Analyze' for analysis, not 'understand')?",
        "scale": "1=Wrong/no Bloom's verb, 3=Approximate match (off by one level), 5=Perfect Bloom's verb for task"
    },
    {
        "criterion": "Cognitive_Demand",
        "question": "Does the actual task complexity match the cognitive level implied by the verb? (e.g., if verb is 'analyze', does task require breaking down/comparing, not just applying?)",
        "scale": "1=Complete mismatch (2+ levels off), 3=Partial match (1 level off), 5=Perfect match"
    },
    {
        "criterion": "Level_Classification",
        "question": "Can you unambiguously classify this LO into a single Bloom's level without debate?",
        "scale": "1=Impossible to classify, 3=Ambiguous between 2 adjacent levels, 5=Clearly one level"
    }
]


# ==================== EVALUATION PROMPTS ====================

def create_abcd_evaluation_prompt(learning_objective: str, run_number: int) -> tuple:
    """Create system prompt (rubric) and user prompt (LO to evaluate) for ABCD framework."""
    system_prompt = f"""You are an expert educational assessment specialist. You evaluate learning objectives against the ABCD framework with strict adherence to the rubric.

{ABCD_RUBRIC}

**YOUR ROLE**: Apply this rubric consistently. Use the binary checklist first, then assign scores. Provide specific evidence from the LO for every score."""
    
    user_prompt = f"""**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVE TO EVALUATE**:
"{learning_objective}"

**INSTRUCTIONS**:
1. First, answer the 5 binary checklist questions (YES/NO)
2. Then, for EACH component (A, B, C, D), assign a score from 1-5 based on the rubric
3. Apply HARD CONSTRAINTS (check if LO uses 'understand'/'know' without demonstration → Behavior ≤2)
4. Provide specific evidence from the LO that justifies your score
5. Identify what's missing or weak

**GRANULAR QUESTIONS** (answer these too):
{json.dumps(ABCD_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
{{
{{
  "overall_scores": {{
    "audience": {{"score": <1-5>, "evidence": "specific quote or observation", "weakness": "what's missing or weak"}},
    "behavior": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "condition": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "degree": {{"score": <1-5>, "evidence": "...", "weakness": "..."}}
  }},
  "granular_responses": [
    {{
      "criterion": "Audience",
      "question": "Is the intended learner explicitly or clearly implicitly stated?",
      "score": <1-5>,
      "justification": "specific reasoning"
    }},
    ...
  ],
  "composite_score": <average of 4 component scores>,
  "overall_assessment": "Brief 2-3 sentence summary of strengths and weaknesses",
  "improvement_suggestions": ["specific suggestion 1", "suggestion 2"]
}}

Return ONLY valid JSON. Be rigorous and objective."""


def create_smart_evaluation_prompt(learning_objective: str, course_context: str, run_number: int) -> tuple:
    """Create system prompt (rubric) and user prompt (LO to evaluate) for SMART framework."""
    system_prompt = f"""You are an expert educational assessment specialist. You evaluate learning objectives against the SMART framework with strict adherence to the rubric.

{SMART_RUBRIC}

**YOUR ROLE**: Apply this rubric consistently. Use the binary checklist first, then assign scores. Remember: SMART Measurable is DISTINCT from ABCD Behavior - focus on assessment criteria, not just observability."""
    
    user_prompt = f"""**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVE TO EVALUATE**:
"{learning_objective}"

**COURSE CONTEXT**:
- Course: Advanced Operating Systems (CS3.304)
- Level: Senior undergraduate / Graduate  
- Focus: {course_context}

**INSTRUCTIONS**:
1. First, answer the 5 binary checklist questions (YES/NO)
2. Then, for EACH component (S, M, A, R, T), assign a score from 1-5 based on the rubric
3. Apply HARD CONSTRAINTS (check if uses 'understand'/'know' → Measurable ≤2; generic content → Specific ≤2)
4. Consider course context for Achievable and Relevant scores
5. Provide specific evidence and justification

**GRANULAR QUESTIONS** (answer these too):
{json.dumps(SMART_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
{{
  "binary_checklist": [
    {{"question": "Student knows EXACTLY what to learn?", "answer": "YES/NO", "reasoning": "..."}},
    {{"question": "Can write specific exam question?", "answer": "YES/NO", "reasoning": "..."}},
    {{"question": "Achievable in one semester?", "answer": "YES/NO", "reasoning": "..."}},
    {{"question": "Aligns with course goals?", "answer": "YES/NO", "reasoning": "..."}},
    {{"question": "Clear timeframe?", "answer": "YES/NO", "reasoning": "..."}}
  ],
  "hard_constraints_applied": {{"uses_weak_verb": false, "too_generic": false, "unrealistic_scope": false}},
  "overall_scores": {{
    "specific": {{"score": <1-5>, "evidence": "specific quote or observation", "weakness": "what's missing"}},
    "measurable": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "achievable": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "relevant": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "time_bound": {{"score": <1-5>, "evidence": "...", "weakness": "..."}}
  }},
  "granular_responses": [
    {{
      "criterion": "Specific",
      "question": "Does the LO clearly identify WHAT specific concept, skill, or knowledge will be learned?",
      "score": <1-5>,
      "justification": "specific reasoning"
    }},
    ...
  ],
  "composite_score": <average of 5 component scores>,
  "overall_assessment": "Brief 2-3 sentence summary",
  "improvement_suggestions": ["specific suggestion 1", "suggestion 2"]
}}

Return ONLY valid JSON. Be rigorous and objective."""
    
    return system_prompt, user_prompt


def create_blooms_evaluation_prompt(all_objectives: List[str], run_number: int) -> tuple:
    """Create system prompt (rubric) and user prompt (LOs to evaluate) for Bloom's Taxonomy."""
    objectives_text = "\n".join([f"{i+1}. {lo}" for i, lo in enumerate(all_objectives)])
    
    system_prompt = f"""You are an expert educational assessment specialist. You evaluate learning objectives against Bloom's Taxonomy with strict adherence to the rubric.

{BLOOMS_RUBRIC}

**YOUR ROLE**: Apply this rubric consistently. Evaluate INDIVIDUAL LOs first (verb accuracy, cognitive demand, classification), then evaluate the SET (progression, coverage). Apply hard constraints rigorously."""
    
    user_prompt = f"""**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVES TO EVALUATE**:
{objectives_text}

**INSTRUCTIONS**:
1. For EACH learning objective (INDIVIDUAL evaluation):
   - Answer the 4 binary checklist questions
   - Identify the Bloom's level (Remember/Understand/Apply/Analyze/Evaluate/Create)
   - Apply HARD CONSTRAINTS (if 'understand'/'know' → must be Remember/Understand level)
   - Score: Verb Accuracy, Cognitive Demand, Level Classification

2. For the COMPLETE SET (SET-LEVEL evaluation):
   - Evaluate progression from lower → higher levels
   - Check prerequisite relationships
   - Assess coverage across levels

**GRANULAR QUESTIONS** (answer for EACH individual LO):
{json.dumps(BLOOMS_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
{{
  "individual_evaluations": [
    {{
      "objective_number": 1,
      "objective_text": "...",
      "identified_level": "Apply/Analyze/etc",
      "binary_checklist": [
        {{"question": "Verb matches cognitive level?", "answer": "YES/NO", "reasoning": "..."}},
        {{"question": "Task complexity matches verb?", "answer": "YES/NO", "reasoning": "..."}},
        {{"question": "Can classify into single level?", "answer": "YES/NO", "reasoning": "..."}},
        {{"question": "If 'understand', scored as low level?", "answer": "YES/NO/NA", "reasoning": "..."}}
      ],
      "hard_constraints_applied": {{"weak_verb_detected": false, "verb_task_mismatch": false}},
      "scores": {{
        "verb_accuracy": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
        "cognitive_demand": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
        "level_classification": {{"score": <1-5>, "evidence": "...", "weakness": "..."}}
      }},
      "granular_responses": [...],
      "composite_score": <average score>,
      "improvement_suggestions": ["...", "..."]
    }},
    ...
  ],
  "set_evaluation": {{
    "progression_score": {{"score": <1-5>, "evidence": "...", "weakness": "..."}},
    "level_distribution": {{"Remember": 0, "Understand": 2, "Apply": 2, "Analyze": 1, "Evaluate": 1, "Create": 1}},
    "overall_assessment": "2-3 sentence summary of the complete set",
    "set_level_composite_score": <score>
  }},
  "overall_composite_score": <average of all individual scores>,
  "recommendations": ["recommendation 1", "recommendation 2"]
}}

Return ONLY valid JSON. Be rigorous and objective."""
    
    return system_prompt, user_prompt


# ==================== EVALUATION EXECUTION ====================

def evaluate_abcd_learning_objective(lo: str, run_number: int, judge: str = "gemini") -> Dict:
    """Evaluate a single LO against ABCD framework."""
    system_prompt, user_prompt = create_abcd_evaluation_prompt(lo, run_number)
    result = call_judge_api(user_prompt, system_prompt=system_prompt, temperature=0.3, judge=judge)
    result["judge"] = judge
    time.sleep(RATE_LIMIT_DELAY)
    return result


def evaluate_smart_learning_objective(lo: str, course_context: str, run_number: int, judge: str = "gemini") -> Dict:
    """Evaluate a single LO against SMART framework."""
    system_prompt, user_prompt = create_smart_evaluation_prompt(lo, course_context, run_number)
    result = call_judge_api(user_prompt, system_prompt=system_prompt, temperature=0.3, judge=judge)
    result["judge"] = judge
    time.sleep(RATE_LIMIT_DELAY)
    return result


def evaluate_blooms_set(objectives: List[str], run_number: int, judge: str = "gemini") -> Dict:
    """Evaluate complete set of LOs against Bloom's Taxonomy."""
    system_prompt, user_prompt = create_blooms_evaluation_prompt(objectives, run_number)
    result = call_judge_api(user_prompt, system_prompt=system_prompt, temperature=0.3, judge=judge)
    result["judge"] = judge
    time.sleep(RATE_LIMIT_DELAY)
    return result


# ==================== CONSISTENCY ANALYSIS ====================

def analyze_consistency(multiple_runs: List[Dict], framework: str) -> Dict:
    """Analyze consistency across multiple evaluation runs.
    
    Reports mean, standard deviation, mode, and range for scores.
    Mode is important: if scores are 3,3,4, mean=3.33 but mode=3 (more honest).
    """
    
    if framework in ["ABCD", "SMART"]:
        # Extract scores from each run
        criteria_scores = defaultdict(list)
        composite_scores = []
        
        for run in multiple_runs:
            composite_scores.append(run.get("composite_score", 0))
            for criterion, data in run.get("overall_scores", {}).items():
                criteria_scores[criterion].append(data.get("score", 0))
        
        # Calculate statistics including mode
        try:
            composite_mode = statistics.mode(composite_scores)
        except statistics.StatisticsError:
            # No unique mode (all different or multimodal)
            composite_mode = None
        
        consistency_report = {
            "composite_score_mean": statistics.mean(composite_scores),
            "composite_score_mode": composite_mode,
            "composite_score_stdev": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
            "composite_score_range": [min(composite_scores), max(composite_scores)],
            "criteria_consistency": {}
        }
        
        for criterion, scores in criteria_scores.items():
            try:
                criterion_mode = statistics.mode(scores)
            except statistics.StatisticsError:
                criterion_mode = None
            
            consistency_report["criteria_consistency"][criterion] = {
                "mean": statistics.mean(scores),
                "mode": criterion_mode,
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "range": [min(scores), max(scores)],
                "is_consistent": statistics.stdev(scores) <= 0.5 if len(scores) > 1 else True
            }
    
    elif framework == "BLOOMS":
        # Bloom's has different structure
        individual_composites = []
        set_scores = []
        
        for run in multiple_runs:
            for eval_item in run.get("individual_evaluations", []):
                individual_composites.append(eval_item.get("composite_score", 0))
            set_scores.append(run.get("set_evaluation", {}).get("set_level_composite_score", 0))
        
        consistency_report = {
            "individual_mean": statistics.mean(individual_composites) if individual_composites else 0,
            "individual_stdev": statistics.stdev(individual_composites) if len(individual_composites) > 1 else 0,
            "set_level_mean": statistics.mean(set_scores) if set_scores else 0,
            "set_level_stdev": statistics.stdev(set_scores) if len(set_scores) > 1 else 0,
            "overall_mean": statistics.mean(individual_composites + set_scores) if (individual_composites or set_scores) else 0
        }
    
    return consistency_report


def calculate_inter_judge_agreement(judge1_runs: List[Dict], judge2_runs: List[Dict], framework: str) -> Dict:
    """Calculate agreement metrics between two judges (e.g., Gemini vs Groq).
    
    Metrics:
    - Exact agreement: % of times judges gave exactly same score
    - Within-±1 agreement: % of times judges were within 1 point
    - Mean bias: Average difference (Judge1 - Judge2)
    - Correlation: Pearson correlation coefficient
    - Cohen's Kappa: Inter-rater reliability (adjusted for chance)
    """
    
    if framework in ["ABCD", "SMART"]:
        agreement_report = {
            "criteria_agreement": {},
            "composite_agreement": {}
        }
        
        # Extract composite scores
        judge1_composites = [run.get("composite_score", 0) for run in judge1_runs]
        judge2_composites = [run.get("composite_score", 0) for run in judge2_runs]
        
        # Calculate composite agreement
        exact_matches = sum(1 for j1, j2 in zip(judge1_composites, judge2_composites) if abs(j1 - j2) < 0.1)
        within_one = sum(1 for j1, j2 in zip(judge1_composites, judge2_composites) if abs(j1 - j2) <= 1.0)
        
        agreement_report["composite_agreement"] = {
            "exact_agreement_pct": (exact_matches / len(judge1_composites)) * 100 if judge1_composites else 0,
            "within_1_agreement_pct": (within_one / len(judge1_composites)) * 100 if judge1_composites else 0,
            "mean_bias": statistics.mean([j1 - j2 for j1, j2 in zip(judge1_composites, judge2_composites)]) if judge1_composites else 0,
            "correlation": calculate_pearson_correlation(judge1_composites, judge2_composites) if len(judge1_composites) > 1 else 0
        }
        
        # Calculate criteria-level agreement
        criteria = set()
        for run in judge1_runs:
            criteria.update(run.get("overall_scores", {}).keys())
        
        for criterion in criteria:
            judge1_scores = [run.get("overall_scores", {}).get(criterion, {}).get("score", 0) for run in judge1_runs]
            judge2_scores = [run.get("overall_scores", {}).get(criterion, {}).get("score", 0) for run in judge2_runs]
            
            exact = sum(1 for j1, j2 in zip(judge1_scores, judge2_scores) if j1 == j2)
            within_1 = sum(1 for j1, j2 in zip(judge1_scores, judge2_scores) if abs(j1 - j2) <= 1)
            
            agreement_report["criteria_agreement"][criterion] = {
                "exact_agreement_pct": (exact / len(judge1_scores)) * 100 if judge1_scores else 0,
                "within_1_agreement_pct": (within_1 / len(judge1_scores)) * 100 if judge1_scores else 0,
                "mean_bias": statistics.mean([j1 - j2 for j1, j2 in zip(judge1_scores, judge2_scores)]) if judge1_scores else 0,
                "cohens_kappa": calculate_cohens_kappa_simple(judge1_scores, judge2_scores)
            }
    
    elif framework == "BLOOMS":
        # For Bloom's, compare individual evaluations and set-level scores
        judge1_individual = []
        judge2_individual = []
        for j1_run, j2_run in zip(judge1_runs, judge2_runs):
            for j1_eval, j2_eval in zip(j1_run.get("individual_evaluations", []), j2_run.get("individual_evaluations", [])):
                judge1_individual.append(j1_eval.get("composite_score", 0))
                judge2_individual.append(j2_eval.get("composite_score", 0))
        
        exact = sum(1 for j1, j2 in zip(judge1_individual, judge2_individual) if abs(j1 - j2) < 0.1)
        within_1 = sum(1 for j1, j2 in zip(judge1_individual, judge2_individual) if abs(j1 - j2) <= 1.0)
        
        agreement_report = {
            "individual_agreement": {
                "exact_agreement_pct": (exact / len(judge1_individual)) * 100 if judge1_individual else 0,
                "within_1_agreement_pct": (within_1 / len(judge1_individual)) * 100 if judge1_individual else 0,
                "mean_bias": statistics.mean([j1 - j2 for j1, j2 in zip(judge1_individual, judge2_individual)]) if judge1_individual else 0,
                "correlation": calculate_pearson_correlation(judge1_individual, judge2_individual) if len(judge1_individual) > 1 else 0
            }
        }
    
    return agreement_report


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    
    return numerator / (denominator_x * denominator_y)


def calculate_cohens_kappa_simple(scores1: List[int], scores2: List[int]) -> float:
    """Calculate Cohen's Kappa for inter-rater reliability."""
    if len(scores1) != len(scores2) or len(scores1) == 0:
        return 0.0
    
    n = len(scores1)
    
    # Observed agreement
    p_o = sum(1 for s1, s2 in zip(scores1, scores2) if s1 == s2) / n
    
    # Expected agreement by chance
    categories = set(scores1 + scores2)
    p_e = 0.0
    for category in categories:
        p1 = sum(1 for s in scores1 if s == category) / n
        p2 = sum(1 for s in scores2 if s == category) / n
        p_e += p1 * p2
    
    if p_e == 1.0:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


# ==================== MAIN EVALUATION PIPELINE ====================

def evaluate_framework(framework_name: str, input_file: str, course_context: str = ""):
    """Evaluate LOs from a specific framework file using dual judges."""
    
    print(f"\n{'='*70}")
    print(f"  EVALUATING {framework_name} LEARNING OBJECTIVES")
    print(f"  Dual Judges: Gemini 2.0 Flash + Llama 3.3 70B (Groq)")
    print(f"{'='*70}")
    
    # Load learning objectives
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    learning_objectives = data.get("learning_objectives", [])
    
    if not learning_objectives:
        print(f"❌ No learning objectives found in {input_file}")
        return
    
    print(f"\n📚 Found {len(learning_objectives)} learning objectives")
    print(f"🔄 Running {NUM_EVALUATION_RUNS} evaluation rounds × 2 judges for inter-judge agreement\n")
    
    # Run multiple evaluations with BOTH judges
    if framework_name == "BLOOMS":
        # Evaluate entire set multiple times with both judges
        gemini_runs = []
        groq_runs = []
        
        for run_num in range(1, NUM_EVALUATION_RUNS + 1):
            print(f"  Run {run_num}/{NUM_EVALUATION_RUNS}...")
            print(f"    Gemini 2.0 Flash...", end=" ")
            gemini_eval = evaluate_blooms_set(learning_objectives, run_num, judge="gemini")
            gemini_runs.append(gemini_eval)
            print("✓")
            
            print(f"    Llama 3.3 70B (Groq)...", end=" ")
            groq_eval = evaluate_blooms_set(learning_objectives, run_num, judge="groq")
            groq_runs.append(groq_eval)
            print("✓")
        
        # Analyze consistency for each judge
        gemini_consistency = analyze_consistency(gemini_runs, "BLOOMS")
        groq_consistency = analyze_consistency(groq_runs, "BLOOMS")
        
        # Analyze inter-judge agreement
        inter_judge_agreement = calculate_inter_judge_agreement(gemini_runs, groq_runs, "BLOOMS")
        
        # Save results
        output = {
            "framework": framework_name,
            "course_title": data.get("course_title"),
            "course_code": data.get("course_code"),
            "num_objectives": len(learning_objectives),
            "gemini_evaluation": {
                "model": GEMINI_MODEL,
                "evaluation_runs": gemini_runs,
                "consistency_analysis": gemini_consistency
            },
            "groq_evaluation": {
                "model": GROQ_MODEL,
                "evaluation_runs": groq_runs,
                "consistency_analysis": groq_consistency
            },
            "inter_judge_agreement": inter_judge_agreement,
            "metadata": {
                "primary_judge": GEMINI_MODEL,
                "validation_judge": GROQ_MODEL,
                "num_runs": NUM_EVALUATION_RUNS,
                "rubric": "Bloom's Taxonomy"
            }
        }
        
    else:
        # Evaluate each LO individually with both judges, multiple times
        all_evaluations = []
        
        for i, lo in enumerate(learning_objectives, 1):
            print(f"\n  LO {i}/{len(learning_objectives)}: {lo[:80]}...")
            gemini_runs = []
            groq_runs = []
            
            for run_num in range(1, NUM_EVALUATION_RUNS + 1):
                print(f"    Run {run_num}/{NUM_EVALUATION_RUNS}:")
                
                # Gemini evaluation
                print(f"      Gemini...", end=" ")
                if framework_name == "ABCD":
                    gemini_eval = evaluate_abcd_learning_objective(lo, run_num, judge="gemini")
                elif framework_name == "SMART":
                    gemini_eval = evaluate_smart_learning_objective(lo, course_context, run_num, judge="gemini")
                gemini_runs.append(gemini_eval)
                print(f"✓ Score: {gemini_eval.get('composite_score', 0):.2f}")
                
                # Groq evaluation
                print(f"      Groq...", end=" ")
                if framework_name == "ABCD":
                    groq_eval = evaluate_abcd_learning_objective(lo, run_num, judge="groq")
                elif framework_name == "SMART":
                    groq_eval = evaluate_smart_learning_objective(lo, course_context, run_num, judge="groq")
                groq_runs.append(groq_eval)
                print(f"✓ Score: {groq_eval.get('composite_score', 0):.2f}")
            
            # Analyze consistency for each judge
            gemini_consistency = analyze_consistency(gemini_runs, framework_name)
            groq_consistency = analyze_consistency(groq_runs, framework_name)
            
            # Analyze inter-judge agreement for this LO
            lo_agreement = calculate_inter_judge_agreement(gemini_runs, groq_runs, framework_name)
            
            all_evaluations.append({
                "learning_objective": lo,
                "objective_number": i,
                "gemini_evaluation": {
                    "evaluation_runs": gemini_runs,
                    "consistency_analysis": gemini_consistency
                },
                "groq_evaluation": {
                    "evaluation_runs": groq_runs,
                    "consistency_analysis": groq_consistency
                },
                "inter_judge_agreement": lo_agreement
            })
        
        # Calculate overall inter-judge agreement across all LOs
        all_gemini_scores = []
        all_groq_scores = []
        for eval_data in all_evaluations:
            all_gemini_scores.extend([run.get("composite_score", 0) for run in eval_data["gemini_evaluation"]["evaluation_runs"]])
            all_groq_scores.extend([run.get("composite_score", 0) for run in eval_data["groq_evaluation"]["evaluation_runs"]])
        
        overall_agreement = {
            "exact_agreement_pct": (sum(1 for g, gr in zip(all_gemini_scores, all_groq_scores) if abs(g - gr) < 0.1) / len(all_gemini_scores)) * 100 if all_gemini_scores else 0,
            "within_1_agreement_pct": (sum(1 for g, gr in zip(all_gemini_scores, all_groq_scores) if abs(g - gr) <= 1.0) / len(all_gemini_scores)) * 100 if all_gemini_scores else 0,
            "mean_bias": statistics.mean([g - gr for g, gr in zip(all_gemini_scores, all_groq_scores)]) if all_gemini_scores else 0,
            "correlation": calculate_pearson_correlation(all_gemini_scores, all_groq_scores) if len(all_gemini_scores) > 1 else 0
        }
        
        # Save results
        output = {
            "framework": framework_name,
            "course_title": data.get("course_title"),
            "course_code": data.get("course_code"),
            "num_objectives": len(learning_objectives),
            "evaluations": all_evaluations,
            "overall_inter_judge_agreement": overall_agreement,
            "metadata": {
                "primary_judge": GEMINI_MODEL,
                "validation_judge": GROQ_MODEL,
                "num_runs": NUM_EVALUATION_RUNS,
                "rubric": f"{framework_name} Framework"
            }
        }
    
    # Save output
    output_file = os.path.join(OUTPUT_DIR, f"evaluation_{framework_name.lower()}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved evaluation results: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  INTER-JUDGE AGREEMENT SUMMARY - {framework_name}")
    print(f"{'='*70}")
    
    if framework_name == "BLOOMS":
        agreement = inter_judge_agreement.get("individual_agreement", {})
        print(f"  Exact Agreement: {agreement.get('exact_agreement_pct', 0):.1f}%")
        print(f"  Within-±1 Agreement: {agreement.get('within_1_agreement_pct', 0):.1f}%")
        print(f"  Mean Bias (Gemini - Llama): {agreement.get('mean_bias', 0):+.2f}")
        print(f"  Correlation: {agreement.get('correlation', 0):.3f}")
        print(f"\n  Gemini Avg: {gemini_consistency['overall_mean']:.2f}")
        print(f"  Llama Avg: {groq_consistency['overall_mean']:.2f}")
    else:
        agreement = output.get("overall_inter_judge_agreement", {})
        print(f"  Exact Agreement: {agreement.get('exact_agreement_pct', 0):.1f}%")
        print(f"  Within-±1 Agreement: {agreement.get('within_1_agreement_pct', 0):.1f}%")
        print(f"  Mean Bias (Gemini - Llama): {agreement.get('mean_bias', 0):+.2f}")
        print(f"  Correlation: {agreement.get('correlation', 0):.3f}")
        
        # Print per-LO summaries
        gemini_scores = [eval_data["gemini_evaluation"]["consistency_analysis"]["composite_score_mean"] for eval_data in all_evaluations]
        groq_scores = [eval_data["groq_evaluation"]["consistency_analysis"]["composite_score_mean"] for eval_data in all_evaluations]
        print(f"\n  Gemini Avg: {statistics.mean(gemini_scores):.2f} ± {statistics.stdev(gemini_scores):.2f}")
        print(f"  Llama Avg: {statistics.mean(groq_scores):.2f} ± {statistics.stdev(groq_scores):.2f}")
        
        # Print interpretation
        if agreement.get('within_1_agreement_pct', 0) >= 80:
            print(f"\n  ✅ Strong inter-judge agreement (≥80% within ±1)")
        elif agreement.get('within_1_agreement_pct', 0) >= 60:
            print(f"\n  ⚠️  Moderate inter-judge agreement (60-80% within ±1)")
        else:
            print(f"\n  ❌ Weak inter-judge agreement (<60% within ±1) - consider rubric refinement")


def main():
    """Run complete evaluation pipeline with dual judges."""
    print("="*70)
    print("  LLM-AS-JUDGE EVALUATION FRAMEWORK v2.1")
    print("  Dual Judges: Gemini 2.0 Flash + Llama 3.3 70B (Groq)")
    print("  Evaluating Learning Objectives with Detailed Rubrics")
    print("="*70)
    
    # Check API keys
    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY not found in environment!")
        print("   Add to .env file: GEMINI_API_KEY=your_key_here")
        return
    
    if not GROQ_API_KEY:
        print("\n❌ GROQ_API_KEY not found in environment!")
        print("   Add to .env file: GROQ_API_KEY=your_key_here")
        return
    
    print("\n✓ API keys found for both judges")
    print(f"  Primary: {GEMINI_MODEL}")
    print(f"  Validation: {GROQ_MODEL}")
    
    # Evaluate each framework
    course_context = "Process scheduling, synchronization, memory management, file systems, deadlock, security"
    
    # 1. ABCD Framework
    if os.path.exists(ABCD_INPUT):
        evaluate_framework("ABCD", ABCD_INPUT, course_context)
    else:
        print(f"\n⚠️  ABCD input file not found: {ABCD_INPUT}")
    
    # 2. SMART Framework
    if os.path.exists(SMART_INPUT):
        evaluate_framework("SMART", SMART_INPUT, course_context)
    else:
        print(f"\n⚠️  SMART input file not found: {SMART_INPUT}")
    
    # 3. Bloom's Taxonomy
    if os.path.exists(BLOOMS_INPUT):
        evaluate_framework("BLOOMS", BLOOMS_INPUT, course_context)
    else:
        print(f"\n⚠️  Blooms input file not found: {BLOOMS_INPUT}")
    
    print("\n" + "="*70)
    print("  ✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {OUTPUT_DIR}/")
    print("   - evaluation_abcd.json")
    print("   - evaluation_smart.json")
    print("   - evaluation_blooms.json")


if __name__ == "__main__":
    main()
