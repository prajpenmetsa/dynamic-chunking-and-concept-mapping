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

# Ollama API Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = "https://ollama.com"
MODEL_NAME = "gpt-oss:20b-cloud"
RATE_LIMIT_DELAY = 5

# Paths
ABCD_INPUT = "../../datasets/slide_based_los_simple_abcd.json"
SMART_INPUT = "../../datasets/slide_based_los_simple_smart.json"
BLOOMS_INPUT = "../../datasets/slide_based_los_simple_blooms.json"
OUTPUT_DIR = "../../datasets/evaluation"

# Evaluation settings
NUM_EVALUATION_RUNS = 3  # Run each evaluation multiple times for consistency


# ==================== API HELPER ====================
def call_ollama_api(prompt: str, temperature: float = 0.3) -> Dict:
    """Call Ollama Cloud API with JSON response parsing."""
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "format": "json"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the JSON response
            response_text = result.get("response", "")
            return json.loads(response_text)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"   ⚠️  Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    
    raise Exception("Max retries exceeded")


# ==================== RUBRIC DEFINITIONS ====================

ABCD_RUBRIC = """
## ABCD Framework Rubric (1-5 Scale)

### A - Audience (Who will learn?)
**5 - Excellent**: Explicitly states the learner (e.g., "Students will...", "Learners will..."). Crystal clear who is learning.
**4 - Good**: Audience is implicit but clear from context (e.g., "Understand..." implies student audience).
**3 - Adequate**: Audience can be inferred but not explicitly stated. Some ambiguity exists.
**2 - Poor**: Audience is unclear or could apply to multiple groups (students, teachers, developers).
**1 - Unacceptable**: No identifiable audience. Generic statement that doesn't specify who learns.

### B - Behavior (What will they DO?)
**5 - Excellent**: Uses precise, observable, measurable action verb (analyze, design, implement, evaluate). Behavior is concrete and demonstrable.
**4 - Good**: Uses clear action verb but slightly less measurable (understand, explain, describe). Still actionable.
**3 - Adequate**: Uses action verb but it's vague or weak (know, learn, be aware of). Behavior is poorly defined.
**2 - Poor**: Verb is passive or non-behavioral (be exposed to, appreciate). Hard to observe/measure.
**1 - Unacceptable**: No action verb or uses "understand" without clarifying how it's demonstrated.

### C - Condition (Under what circumstances?)
**5 - Excellent**: Explicitly states conditions/context (e.g., "After completing the project...", "Using synchronization primitives...", "Given a scheduling scenario...").
**4 - Good**: Conditions are implied by the content but not explicitly stated. Context is clear.
**3 - Adequate**: Some contextual hints exist but conditions are vague. Unclear when/how behavior occurs.
**2 - Poor**: Missing meaningful conditions. No context for when learning is demonstrated.
**1 - Unacceptable**: Completely absent. No indication of circumstances or context.

### D - Degree (How well?)
**5 - Excellent**: Specific performance standard (e.g., "with 90% accuracy", "identifying at least 3 differences", "correctly handling all edge cases").
**4 - Good**: Implicit standard that's reasonably clear (e.g., "effectively", "accurately", "correctly").
**3 - Adequate**: Vague standard (e.g., "well", "properly"). Unclear what constitutes success.
**2 - Poor**: Minimal or weak standard. Success criteria are ambiguous.
**1 - Unacceptable**: No standard specified. Impossible to determine what constitutes achievement.

**CRITICAL EVALUATION QUESTIONS:**
1. Can you identify WHO is learning without ambiguity?
2. Is the behavior something you can OBSERVE and MEASURE?
3. Are the CONDITIONS/CONTEXT under which the behavior occurs clear?
4. Is there a STANDARD that defines successful achievement?
5. Could this LO be assessed objectively in an exam or assignment?
"""

SMART_RUBRIC = """
## SMART Framework Rubric (1-5 Scale)

### S - Specific (Is it focused and clear?)
**5 - Excellent**: Highly specific, names exact concepts/skills (e.g., "Implement the Banker's algorithm for deadlock avoidance"). No ambiguity.
**4 - Good**: Specific but covers broader area (e.g., "Analyze deadlock prevention strategies"). Clear scope.
**3 - Adequate**: Somewhat specific but includes vague terms (e.g., "Understand operating system concepts"). Limited focus.
**2 - Poor**: Very broad or generic (e.g., "Learn about OS"). Unclear what exactly is covered.
**1 - Unacceptable**: Completely vague or meaningless (e.g., "Gain knowledge"). No discernible focus.

### M - Measurable (Can achievement be assessed?)
**5 - Excellent**: Includes explicit success criteria or metrics (e.g., "correctly implement 3 scheduling algorithms", "identify all synchronization primitives").
**4 - Good**: Uses measurable verbs (analyze, design, implement) that allow objective assessment. Clear assessment path.
**3 - Adequate**: Can be measured but assessment method unclear. Verb allows some measurement (explain, describe).
**2 - Poor**: Difficult to measure objectively. Uses weak verbs (understand, know, appreciate).
**1 - Unacceptable**: Cannot be measured in any meaningful way. Purely subjective or intangible.

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
**5 - Excellent**: Explicit timeframe (e.g., "by end of course", "after module 3"). Clear deadline.
**4 - Good**: Timeframe implicit but reasonable (assumes end of course). Standard expectation.
**3 - Adequate**: Timeframe vague. Could be interpreted multiple ways.
**2 - Poor**: No clear timeframe. Unclear when achievement is expected.
**1 - Unacceptable**: Timeframe makes no sense or is completely absent.

**CRITICAL EVALUATION QUESTIONS:**
1. If you gave this to a student, would they know EXACTLY what to learn?
2. Can you create an exam question or assignment that measures this objectively?
3. Is this achievable within a single semester course for the target audience?
4. Would a course instructor agree this belongs in THIS specific course?
5. Is there a clear timeframe (even if implicit) for when this should be achieved?
"""

BLOOMS_RUBRIC = """
## Bloom's Taxonomy Rubric (1-5 Scale)

### Overall Taxonomy Alignment
**5 - Excellent**: Uses precise Bloom's verb from correct level. Cognitive demand matches verb perfectly. No ambiguity about intended level.
**4 - Good**: Uses appropriate Bloom's verb. Minor mismatch between verb and cognitive complexity. Generally aligned.
**3 - Adequate**: Uses Bloom's verb but complexity doesn't match stated level (e.g., says "analyze" but task is actually "apply").
**2 - Poor**: Uses weak or incorrect verb for intended level. Cognitive demand unclear or mismatched.
**1 - Unacceptable**: No Bloom's verb or completely wrong level (e.g., uses "remember" verb for "create" level task).

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

### Progression & Hierarchy
**5 - Excellent**: LOs show clear progression from lower to higher Bloom's levels. Build on each other logically.
**4 - Good**: Mostly progressive. Minor sequencing issues.
**3 - Adequate**: Some progression but gaps or illogical jumps exist.
**2 - Poor**: Little progression. Random ordering of levels.
**1 - Unacceptable**: All at same level or entirely reversed (high to low).

**CRITICAL EVALUATION QUESTIONS:**
1. Does the verb match the actual cognitive demand of the task?
2. Could you classify this LO into a single Bloom's level without ambiguity?
3. Does the task description match the complexity implied by the verb?
4. Do the LOs progress from foundational (Remember/Understand) to advanced (Analyze/Evaluate/Create)?
5. Are lower-level outcomes prerequisites for higher-level ones?
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
        "question": "Does the action verb accurately represent the cognitive level according to Bloom's Taxonomy (e.g., 'Analyze' for analysis, not recall)?",
        "scale": "1=Wrong/no Bloom's verb, 3=Approximate match, 5=Perfect Bloom's verb for intended level"
    },
    {
        "criterion": "Cognitive_Demand",
        "question": "Does the actual task complexity match the cognitive level implied by the verb?",
        "scale": "1=Complete mismatch, 3=Partial match, 5=Task perfectly matches verb level"
    },
    {
        "criterion": "Level_Classification",
        "question": "Can you unambiguously classify this LO into a single Bloom's level (Remember/Understand/Apply/Analyze/Evaluate/Create)?",
        "scale": "1=Impossible to classify, 3=Ambiguous between 2 levels, 5=Clearly one level"
    },
    {
        "criterion": "Progression",
        "question": "When viewed as a set, do the LOs progress from lower-order thinking (Remember/Understand) to higher-order (Analyze/Evaluate/Create)?",
        "scale": "1=No progression/reversed, 3=Some progression, 5=Clear logical progression"
    }
]


# ==================== EVALUATION PROMPTS ====================

def create_abcd_evaluation_prompt(learning_objective: str, run_number: int) -> str:
    """Create prompt for ABCD framework evaluation."""
    return f"""You are an expert educational assessment specialist evaluating learning objectives against the ABCD framework.

**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVE TO EVALUATE**:
"{learning_objective}"

**YOUR TASK**: Evaluate this learning objective against the ABCD framework using the detailed rubric below.

{ABCD_RUBRIC}

**EVALUATION INSTRUCTIONS**:
1. For EACH component (A, B, C, D), assign a score from 1-5 based on the rubric
2. Provide specific evidence from the LO that justifies your score
3. Answer the granular questions below with scores and justification

**GRANULAR QUESTIONS**:
{json.dumps(ABCD_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
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


def create_smart_evaluation_prompt(learning_objective: str, course_context: str, run_number: int) -> str:
    """Create prompt for SMART framework evaluation."""
    return f"""You are an expert educational assessment specialist evaluating learning objectives against the SMART framework.

**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVE TO EVALUATE**:
"{learning_objective}"

**COURSE CONTEXT**:
- Course: Advanced Operating Systems (CS3.304)
- Level: Senior undergraduate / Graduate
- Focus: {course_context}

**YOUR TASK**: Evaluate this learning objective against the SMART framework using the detailed rubric below.

{SMART_RUBRIC}

**EVALUATION INSTRUCTIONS**:
1. For EACH component (S, M, A, R, T), assign a score from 1-5 based on the rubric
2. Consider the course context when evaluating Achievable and Relevant
3. Provide specific evidence and justification

**GRANULAR QUESTIONS**:
{json.dumps(SMART_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
{{
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


def create_blooms_evaluation_prompt(all_objectives: List[str], run_number: int) -> str:
    """Create prompt for Bloom's Taxonomy evaluation."""
    objectives_text = "\n".join([f"{i+1}. {lo}" for i, lo in enumerate(all_objectives)])
    
    return f"""You are an expert educational assessment specialist evaluating learning objectives against Bloom's Taxonomy.

**EVALUATION RUN**: {run_number}/3 (Evaluate independently each time)

**LEARNING OBJECTIVES TO EVALUATE**:
{objectives_text}

**YOUR TASK**: Evaluate these learning objectives against Bloom's Taxonomy using the detailed rubric below.

{BLOOMS_RUBRIC}

**EVALUATION INSTRUCTIONS**:
1. For EACH learning objective:
   - Identify the Bloom's level (Remember/Understand/Apply/Analyze/Evaluate/Create)
   - Evaluate verb accuracy, cognitive demand alignment, and clarity
   - Assign a score from 1-5 for each criterion

2. For the SET of learning objectives:
   - Evaluate overall progression from lower to higher levels
   - Check for logical prerequisite relationships
   - Assess coverage across Bloom's levels

**GRANULAR QUESTIONS** (answer for each LO):
{json.dumps(BLOOMS_QUESTIONS, indent=2)}

**OUTPUT FORMAT** (JSON ONLY):
{{
  "individual_evaluations": [
    {{
      "objective_number": 1,
      "objective_text": "...",
      "identified_level": "Apply/Analyze/etc",
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


# ==================== EVALUATION EXECUTION ====================

def evaluate_abcd_learning_objective(lo: str, run_number: int) -> Dict:
    """Evaluate a single LO against ABCD framework."""
    prompt = create_abcd_evaluation_prompt(lo, run_number)
    result = call_ollama_api(prompt, temperature=0.3)
    time.sleep(RATE_LIMIT_DELAY)
    return result


def evaluate_smart_learning_objective(lo: str, course_context: str, run_number: int) -> Dict:
    """Evaluate a single LO against SMART framework."""
    prompt = create_smart_evaluation_prompt(lo, course_context, run_number)
    result = call_ollama_api(prompt, temperature=0.3)
    time.sleep(RATE_LIMIT_DELAY)
    return result


def evaluate_blooms_set(objectives: List[str], run_number: int) -> Dict:
    """Evaluate complete set of LOs against Bloom's Taxonomy."""
    prompt = create_blooms_evaluation_prompt(objectives, run_number)
    result = call_ollama_api(prompt, temperature=0.3)
    time.sleep(RATE_LIMIT_DELAY)
    return result


# ==================== CONSISTENCY ANALYSIS ====================

def analyze_consistency(multiple_runs: List[Dict], framework: str) -> Dict:
    """Analyze consistency across multiple evaluation runs."""
    
    if framework in ["ABCD", "SMART"]:
        # Extract scores from each run
        criteria_scores = defaultdict(list)
        composite_scores = []
        
        for run in multiple_runs:
            composite_scores.append(run.get("composite_score", 0))
            for criterion, data in run.get("overall_scores", {}).items():
                criteria_scores[criterion].append(data.get("score", 0))
        
        # Calculate statistics
        consistency_report = {
            "composite_score_mean": statistics.mean(composite_scores),
            "composite_score_stdev": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
            "composite_score_range": [min(composite_scores), max(composite_scores)],
            "criteria_consistency": {}
        }
        
        for criterion, scores in criteria_scores.items():
            consistency_report["criteria_consistency"][criterion] = {
                "mean": statistics.mean(scores),
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


# ==================== MAIN EVALUATION PIPELINE ====================

def evaluate_framework(framework_name: str, input_file: str, course_context: str = ""):
    """Evaluate LOs from a specific framework file."""
    
    print(f"\n{'='*70}")
    print(f"  EVALUATING {framework_name} LEARNING OBJECTIVES")
    print(f"{'='*70}")
    
    # Load learning objectives
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    learning_objectives = data.get("learning_objectives", [])
    
    if not learning_objectives:
        print(f"❌ No learning objectives found in {input_file}")
        return
    
    print(f"\n📚 Found {len(learning_objectives)} learning objectives")
    print(f"🔄 Running {NUM_EVALUATION_RUNS} evaluation rounds for consistency analysis\n")
    
    # Run multiple evaluations
    if framework_name == "BLOOMS":
        # Evaluate entire set multiple times
        all_runs = []
        for run_num in range(1, NUM_EVALUATION_RUNS + 1):
            print(f"  Run {run_num}/{NUM_EVALUATION_RUNS}...")
            evaluation = evaluate_blooms_set(learning_objectives, run_num)
            all_runs.append(evaluation)
        
        # Analyze consistency
        consistency = analyze_consistency(all_runs, "BLOOMS")
        
        # Save results
        output = {
            "framework": framework_name,
            "course_title": data.get("course_title"),
            "course_code": data.get("course_code"),
            "num_objectives": len(learning_objectives),
            "evaluation_runs": all_runs,
            "consistency_analysis": consistency,
            "metadata": {
                "model": MODEL_NAME,
                "num_runs": NUM_EVALUATION_RUNS,
                "rubric": "Bloom's Taxonomy"
            }
        }
        
    else:
        # Evaluate each LO individually, multiple times
        all_evaluations = []
        
        for i, lo in enumerate(learning_objectives, 1):
            print(f"\n  LO {i}/{len(learning_objectives)}: {lo[:80]}...")
            lo_runs = []
            
            for run_num in range(1, NUM_EVALUATION_RUNS + 1):
                print(f"    Run {run_num}/{NUM_EVALUATION_RUNS}...", end=" ")
                
                if framework_name == "ABCD":
                    evaluation = evaluate_abcd_learning_objective(lo, run_num)
                elif framework_name == "SMART":
                    evaluation = evaluate_smart_learning_objective(lo, course_context, run_num)
                
                lo_runs.append(evaluation)
                print(f"✓ Score: {evaluation.get('composite_score', 0):.2f}")
            
            # Analyze consistency for this LO
            consistency = analyze_consistency(lo_runs, framework_name)
            
            all_evaluations.append({
                "learning_objective": lo,
                "objective_number": i,
                "evaluation_runs": lo_runs,
                "consistency_analysis": consistency
            })
        
        # Save results
        output = {
            "framework": framework_name,
            "course_title": data.get("course_title"),
            "course_code": data.get("course_code"),
            "num_objectives": len(learning_objectives),
            "evaluations": all_evaluations,
            "metadata": {
                "model": MODEL_NAME,
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
    print(f"  CONSISTENCY SUMMARY - {framework_name}")
    print(f"{'='*70}")
    
    if framework_name == "BLOOMS":
        print(f"  Average Score: {consistency['overall_mean']:.2f}")
        print(f"  Individual LOs StdDev: {consistency['individual_stdev']:.2f}")
        print(f"  Set-Level StdDev: {consistency['set_level_stdev']:.2f}")
    else:
        print(f"  Average Composite Score: {consistency['composite_score_mean']:.2f} ± {consistency['composite_score_stdev']:.2f}")
        print(f"\n  Criteria Consistency:")
        for criterion, stats in consistency.get("criteria_consistency", {}).items():
            status = "✓ Consistent" if stats["is_consistent"] else "⚠️  Variable"
            print(f"    {criterion.capitalize():12s}: {stats['mean']:.2f} ± {stats['stdev']:.2f} [{stats['range'][0]}-{stats['range'][1]}] {status}")


def main():
    """Run complete evaluation pipeline."""
    print("="*70)
    print("  LLM-AS-JUDGE EVALUATION FRAMEWORK")
    print("  Evaluating Learning Objectives with Detailed Rubrics")
    print("="*70)
    
    if not OLLAMA_API_KEY:
        print("\n❌ OLLAMA_API_KEY not found in environment!")
        print("   Add to .env file: OLLAMA_API_KEY=your_key_here")
        return
    
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
