# LLM-as-Judge Evaluation Framework

A rigorous evaluation system for assessing learning objectives against **ABCD Framework**, **SMART Criteria**, and **Bloom's Taxonomy** using **dual LLM judges** with **hard constraints** and **inter-judge agreement validation**.

## Overview

This framework provides:
- **Dual-judge architecture** using Gemini 2.0 Flash (proprietary) + Llama 3.3 70B via Groq (open-source)
- **Inter-judge agreement metrics** (exact agreement %, within-±1%, Cohen's Kappa, correlation)
- **Detailed rubrics** with 1-5 scoring scales and **hard constraints** (e.g., "if uses 'understand', score ≤2")
- **Discriminating evaluation lenses** to distinguish ABCD (structural completeness) from SMART (practical assessability)
- **Multiple evaluation runs** (default: 3) with mean, **mode**, and stdev reporting for consistency
- **Binary checklists** embedded in prompts to enforce disciplined evaluation
- **System prompts** (rubrics) vs user prompts (LOs) for better LLM internalization
- **Calibration tooling** to measure human-LLM agreement (Cohen's Kappa, within-±1 agreement)

## Why Dual Judges?

### Methodological Rigor
Using **two independent judges** from different providers strengthens evaluation validity:

1. **Gemini 2.0 Flash** (Primary Judge)
   - Proprietary model from Google AI Studio
   - Free tier: 1500 requests/day
   - Exceptionally strong at structured output and complex rubrics
   - Fast inference (critical for 3 runs × N LOs × 3 frameworks)
   - Well-recognized in evaluation literature

2. **Llama 3.3 70B via Groq** (Validation Judge)
   - Open-source model (reproducible by reviewers)
   - Free tier with very generous rate limits
   - Comparable instruction-following to GPT-4 class
   - Ultra-fast inference via Groq (no bottleneck)
   - Open-weights enable precise model card citation

### Inter-Judge Agreement as a Validity Signal

**If both judges agree** → Strong evidence of rubric clarity and score objectivity  
**If judges disagree** → Interesting finding worth discussing in paper (rubric ambiguity, model bias, or genuine LO quality ambiguity)

**Reporting thresholds for papers**:
- **Excellent**: Within-±1 agreement ≥80%
- **Good**: Within-±1 agreement ≥60%
- **Needs improvement**: <60% (consider rubric refinement)

**Why this matters**:
- Demonstrates that scores aren't model-specific artifacts
- Combines proprietary + open-source for reproducibility
- Reviewers can re-run with Llama 3.3 to verify results

## Key Improvements (v2.0)

### 1. Hard Constraints (Addresses Score Inconsistencies)
**Problem**: Original rubric gave "understand" a score of 4/5 in Behavior, contradicting Bloom's guidance.

**Solution**: Explicit hard constraints in rubrics:

**Solution**: Explicit hard constraints in rubrics:

```python
# ABCD Behavior
- If LO uses "understand", "know", "learn" WITHOUT clarifying demonstration → Score ≤ 2

# SMART Measurable  
- If LO uses "understand", "know", "appreciate" → Score ≤ 2 (cannot assess objectively)

# Bloom's Taxonomy
- If verb is "understand"/"know" → MUST classify as Remember/Understand (low level)
- If verb says "analyze" but task is just recall → Cognitive Demand ≤ 2
```

These prevent the LLM from being generous when rubric-violating patterns appear.

### 2. Discriminating Evaluation Lenses (Addresses Framework Overlap)

**Problem**: ABCD "Behavior" and SMART "Measurable" evaluate nearly the same thing, producing correlated scores without adding signal.

**Solution**: Explicit lens differentiation in system prompts:

- **ABCD Lens**: "Focus on STRUCTURE and COMPLETENESS. Does it have all four components?"
- **SMART Lens**: "Focus on CLARITY and PRACTICAL FEASIBILITY. Can we define SUCCESS criteria with concrete metrics?"

**Key distinction**:
- ABCD Behavior asks: "Is it **observable**?" (yes/no structural check)
- SMART Measurable asks: "Can we **measure success** with concrete criteria?" (grading rubric feasibility)

### 3. Individual vs. Set-Level Bloom's Criteria (Fixes Logical Error)

**Problem**: Original design scored individual LOs on "progression" - but progression is a property of the complete set, not a single LO.

**Solution**: Split Bloom's evaluation into two phases:

**INDIVIDUAL LO CRITERIA** (per LO):
- Verb Accuracy: Does verb match cognitive level?
- Cognitive Demand: Does task match verb complexity?
- Level Classification: Can you classify unambiguously?

**SET-LEVEL CRITERIA** (all LOs together):
- Progression: Do LOs build Remember → Create?
- Coverage: Span multiple Bloom's levels?
- Prerequisites: Are lower levels prerequisites for higher?

### 4. Binary Checklists in Prompts (Enforces Discipline)

**Problem**: LLMs could jump to holistic scores without systematic analysis.

**Solution**: Five binary YES/NO questions embedded in evaluation prompts, answered BEFORE scoring:

**ABCD Checklist**:
```
[ ] Can you identify WHO is learning?
[ ] Is the verb OBSERVABLE?  
[ ] Are CONDITIONS clear?
[ ] Is there a STANDARD?
[ ] Could you assess this objectively?
```

LLM must answer these first, then assign scores - prevents skipping critical dimensions.

### 5. System Prompts for Rubrics (Better LLM Internalization)

**Problem**: Sending rubric + LO together as user prompt meant LLM saw them as equal-weight information.

**Solution**: Rubrics sent as **system prompt**, LO as **user prompt**:

```python
system_prompt = f"""You are an expert educational assessment specialist.
{ABCD_RUBRIC}
Apply this rubric consistently."""

user_prompt = f"""Evaluate this LO: "{learning_objective}" """

call_ollama_api(user_prompt, system_prompt=system_prompt)
```

This "internalizes" the rubric before seeing the LO, improving consistency.

### 6. Mode Reporting (Honest Score Summary)

**Problem**: If scores are 3, 3, 4 across runs, mean = 3.33 but the LLM actually chose 3 twice.

**Solution**: Report both mean and **mode**:

```json
{
  "composite_score_mean": 3.33,
  "composite_score_mode": 3,  ← More honest representation
  "composite_score_stdev": 0.47
}
```

Mode is reported in calibration analysis and evaluation outputs.

### 7. Calibration System (Validates LLM Judge Quality)

**Problem**: No way to validate if LLM scores match expert human judgment.

**Solution**: Full calibration workflow:

1. **Manually score 5-10 LOs** using rubrics (template provided)
2. **Run LLM evaluation** on same LOs  
3. **Run calibration analysis** to compute:
   - Exact agreement %
   - Within-±1 agreement % (practical threshold)
   - Cohen's Kappa (inter-rater reliability)
   - Mean bias (LLM too lenient/strict?)
   - Pearson correlation

**Paper-ready thresholds**:
- Cohen's Kappa ≥ 0.40 (moderate agreement)
- Within-±1 agreement ≥ 60% (practical reliability)

## Rubric Design Philosophy

### Hard Constraints vs. Soft Scoring

**Soft scoring** (original): "Score 1-5 based on quality"
→ Problem: LLM can rationalize high scores even for weak LOs

**Hard constraints** (v2): "If X condition, score ≤ Y"
→ Benefit: Prevents generous scoring when rubric-violating patterns appear

Example:
```
If the LO uses "understand" without clarifying HOW it will be demonstrated
→ Behavior score ≤ 2 (not negotiable)
```

### Discriminating Lenses

Each framework evaluates from a distinct perspective:

| Framework | Focus | Key Question |
|-----------|-------|--------------|
| **ABCD** | Structural completeness | "Are all 4 components present?" |
| **SMART** | Practical usability | "Can we create grading criteria?" |
| **Bloom's** | Cognitive accuracy | "Does the verb match the thinking required?" |

This prevents frameworks from collapsing into redundant measures.

## Installation

```bash
# Install requirements
pip install requests python-dotenv

# Set up API keys (see below for how to obtain)
cp .env.example .env
# Edit .env and add your actual API keys
```

### Getting API Keys

#### 1. Gemini API Key (Primary Judge)
1. Go to **Google AI Studio**: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Get API key" or "Create API key"
4. Copy the key and add to `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_key_here
   ```
5. **Free tier**: 1500 requests/day (more than enough for evaluation)

#### 2. Groq API Key (Validation Judge)
1. Go to **Groq Console**: https://console.groq.com/keys
2. Sign up or log in
3. Click "Create API Key"
4. Copy the key and add to `.env` file:
   ```
   GROQ_API_KEY=your_groq_key_here
   ```
5. **Free tier**: Very generous rate limits with ultra-fast inference

**Your `.env` file should look like**:
```bash
GEMINI_API_KEY=AIzaSyC...your_actual_key
GROQ_API_KEY=gsk_...your_actual_key
```

**No billing required** for either service at expected evaluation volumes.

## Usage

### Quick Start (Full Pipeline)

```bash
# 1. Run LLM evaluation on all frameworks
cd src/evaluation
python llm_judge_evaluation.py

# 2. Generate human-readable reports
python generate_evaluation_report.py

# 3. (Optional) Validate with human calibration
# Edit calibration_set.json with your manual scores
python calibration_analysis.py
```

### Step-by-Step: LLM Evaluation

```bash
python llm_judge_evaluation.py
```

**What happens**:
1. Loads LOs from `datasets/slide_based_los_simple_{abcd,smart,blooms}.json`
2. For each LO and each judge (Gemini + Llama):
   - Sends rubric as **system prompt** (internalize standards)
   - Sends LO as **user prompt** (evaluate this)
   - LLM answers binary checklist first
   - LLM applies hard constraints
   - LLM assigns scores with evidence
3. Repeats 3 times per LO per judge for consistency analysis
4. Calculates **inter-judge agreement** metrics
5. Reports mean, mode, stdev, range, and agreement statistics

**Expected runtime**: ~20-30 minutes for 7 LOs × 3 frameworks × 3 runs × 2 judges

**Output**: `datasets/evaluation/evaluation_{abcd,smart,blooms}.json` with dual-judge results

**Output structure**:
```json
{
  "framework": "ABCD",
  "evaluations": [
    {
      "learning_objective": "...",
      "gemini_evaluation": {
        "evaluation_runs": [...],  // 3 runs
        "consistency_analysis": {...}
      },
      "groq_evaluation": {
        "evaluation_runs": [...],  // 3 runs
        "consistency_analysis": {...}
      },
      "inter_judge_agreement": {
        "exact_agreement_pct": 66.7,
        "within_1_agreement_pct": 100.0,
        "mean_bias": -0.2,
        "correlation": 0.95
      }
    }
  ],
  "overall_inter_judge_agreement": {...}
}
```

### Step-by-Step: Report Generation

```bash
python generate_evaluation_report.py
```

Creates four reports in `datasets/evaluation/reports/`:
- `report_abcd.txt` - Detailed ABCD analysis with consistency metrics
- `report_smart.txt` - Detailed SMART analysis  
- `report_blooms.txt` - Bloom's individual + set-level analysis
- `report_summary.txt` - Cross-framework comparison

### Step-by-Step: Calibration (Validation)

This validates the LLM judge against expert human scoring.

**1. Manually score 5-10 LOs**:
```bash
# Copy template
cp datasets/evaluation/calibration_set_TEMPLATE.json \
   datasets/evaluation/calibration_set.json

# Edit calibration_set.json:
# - Choose 5-10 LOs from your generated set
# - Score each using RUBRIC_REFERENCE.md
# - Document your reasoning
```

**2. Run calibration analysis**:
```bash
python calibration_analysis.py
```

**Output**: `datasets/evaluation/reports/calibration_report.txt`

**Metrics computed**:
- **Exact agreement**: % of scores that match exactly
- **Within-±1 agreement**: % within one point (practical threshold)
- **Cohen's Kappa**: Inter-rater reliability (0.40+ = acceptable)
- **Mean bias**: Is LLM systematically higher/lower?
- **Pearson correlation**: Linear relationship strength

**Interpretation**:
```
Kappa ≥ 0.60  +  Within-1 ≥ 70%  →  Excellent (use in paper)
Kappa ≥ 0.40  +  Within-1 ≥ 60%  →  Acceptable (report with caveats)
Kappa < 0.40  or  Within-1 < 60%  →  Needs improvement (iterate on prompts)
```

## Configuration

Edit `llm_judge_evaluation.py`:

```python
# Number of evaluation runs for consistency
NUM_EVALUATION_RUNS = 3  # Increase to 5 for more rigorous testing

# Model temperature (lower = more deterministic)
temperature = 0.3  # Range: 0.0 (deterministic) to 1.0 (creative)

# Rate limiting
RATE_LIMIT_DELAY = 5  # seconds between API calls
```

## Output Structure

### Evaluation JSON (example: `evaluation_abcd.json`)

```json
{
  "framework": "ABCD",
  "course_title": "Advanced Operating Systems",
  "num_objectives": 7,
  "evaluations": [
    {
      "learning_objective": "Students will analyze...",
      "objective_number": 1,
      "evaluation_runs": [
        {
          "overall_scores": {
            "audience": {"score": 5, "evidence": "...", "weakness": "..."},
            "behavior": {"score": 4, "evidence": "...", "weakness": "..."},
            "condition": {"score": 3, "evidence": "...", "weakness": "..."},
            "degree": {"score": 2, "evidence": "...", "weakness": "..."}
          },
          "granular_responses": [...],
          "composite_score": 3.5,
          "overall_assessment": "...",
          "improvement_suggestions": [...]
        },
        // Runs 2 and 3...
      ],
      "consistency_analysis": {
        "composite_score_mean": 3.47,
        "composite_score_stdev": 0.15,
        "criteria_consistency": {
          "audience": {
            "mean": 4.67,
            "stdev": 0.47,
            "is_consistent": true
          },
          // Other criteria...
        }
      }
    }
    // More LOs...
  ]
}
```

### Report Text (example: `report_abcd.txt`)

```
================================================================================
  ABCD FRAMEWORK EVALUATION REPORT
================================================================================

Course: Advanced Operating Systems (CS3.304)
Number of Learning Objectives: 7
Evaluation Runs per LO: 3

────────────────────────────────────────────────────────────────────────────────
LEARNING OBJECTIVE #1
────────────────────────────────────────────────────────────────────────────────

Students will analyze the trade-offs between different process scheduling algorithms...

📊 CONSISTENCY ANALYSIS (across 3 runs):
   Overall Score: 3.47 ± 0.15
   Score Range: [3.25 - 3.50]

   Component Scores:
      ✓ AUDIENCE    : 4.67 ± 0.47  [4 - 5]
      ✓ BEHAVIOR    : 4.33 ± 0.47  [4 - 5]
      ⚠️ CONDITION  : 2.67 ± 0.47  [2 - 3]
      ⚠️ DEGREE     : 2.33 ± 0.47  [2 - 3]

📋 DETAILED EVALUATION (Run 1):
   AUDIENCE - Score: 5/5
      Evidence: "Explicitly states 'Students will'"
      
   BEHAVIOR - Score: 4/5
      Evidence: "Uses 'analyze' which is observable and measurable"
      Weakness: "Could be more specific about what analyzing entails"
      
   CONDITION - Score: 3/5
      Evidence: "Implies studying scheduling algorithms"
      Weakness: "No explicit conditions like 'given scenarios' or 'using simulation'"
      
   DEGREE - Score: 2/5
      Evidence: "No performance standard specified"
      Weakness: "Doesn't state how many algorithms, accuracy level, or criteria"

🔍 GRANULAR QUESTION RESPONSES:
   • Audience: 5/5
     Q: Is the intended learner explicitly stated?
     A: Yes, clearly states "Students will" at the beginning

   [... more responses ...]

💡 OVERALL ASSESSMENT:
   Strong audience identification and behavioral verb, but lacks explicit 
   conditions and performance standards. The LO is measurable but would 
   benefit from clearer success criteria.

🎯 IMPROVEMENT SUGGESTIONS:
   1. Add conditions: "Given 3 scheduling scenarios, students will analyze..."
   2. Add degree: "...correctly identifying at least 2 trade-offs per algorithm"
   3. Specify which algorithms: "...between FCFS, SJF, and Round-Robin"
```

## Rubric Details

### Scoring Interpretation

| Score | Meaning | Description |
|-------|---------|-------------|
| **5** | Excellent | Fully meets criterion with no gaps |
| **4** | Good | Meets criterion with minor issues |
| **3** | Adequate | Partially meets criterion, needs improvement |
| **2** | Poor | Significant gaps in meeting criterion |
| **1** | Unacceptable | Does not meet criterion at all |

### Consistency Thresholds

| Standard Deviation | Interpretation |
|-------------------|----------------|
| ≤ 0.5 | Consistent (✓) |
| 0.5 - 1.0 | Moderately variable (⚠️) |
| > 1.0 | Highly variable (❌) |

### Consistency Thresholds

| Standard Deviation | Interpretation |
|-------------------|----------------|
| ≤ 0.5 | Consistent (✓) - Reliable across runs |
| 0.5 - 1.0 | Moderately variable (⚠️) - Some disagreement |
| > 1.0 | Highly variable (❌) - Unreliable, needs revision |

### Score Interpretation for Papers

When reporting in academic papers:

**Report all three**: Mean, Mode, and Standard Deviation
- **Mean**: Average across runs (standard metric)
- **Mode**: Most common score (more honest when skewed)
- **StdDev**: Consistency indicator (lower is better)

**Example**:
> "ABCD Behavior criterion scored 3.33 ± 0.47 (mode=3) across 3 evaluation runs, 
> indicating moderate consistency (StdDev < 0.5)."

**Quality thresholds**:
| Score Range | Quality Level | Action |
|-------------|--------------|--------|
| 4.0 - 5.0 | Excellent | Minor refinements only |
| 3.0 - 3.9 | Good | Targeted improvements |
| 2.0 - 2.9 | Adequate | Significant revision needed |
| 1.0 - 1.9 | Poor | Major restructuring required |

**Calibration requirements for publication**:
- Cohen's Kappa ≥ 0.40 (minimum acceptable inter-rater reliability)
- Within-±1 agreement ≥ 60% (practical reliability threshold)
- If both met → Can justify LLM-as-judge approach in methodology

## Common Scoring Patterns

### Pattern 1: "Understand" Verb

**LO**: "Students will understand process scheduling."

**Expected Scores** (with hard constraints):
- ABCD Behavior: ≤ 2 (uses "understand" without demonstration method)
- SMART Measurable: ≤ 2 (cannot assess objectively - what proves understanding?)
- Bloom's Classification: Remember/Understand (low level, correctly identified)
- Bloom's Verb Accuracy: 1 (not a proper Bloom's verb - should be "explain" or "apply")

**Fix**: "Students will **explain** three process scheduling algorithms by comparing their trade-offs" → Bloom's Understand level with measurable behavior

### Pattern 2: High Behavior, Low Degree

**LO**: "Students will analyze synchronization mechanisms."

**Expected Scores**:
- ABCD Audience: 5 (explicit "Students")
- ABCD Behavior: 4-5 ("analyze" is observable)
- ABCD Condition: 3 (implicit context)
- ABCD Degree: 1-2 (no standard - how many? how well? what criteria?)

**Result**: Inconsistent component scores → composite ~3.0

**Fix**: Add degree: "...analyzing **at least 3** mechanisms by **identifying 2+ trade-offs each**"

### Pattern 3: Generic Content

**LO**: "Learn about operating system concepts."

**Expected Scores** (with hard constraints):
- SMART Specific: ≤ 2 ("OS concepts" is too generic - violates hard constraint)
- SMART Measurable: 1 (cannot create exam question from this)
- SMART Achievable: N/A (can't assess without knowing what's included)
- Bloom's Classification: 1 (impossible to classify - too vague)

**Fix**: "**Implement three memory management algorithms** (FIFO, LRU, Clock) to handle page faults"

## Troubleshooting

### Issue: Inconsistent Scores (high stdev > 0.5)

**Cause**: Model temperature too high, or ambiguous rubric language

**Solutions**:
```python
# 1. Lower temperature for more deterministic outputs
temperature = 0.1  # in call_ollama_api() - was 0.3

# 2. Increase evaluation runs to detect patterns
NUM_EVALUATION_RUNS = 5  # was 3

# 3. Add more hard constraints to rubric if specific patterns emerge
# Example: If LLM keeps scoring "describe" as 4, add constraint:
# "If verb is 'describe' → Behavior score ≤ 3"
```

### Issue: LLM Too Lenient (scores consistently high)

**Cause**: LLM being generous despite rubric constraints

**Check calibration**: Run `calibration_analysis.py` - if mean bias > +0.5, LLM is too lenient

**Solutions**:
1. **Strengthen hard constraints**:
   ```
   Current: "If uses 'understand' → score ≤ 2"
   Stricter: "If uses 'understand' OR 'know' OR 'learn' OR 'describe' 
              without specifying HOW demonstrated → score ≤ 2"
   ```

2. **Add negative examples to system prompt**:
   ```python
   system_prompt += """
   
   NEGATIVE EXAMPLES (should score ≤ 2):
   - "Understand operating systems" (vague verb)
   - "Learn about scheduling" (no observable behavior)
   - "Analyze processes" (no degree/standard)
   """
   ```

3. **Explicit "cannot exceed" language**:
   ```
   "Behavior scores CANNOT EXCEED 3 unless the verb is from 
    the approved list: analyze, design, implement, evaluate, create"
   ```

### Issue: LLM Too Strict (scores consistently low)

**Check calibration**: If mean bias < -0.5, LLM is too strict

**Solutions**:
1. **Add positive examples**:
   ```python
   system_prompt += """
   
   POSITIVE EXAMPLES (should score 4-5):
   - "Students will implement 3 scheduling algorithms..." (clear, measurable)
   - "Analyze at least 2 synchronization mechanisms..." (specific, has degree)
   ```

2. **Clarify "implicit is acceptable"**:
   ```
   "Audience score can be 4 even if 'Students' not stated, 
    IF context unambiguously implies student learners"
   ```

### Issue: Low Human-LLM Agreement (Kappa < 0.40)

**This is the critical issue for paper validity**

**Diagnosis workflow**:
```bash
# 1. Run calibration
python calibration_analysis.py

# 2. Check calibration_report.txt for:
#    - Which criteria have worst agreement?
#    - Is there systematic bias (LLM higher/lower)?
#    - Are disagreements random or patterned?

# 3. Review divergent cases manually
# Look for: What made you score 4 but LLM scored 2?
```

**Common causes & fixes**:

| Cause | Fix |
|-------|-----|
| **Rubric ambiguity** | Add concrete examples for each score level |
| **Missing hard constraints** | Add "if X then score ≤Y" rules for edge cases |
| **Instructions ignored** | Move critical rules from user prompt to system prompt |
| **Different interpretation** | Add explicit definitions (e.g., what counts as "condition"?) |
| **LLM sees patterns you don't** | Review LLM's evidence field - might be catching issues you missed |

**Iterative improvement**:
1. Run calibration (get Kappa)
2. Identify worst-agreement criteria
3. Update rubric with targeted constraints
4. Re-run LLM evaluation
5. Re-run calibration → Check if Kappa improved
6. Repeat until Kappa ≥ 0.40

### Issue: API Rate Limits

```bash
# Increase delay between calls
RATE_LIMIT_DELAY = 10  # in llm_judge_evaluation.py (was 5)

# Or reduce evaluation runs
NUM_EVALUATION_RUNS = 2  # (was 3) - but check consistency doesn't suffer
```

### Issue: JSON Parsing Errors

The script includes robust JSON parsing that handles:
- Markdown code blocks (```json ... ```)
- Preamble text ("Here's my evaluation...")
- Malformed JSON (attempts to extract valid portions)

If issues persist:
```python
# Add fallback in call_ollama_api():
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Raw response: {response_text[:500]}")
    # Debug: see what LLM actually returned
```

## For Your Paper: Reporting Guidelines

### Methodology Section

**Template**:
> We employed a dual-judge LLM evaluation approach to assess learning objectives 
> against three taxonomic frameworks (ABCD, SMART, Bloom's). Each LO was evaluated 
> independently 3 times by two judges: Gemini 2.0 Flash (proprietary) and Llama 3.3 70B 
> via Groq (open-source), with temperature=0.3. This dual-judge design strengthens 
> validity by demonstrating that scores are not model-specific artifacts.
> 
> Rubrics were provided as system prompts containing:
> 1. Detailed 1-5 scoring scales for each criterion
> 2. Hard constraints (e.g., "if uses 'understand' → score ≤2")  
> 3. Binary checklists to enforce systematic evaluation
> 4. Discriminating evaluation lenses to avoid framework redundancy
> 
> **Inter-judge agreement** was XX% exact agreement and XX% within-±1 agreement 
> (r = X.XX), indicating [strong/moderate] consensus. **Calibration validation**: 
> We manually scored 10 LOs and computed Cohen's Kappa = 0.XX (95% CI: [X.XX, X.XX]) 
> for human-LLM agreement, demonstrating [substantial/moderate] reliability.

### Results Section

**Report for BOTH judges**:
```
ABCD Framework Scores:

Gemini 2.0 Flash (mean ± stdev, mode):
- Audience: 4.2 ± 0.4 (mode=4) ✓
- Behavior: 3.1 ± 0.7 (mode=3) ⚠  
- Condition: 2.8 ± 0.4 (mode=3) ✓
- Degree: 2.3 ± 0.5 (mode=2) ✓

Llama 3.3 70B (mean ± stdev, mode):
- Audience: 4.0 ± 0.5 (mode=4) ✓
- Behavior: 3.3 ± 0.6 (mode=3) ✓  
- Condition: 2.9 ± 0.3 (mode=3) ✓
- Degree: 2.5 ± 0.4 (mode=2) ✓

Inter-judge agreement: 
- Within-±1: 95% (19/20 criterion-level comparisons)
- Mean bias: -0.15 (Gemini slightly stricter)
- Correlation: r = 0.89 (strong agreement)
```

**Interpret StdDev**:
- ≤ 0.5 → "Consistent across runs"
- 0.5-1.0 → "Moderate variability; scores should be interpreted with caution"
- > 1.0 → "High variability; indicates ambiguous criterion or rubric issue"

**Interpret Inter-Judge Agreement**:
- Within-±1 ≥ 80% → "Strong inter-judge consensus validates rubric objectivity"
- Within-±1 60-80% → "Moderate agreement; some rubric ambiguity remains"
- Within-±1 < 60% → "Low agreement; rubric refinement needed"

**Calibration results** (Human vs LLM):
```
Gemini vs Human: Kappa = 0.XX, Within-±1 = XX%
Llama vs Human: Kappa = 0.XX, Within-±1 = XX%
Mean bias (Gemini): ±X.X [lenient/strict/unbiased]
Mean bias (Llama): ±X.X [lenient/strict/unbiased]
```

### Discussion Section

**Dual-judge findings**:
> Both judges showed strong agreement (XX% within-±1), suggesting that evaluation 
> scores are robust across model architectures. The slight bias toward [leniency/strictness] 
> in [Model] (mean bias = ±X.X) was consistent but small. Using both a proprietary 
> model (Gemini) and open-source model (Llama 3.3) enhances reproducibility—future 
> researchers can validate our findings using publicly available weights.

**If judges disagreed**:
> Inter-judge agreement was moderate (XX% within-±1), with notable disagreement on 
> the [Criterion] dimension. This suggests [rubric ambiguity OR genuine LO quality 
> uncertainty]. Manual review of discrepant cases revealed [specific insight].

### Limitations Section

**Be honest**:
- "Both LLM judges showed slight bias toward [leniency/strictness] (mean bias = ±X.X)"
- "Kappa = 0.XX is [below optimal/acceptable] for human-LLM agreement, suggesting [action]"
- "Criterion X had low consistency (StdDev = X.X) and low inter-judge agreement, indicating rubric ambiguity"
- "While using two judges strengthens validity, both are transformer-based models and may share systematic biases"

## Citation

If you use this evaluation framework in research, please cite:

```bibtex
@misc{llm_judge_lo_eval_v2_1,
  title={Dual-Judge LLM Evaluation Framework for Learning Objectives (v2.1)},
  author={Your Name},
  year={2026},
  note={Rubric-based evaluation with dual judges (Gemini + Llama 3.3), 
        inter-judge agreement metrics, hard constraints, calibration validation, 
        and discriminating lenses for ABCD, SMART, and Bloom's Taxonomy}
}
```

## Version History

**v2.1 (2026-02-21)** - Dual-judge architecture:
- Added Gemini 2.0 Flash as primary judge (proprietary, strong structured output)
- Added Llama 3.3 70B via Groq as validation judge (open-source, reproducible)
- Inter-judge agreement metrics (exact %, within-±1%, correlation, Cohen's Kappa)
- Updated API integration with retry logic and rate limit handling
- Enhanced methodology for research paper publication
- Free tier support for both judges (no billing required)

**v2.0 (2026-02-21)** - Major improvements based on rigorous review:
- Added hard constraints to prevent generous scoring
- Discriminating evaluation lenses (ABCD vs SMART distinction)
- Split Bloom's into individual vs set-level criteria
- Binary checklists embedded in prompts
- System prompts for rubric internalization
- Mode reporting alongside mean
- Complete calibration system with Cohen's Kappa
- Calibration template and tooling

**v1.0 (2026-02-07)** - Initial release:
- Basic rubrics for ABCD, SMART, Bloom's
- Multiple runs for consistency
- Report generation

## Files Reference

```
src/evaluation/
├── llm_judge_evaluation.py       # Main evaluation script (UPDATED)
├── generate_evaluation_report.py # Report generator
├── calibration_analysis.py       # Human-LLM agreement analysis (NEW)
├── README.md                      # This file (UPDATED)
└── RUBRIC_REFERENCE.md           # Quick reference guide

datasets/evaluation/
├── evaluation_abcd.json                    # LLM evaluation results
├── evaluation_smart.json
├── evaluation_blooms.json
├── calibration_set_TEMPLATE.json          # Template for manual scoring (NEW)
├── calibration_set.json                    # Your manual scores (you create)
└── reports/
    ├── report_abcd.txt
    ├── report_smart.txt
    ├── report_blooms.txt
    ├── report_summary.txt
    └── calibration_report.txt              # Human-LLM agreement (NEW)
```

## Quick Command Reference

```bash
# Full evaluation pipeline
python llm_judge_evaluation.py      # Evaluate all LOs (3 runs each)
python generate_evaluation_report.py # Create readable reports

# Calibration workflow
cp datasets/evaluation/calibration_set_TEMPLATE.json \
   datasets/evaluation/calibration_set.json
# Edit calibration_set.json with your manual scores
python calibration_analysis.py      # Compute agreement metrics

# Check outputs
cat datasets/evaluation/reports/report_summary.txt
cat datasets/evaluation/reports/calibration_report.txt

# Verify for paper readiness
# Look for: Kappa ≥ 0.40, Within-1 ≥ 60%, StdDev ≤ 0.5
```

## Key Takeaways for Papers

1. **Always report mode alongside mean** when scores are skewed
2. **Run calibration** with 5-10 manually scored LOs before claiming LLM-judge validity
3. **Report Cohen's Kappa and within-±1 agreement** as inter-rater reliability metrics
4. **Acknowledge limitations** (systematic bias, criteria with high StdDev)
5. **Explain hard constraints** in methodology (prevents generous scoring)
6. **Justify discriminating lenses** (why ABCD ≠ SMART despite overlap)

## License

MIT License - see LICENSE file for details
