# LLM-as-Judge Evaluation Framework

A comprehensive evaluation system for assessing learning objectives against **ABCD Framework**, **SMART Criteria**, and **Bloom's Taxonomy** using LLM-based rubrics.

## Overview

This framework provides:
- **Detailed rubrics** with 1-5 scoring scales for each taxonomy criterion
- **Multiple evaluation runs** (default: 3) to assess consistency
- **Granular questions** that probe specific aspects of each framework
- **Automated report generation** with human-readable analysis

## Features

### 1. Highly Specific Rubrics

Each framework has detailed scoring criteria:

#### ABCD Framework (4 components)
- **A**udience: Who will learn? (1-5 scale)
- **B**ehavior: What will they do? (observable, measurable verbs)
- **C**ondition: Under what circumstances?
- **D**egree: How well must they perform?

#### SMART Framework (5 components)
- **S**pecific: Focused and clear content
- **M**easurable: Can achievement be assessed objectively?
- **A**chievable: Realistic for target learners
- **R**elevant: Aligned with course goals
- **T**ime-bound: Clear timeframe (explicit or implicit)

#### Bloom's Taxonomy (3 aspects per LO + set-level)
- **Verb Accuracy**: Does verb match cognitive level?
- **Cognitive Demand**: Does task match verb complexity?
- **Level Classification**: Can you unambiguously classify the level?
- **Progression**: Do LOs build from Remember → Create?

### 2. Granular Evaluation Questions

Each criterion has specific questions with clear scales:

**Example (ABCD - Behavior)**:
```
Question: "Does the LO use an observable, measurable action verb 
           (not vague terms like 'understand' or 'know')?"
Scale: 1=Non-behavioral/passive, 
       3=Vague action verb, 
       5=Clear measurable verb
```

### 3. Consistency Checking

Each evaluation runs **3 times** (configurable) to measure:
- **Mean scores** across runs
- **Standard deviation** (low = consistent)
- **Score ranges** to identify variability

### 4. Detailed Evidence & Justification

For every score, the LLM provides:
- **Evidence**: Specific quotes or observations
- **Weakness**: What's missing or could improve
- **Justification**: Reasoning for the score

## Installation

```bash
# Install requirements
pip install requests python-dotenv

# Set up environment
echo "OLLAMA_API_KEY=your_api_key_here" >> .env
```

## Usage

### Step 1: Run Evaluation

```bash
cd src/evaluation
python llm_judge_evaluation.py
```

This will:
1. Load learning objectives from:
   - `datasets/slide_based_los_simple_abcd.json`
   - `datasets/slide_based_los_simple_smart.json`
   - `datasets/slide_based_los_simple_blooms.json`

2. Evaluate each LO 3 times for consistency

3. Save JSON results to `datasets/evaluation/`

**Expected runtime**: ~5-10 minutes per framework (depending on number of LOs)

### Step 2: Generate Reports

```bash
python generate_evaluation_report.py
```

This creates human-readable reports:
- `reports/report_abcd.txt` - Detailed ABCD analysis
- `reports/report_smart.txt` - Detailed SMART analysis
- `reports/report_blooms.txt` - Detailed Bloom's analysis
- `reports/report_summary.txt` - Cross-framework comparison

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

## Configuration

Edit `llm_judge_evaluation.py` to customize:

```python
# Number of evaluation runs for consistency
NUM_EVALUATION_RUNS = 3  # Increase for more rigorous consistency testing

# Model temperature (lower = more consistent)
temperature = 0.3  # Range: 0.0-1.0

# Rate limiting
RATE_LIMIT_DELAY = 5  # seconds between API calls
```

## Use Cases

### 1. Quality Assurance
Evaluate generated LOs before course deployment:
```bash
python llm_judge_evaluation.py
python generate_evaluation_report.py
# Review reports/report_summary.txt for overall quality
```

### 2. Iterative Improvement
Run evaluation → refine LOs → re-evaluate:
```bash
# After modifying LOs in datasets/
python llm_judge_evaluation.py
# Compare new scores with previous run
```

### 3. Framework Comparison
Compare which framework produces highest-quality LOs:
```bash
# Check report_summary.txt for cross-framework scores
cat datasets/evaluation/reports/report_summary.txt
```

### 4. Consistency Testing
Test if evaluation is reliable:
```bash
# Increase NUM_EVALUATION_RUNS to 5-10
# Check if stdev remains low (<0.5)
```

## Rubric Philosophy

### Design Principles

1. **Specificity over Generality**
   - Each score level has concrete examples
   - Clear distinctions between adjacent scores
   - No ambiguous terms like "somewhat" or "fairly"

2. **Observable Evidence Required**
   - LLM must cite specific text from LO
   - Can't give high scores without justification
   - Forces granular analysis

3. **Granular Questions**
   - Break each criterion into testable questions
   - Binary or scale-based answers
   - Reduces bias by constraining evaluation

4. **Multiple Runs**
   - Detect evaluation inconsistency
   - Average out stochastic variation
   - Provide confidence intervals

### Critical Evaluation Questions

Each rubric includes questions that probe framework adherence:

**ABCD Critical Questions:**
1. Can you identify WHO is learning without ambiguity?
2. Is the behavior something you can OBSERVE and MEASURE?
3. Are the CONDITIONS/CONTEXT under which the behavior occurs clear?
4. Is there a STANDARD that defines successful achievement?
5. Could this LO be assessed objectively in an exam?

**SMART Critical Questions:**
1. If you gave this to a student, would they know EXACTLY what to learn?
2. Can you create an exam question that measures this objectively?
3. Is this achievable within a single semester?
4. Would a course instructor agree this belongs in THIS course?
5. Is there a clear timeframe for achievement?

**Bloom's Critical Questions:**
1. Does the verb match the actual cognitive demand?
2. Could you classify this into a single level without ambiguity?
3. Does the task description match verb complexity?
4. Do LOs progress from foundational to advanced?
5. Are lower-level outcomes prerequisites for higher-level ones?

## Troubleshooting

### Issue: API Rate Limits
```bash
# Increase delay between calls
RATE_LIMIT_DELAY = 10  # in llm_judge_evaluation.py
```

### Issue: Inconsistent Scores (high stdev)
```bash
# Lower temperature for more deterministic outputs
temperature = 0.1  # in call_ollama_api()

# Or increase evaluation runs
NUM_EVALUATION_RUNS = 5
```

### Issue: JSON Parsing Errors
The script includes robust JSON parsing that handles:
- Markdown code blocks (```json ... ```)
- Preamble text ("Here's my evaluation...")
- Malformed JSON (attempts to extract valid portions)

If issues persist, check raw API responses in evaluation JSON files.

## Citation

If you use this evaluation framework in research, please cite:

```bibtex
@misc{llm_judge_lo_eval,
  title={LLM-as-Judge Evaluation Framework for Learning Objectives},
  author={Your Name},
  year={2026},
  note={Rubric-based evaluation system for ABCD, SMART, and Bloom's Taxonomy}
}
```

## Future Enhancements

Potential additions:
- [ ] Inter-rater reliability with multiple models
- [ ] Automated LO refinement based on weak scores
- [ ] Visualization of score distributions
- [ ] Comparative analysis across courses
- [ ] Integration with LMS systems

## License

MIT License - see LICENSE file for details
