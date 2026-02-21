# LLM-as-Judge Framework v2.0: Key Improvements

## Summary of Changes (Based on Rigorous Review)

This document outlines the improvements made to address critical issues in the original evaluation framework.

---

## 1. Fixed Score Granularity Inconsistencies ✅

### Problem
Original rubric had contradictions:
- ABCD Behavior gave "understand" a score of 4/5
- Bloom's section correctly flagged "understand" as weak/vague
- LLM received conflicting signals → inconsistent scoring

### Solution
**Hard Constraints** explicitly coded into rubrics:

```python
# ABCD Framework
"If the LO uses 'understand', 'know', 'learn' WITHOUT clarifying 
 HOW it will be demonstrated → Behavior score ≤ 2"

# SMART Framework  
"If the LO uses 'understand', 'know', 'appreciate' 
 → Measurable score ≤ 2 (cannot assess objectively)"

# Bloom's Taxonomy
"If verb is 'understand', 'know', 'learn' 
 → MUST be classified as Remember/Understand (low level), NOT higher"
```

**Impact**: Prevents LLM from being generous when rubric-violating patterns appear. Scores are now consistent with pedagogical best practices.

---

## 2. Distinguished ABCD vs SMART Evaluation Lenses ✅

### Problem
- ABCD "Behavior" and SMART "Measurable" evaluated nearly the same thing
- Produced correlated scores without adding signal
- Redundant evaluation wasted API calls and analysis time

### Solution
**Explicit Evaluation Lenses** in system prompts:

| Framework | Lens | Focus | Key Question |
|-----------|------|-------|--------------|
| **ABCD** | Structural Completeness | Are all 4 components present and well-specified? | "Is the behavior OBSERVABLE?" |
| **SMART** | Practical Feasibility | Can we create concrete grading criteria? | "Can we MEASURE SUCCESS with metrics?" |

**Key Distinction**:
- **ABCD Behavior**: Checks if verb describes an action (yes/no structural check)
- **SMART Measurable**: Checks if success can be assessed with rubrics/criteria (grading feasibility)

**Example**:
- LO: "Students will explain process scheduling"
- ABCD Behavior: 4/5 (✓ "explain" is observable)
- SMART Measurable: 3/5 (⚠️ how do you measure "explanation quality"? Needs rubric)

**Impact**: Frameworks now provide non-redundant information. ABCD checks structure, SMART checks assessment design.

---

## 3. Separated Individual vs Set-Level Bloom's Criteria ✅

### Problem
Original design scored individual LOs on "progression" - but progression is a property of the complete set, not a single LO. This was logically impossible and produced meaningless scores.

### Solution
Split Bloom's evaluation into TWO phases:

**Phase 1: INDIVIDUAL LO EVALUATION** (per LO)
- Verb Accuracy: Does verb match cognitive level?
- Cognitive Demand: Does task match verb complexity?  
- Level Classification: Can you classify unambiguously?

**Phase 2: SET-LEVEL EVALUATION** (all LOs together)
- Progression: Do LOs build Remember → Apply → Analyze → Create?
- Coverage: Span multiple Bloom's levels appropriately?
- Prerequisites: Are lower levels prerequisites for higher?

**Impact**: Scoring now logically sound. Individual LOs scored on individual properties; set scored on set properties.

---

## 4. Embedded Binary Checklists in Prompts ✅

### Problem
- Original prompts let LLM jump straight to holistic scores
- Skipped systematic analysis
- Critical dimensions could be overlooked

### Solution
**Five YES/NO questions** that LLM must answer BEFORE scoring:

**ABCD Example**:
```
[ ] Can you identify WHO is learning without making assumptions?
[ ] Does the verb describe an OBSERVABLE action (not a mental state)?
[ ] Are the CONDITIONS/CONTEXT stated or unambiguously implied?
[ ] Is there a STANDARD (explicit or clear implicit) that defines success?
[ ] Could you create an exam question/rubric to assess this objectively?
```

**Placement**: Embedded in user prompt, answered before scores assigned.

**Impact**: Forces disciplined evaluation. LLM can't skip dimensions or rely on intuition.

---

## 5. System Prompts for Rubrics (Better Internalization) ✅

### Problem
Original design sent rubric + LO together as user prompt. LLM treated them as equal-weight information, didn't "internalize" the rubric as evaluation standards.

### Solution
**Separate prompts**:
- **System prompt**: Contains rubric, hard constraints, evaluation lens
- **User prompt**: Contains LO to evaluate and specific instructions

```python
system_prompt = f"""You are an expert educational assessment specialist.
{ABCD_RUBRIC}  # Rubric internalized as "your role"
Apply this rubric consistently."""

user_prompt = f"""Evaluate this LO: "{learning_objective}" """

call_ollama_api(user_prompt, system_prompt=system_prompt)
```

**Impact**: LLM "internalizes" rubric before seeing LO. Improves consistency and adherence to standards.

---

## 6. Mode Reporting Alongside Mean ✅

### Problem
If scores across 3 runs are: 3, 3, 4
- Mean = 3.33 (suggests "between 3 and 4")
- Reality: LLM chose 3 twice, 4 once → mode = 3 is more honest

### Solution
Report **mean, mode, and stdev** for all scores:

```json
{
  "composite_score_mean": 3.33,
  "composite_score_mode": 3,      ← More honest representation
  "composite_score_stdev": 0.47
}
```

**Impact**: 
- For papers: More accurate reporting ("modal score was 3")
- For debugging: Identifies skewed distributions vs. true variability

---

## 7. Calibration System for Validation ✅

### Problem
No way to validate if LLM scores matched expert human judgment. Critical gap for justifying LLM-as-judge in papers.

### Solution
**Complete Calibration Workflow**:

1. **Manual Scoring**: Use `calibration_set_TEMPLATE.json` to score 5-10 LOs by hand
2. **LLM Evaluation**: Run LLM on same LOs  
3. **Agreement Analysis**: `calibration_analysis.py` computes:
   - Exact agreement %
   - Within-±1 agreement % (practical threshold)
   - **Cohen's Kappa** (inter-rater reliability)
   - Mean bias (LLM too lenient/strict?)
   - Pearson correlation

**Paper-Ready Thresholds**:
```
Cohen's Kappa ≥ 0.40  (moderate agreement, acceptable for publication)
Within-±1 ≥ 60%       (practical reliability threshold)
```

**Output**: Detailed calibration report with interpretation and recommendations.

**Impact**: 
- Validates LLM judge scientifically
- Provides metrics required for paper methodology sections
- Identifies systematic biases (e.g., "LLM scores 0.8 points higher on average")

---

## Comparison Table: v1.0 vs v2.0

| Feature | v1.0 | v2.0 | Impact |
|---------|------|------|--------|
| **Score Constraints** | Soft ("score 1-5 based on quality") | Hard ("if X, score ≤Y") | Prevents generous scoring |
| **Framework Distinction** | Implicit | Explicit evaluation lenses | Non-redundant information |
| **Bloom's Progression** | Scored per LO (illogical) | Scored at set level (logical) | Meaningful scores |
| **Evaluation Process** | Jump to scores | Binary checklist first | Disciplined analysis |
| **Rubric Delivery** | User prompt | System prompt | Better internalization |
| **Score Reporting** | Mean ± stdev | Mean, mode, stdev | Honest representation |
| **Validation** | None | Full calibration system | Scientific rigor |

---

## Remaining Best Practices

### For Evaluation Runs
1. Use temperature = 0.1-0.3 (deterministic but not robotic)
2. Run 3-5 times per LO for consistency check
3. Report StdDev; flag if > 0.5 (inconsistent)

### For Paper Reporting
1. **Always include**:
   - Mean ± StdDev (mode) for all scores
   - Cohen's Kappa from calibration
   - Within-±1 agreement %
   
2. **Be honest about**:
   - Systematic bias (if mean bias ≠ 0)
   - Criteria with high StdDev (ambiguous)
   - Kappa interpretation (use Landis & Koch standards)

3. **Justify approach**:
   - Why LLM-as-judge? (scalability, consistency potential)
   - How validated? (calibration with human expert)
   - Limitations? (acknowledge bias, variability)

---

## When to Iterate

Re-run evaluation and calibration if:
- **Cohen's Kappa < 0.40** → Add harder constraints, clarify rubric
- **Mean bias > ±0.5** → LLM is systematically too lenient/strict
- **StdDev > 0.5** for any criterion → Rubric ambiguity, needs clarification
- **Within-±1 < 60%** → Poor practical reliability, unsatisfactory for publication

---

## Files Changed in v2.0

### Updated
- `llm_judge_evaluation.py`: Hard constraints, system prompts, mode calculation
- `README.md`: Complete rewrite with troubleshooting, paper guidelines
- `RUBRIC_REFERENCE.md`: Fixed score inconsistencies, added hard constraint tables

### New
- `calibration_analysis.py`: Cohen's Kappa, agreement metrics, bias detection
- `calibration_set_TEMPLATE.json`: Template for manual scoring
- `IMPROVEMENTS_v2.md`: This document

---

## Quick Self-Check: Is Your Evaluation Paper-Ready?

✅ **YES** if all true:
- [ ] Cohen's Kappa ≥ 0.40
- [ ] Within-±1 agreement ≥ 60%  
- [ ] StdDev ≤ 0.5 for all major criteria
- [ ] Systematic bias < ±0.5 points
- [ ] Hard constraints prevent "understand" from scoring > 2

❌ **NO** if any true:
- [ ] Kappa < 0.40 (poor inter-rater reliability)
- [ ] Within-±1 < 60% (unreliable in practice)
- [ ] StdDev > 1.0 for any criterion (highly inconsistent)
- [ ] No calibration performed (cannot validate approach)

**If NO**: Iterate on rubrics, add constraints, re-calibrate. Do not publish without validation.

---

## Citation for v2.0

```bibtex
@misc{llm_judge_lo_eval_v2,
  title={LLM-as-Judge Evaluation Framework for Learning Objectives (v2.0): 
         Hard Constraints, Discriminating Lenses, and Calibration Validation},
  author={Your Name},
  year={2026},
  note={Addresses score inconsistencies, framework redundancy, and validation gaps 
        in educational LO assessment using LLMs. Includes Cohen's Kappa calibration.}
}
```

---

**Version**: 2.0  
**Date**: February 21, 2026  
**Status**: Production-ready for research publication
