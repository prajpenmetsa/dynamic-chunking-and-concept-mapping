# v2.1 Update: Dual-Judge Architecture

## Summary

The evaluation framework has been upgraded to use **two independent LLM judges** for increased methodological rigor.

## Changes

### 1. Dual Judges
- **Primary Judge**: Gemini 2.0 Flash (proprietary, excellent structured output)
- **Validation Judge**: Llama 3.3 70B via Groq (open-source, reproducible)

### 2. Why This Matters for Your Paper
- **Reproducibility**: Open-source judge means reviewers can replicate results
- **Validity**: Agreement between two judges from different providers strengthens claims
- **Robustness**: Demonstrates that scores aren't model-specific artifacts
- **Free tier**: Both judges have generous free tiers (no billing required)

### 3. Inter-Judge Agreement Metrics

The framework now calculates:
- **Exact agreement %**: How often judges give identical scores
- **Within-±1 agreement %**: How often judges are within 1 point (key metric for papers)
- **Mean bias**: Average difference between judges (Gemini - Llama)
- **Correlation**: Pearson r correlation between judge scores
- **Cohen's Kappa**: Inter-rater reliability (for criterion-level comparisons)

### 4. API Keys Required

You need **both** API keys in your `.env` file:

```bash
# Get at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_key_here

# Get at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_key_here
```

See `.env.example` for template.

### 5. Running Evaluations

```bash
# Same command as before
python llm_judge_evaluation.py
```

**What's different**:
- Each LO is evaluated by BOTH judges (Gemini + Llama)
- Still 3 runs per judge for consistency
- Total: 6 evaluations per LO (3 Gemini + 3 Llama)
- Runtime: ~20-30 minutes (was ~10-15 minutes)

### 6. Output Format

New JSON structure:

```json
{
  "framework": "ABCD",
  "evaluations": [
    {
      "learning_objective": "...",
      "gemini_evaluation": {
        "evaluation_runs": [run1, run2, run3],
        "consistency_analysis": {...}
      },
      "groq_evaluation": {
        "evaluation_runs": [run1, run2, run3],
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
  "overall_inter_judge_agreement": {...},
  "metadata": {
    "primary_judge": "gemini-2.0-flash-exp",
    "validation_judge": "llama-3.3-70b-versatile"
  }
}
```

### 7. Reporting for Your Paper

**Methodology section**:
> We employed a dual-judge LLM evaluation approach using Gemini 2.0 Flash (proprietary) and Llama 3.3 70B via Groq (open-source). Each LO was evaluated independently 3 times by each judge...

**Results section** - Report both judges:
```
ABCD Framework:
Gemini avg: 3.2 ± 0.4
Llama avg: 3.4 ± 0.3
Inter-judge agreement: 85% within-±1 (r = 0.89)
```

**Interpretation thresholds**:
- Within-±1 ≥80% → "Strong inter-judge consensus"
- Within-±1 60-80% → "Moderate agreement"
- Within-±1 <60% → "Weak agreement, rubric refinement needed"

### 8. Backwards Compatibility

**NOT compatible** with v2.0 evaluation outputs. If you have existing evaluation results from v2.0, they used a single judge (Ollama) and won't work with v2.1 scripts.

**Solution**: Re-run evaluations with v2.1 to get dual-judge results.

### 9. Report Generation

The `generate_evaluation_report.py` script needs updating to handle the new dual-judge structure. Current version expects old format.

**Status**: Report generation will be updated after initial dual-judge evaluations complete. For now, you can:
1. Inspect JSON files directly in `datasets/evaluation/`
2. Write custom analysis scripts
3. Wait for report generation update

### 10. Calibration System

The human-LLM calibration system (`calibration_analysis.py`) now supports comparing:
- **Human vs Gemini**
- **Human vs Llama**
- **Gemini vs Llama** (inter-judge)

You can generate calibration stats for all three comparisons.

## Quick Start

```bash
# 1. Set up API keys
cp .env.example .env
# Edit .env with your actual keys

# 2. Run dual-judge evaluation
python llm_judge_evaluation.py

# 3. Check inter-judge agreement in terminal output
# Look for "INTER-JUDGE AGREEMENT SUMMARY"

# 4. Inspect JSON results
cat ../../datasets/evaluation/evaluation_abcd.json | jq '.overall_inter_judge_agreement'
```

## Expected Console Output

```
======================================================================
  LLM-AS-JUDGE EVALUATION FRAMEWORK v2.1
  Dual Judges: Gemini 2.0 Flash + Llama 3.3 70B (Groq)
======================================================================

✓ API keys found for both judges
  Primary: gemini-2.0-flash-exp
  Validation: llama-3.3-70b-versatile

======================================================================
  EVALUATING ABCD LEARNING OBJECTIVES
  Dual Judges: Gemini 2.0 Flash + Llama 3.3 70B (Groq)
======================================================================

📚 Found 7 learning objectives
🔄 Running 3 evaluation rounds × 2 judges for inter-judge agreement

  LO 1/7: Design and implement schedulers...
    Run 1/3:
      Gemini... ✓ Score: 3.50
      Groq... ✓ Score: 3.75
    Run 2/3:
      Gemini... ✓ Score: 3.25
      Groq... ✓ Score: 3.50
    ...

======================================================================
  INTER-JUDGE AGREEMENT SUMMARY - ABCD
======================================================================
  Exact Agreement: 45.2%
  Within-±1 Agreement: 92.3%
  Mean Bias (Gemini - Llama): -0.18
  Correlation: r = 0.874

  Gemini Avg: 3.15 ± 0.42
  Llama Avg: 3.33 ± 0.38

  ✅ Strong inter-judge agreement (≥80% within ±1)
```

## Troubleshooting

### "GEMINI_API_KEY not found"
- Create `.env` file from `.env.example`
- Add your Gemini API key from https://aistudio.google.com/app/apikey

### "GROQ_API_KEY not found"
- Add your Groq API key from https://console.groq.com/keys

### "Rate limit exceeded"
- Both services have generous free tiers
- Script includes automatic retry logic with backoff
- If you hit rate limits, wait a few minutes and re-run

### "Low inter-judge agreement (<60%)"
- This is a **finding**, not a bug
- Check which criteria have lowest agreement
- Consider rubric refinement for ambiguous criteria
- Document in paper: "Inter-judge disagreement on X suggests..."

### "Report generation fails"
- Expected: report script not updated yet for v2.1
- Use JSON files directly or wait for report script update
- You can extract key metrics with `jq`:
  ```bash
  jq '.overall_inter_judge_agreement' evaluation_abcd.json
  ```

## Next Steps

1. **Run initial evaluation**: Get dual-judge results for all frameworks
2. **Check agreement**: Look at inter-judge agreement percentages
3. **Manual calibration**: Score 5-10 LOs yourself, run `calibration_analysis.py`
4. **Compare all three**: Human vs Gemini, Human vs Llama, Gemini vs Llama
5. **Report in paper**: Use templates from README.md

## Questions?

See [README.md](README.md) for:
- Full methodology documentation
- Paper reporting guidelines
- Calibration workflow
- Troubleshooting guide
