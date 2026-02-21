"""
Calibration Script: Human-LLM Agreement Analysis
Allows manual scoring of LOs, then compares with LLM judge scores
"""

import json
import os
from typing import Dict, List
import statistics

# ==================== CONFIGURATION ====================
CALIBRATION_SET_FILE = "../../datasets/evaluation/calibration_set.json"
LLM_EVALUATION_DIR = "../../datasets/evaluation"
CALIBRATION_REPORT = "../../datasets/evaluation/reports/calibration_report.txt"


# ==================== CALIBRATION SET STRUCTURE ====================
"""
calibration_set.json format:
{
  "calibration_set": [
    {
      "lo_id": "ABCD_LO_1",
      "framework": "ABCD",
      "learning_objective": "Students will analyze...",
      "human_scores": {
        "audience": 5,
        "behavior": 4,
        "condition": 3,
        "degree": 2
      },
      "human_notes": "Clear audience and behavior, but missing explicit conditions..."
    },
    {
      "lo_id": "SMART_LO_1",
      "framework": "SMART",
      "learning_objective": "...",
      "human_scores": {
        "specific": 4,
        "measurable": 3,
        "achievable": 5,
        "relevant": 5,
        "time_bound": 4
      },
      "human_notes": "..."
    }
  ]
}
"""


# ==================== AGREEMENT METRICS ====================

def calculate_exact_agreement(human_scores: List[int], llm_scores: List[int]) -> float:
    """Percentage of scores that match exactly."""
    if len(human_scores) != len(llm_scores):
        raise ValueError("Score lists must be same length")
    
    matches = sum(1 for h, l in zip(human_scores, llm_scores) if h == l)
    return (matches / len(human_scores)) * 100


def calculate_within_one_agreement(human_scores: List[int], llm_scores: List[int]) -> float:
    """Percentage of scores within ±1 point."""
    if len(human_scores) != len(llm_scores):
        raise ValueError("Score lists must be same length")
    
    within_one = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1)
    return (within_one / len(human_scores)) * 100


def calculate_mean_absolute_error(human_scores: List[int], llm_scores: List[int]) -> float:
    """Average absolute difference between human and LLM scores."""
    if len(human_scores) != len(llm_scores):
        raise ValueError("Score lists must be same length")
    
    return statistics.mean(abs(h - l) for h, l in zip(human_scores, llm_scores))


def calculate_correlation(human_scores: List[int], llm_scores: List[int]) -> float:
    """Pearson correlation coefficient."""
    if len(human_scores) != len(llm_scores) or len(human_scores) < 2:
        return 0.0
    
    n = len(human_scores)
    
    # Calculate means
    mean_h = statistics.mean(human_scores)
    mean_l = statistics.mean(llm_scores)
    
    # Calculate correlation
    numerator = sum((h - mean_h) * (l - mean_l) for h, l in zip(human_scores, llm_scores))
    denom_h = sum((h - mean_h) ** 2 for h in human_scores)
    denom_l = sum((l - mean_l) ** 2 for l in llm_scores)
    
    if denom_h == 0 or denom_l == 0:
        return 0.0
    
    return numerator / (denom_h * denom_l) ** 0.5


def calculate_cohens_kappa(human_scores: List[int], llm_scores: List[int], num_categories: int = 5) -> float:
    """Cohen's Kappa for inter-rater reliability (accounts for chance agreement)."""
    if len(human_scores) != len(llm_scores):
        raise ValueError("Score lists must be same length")
    
    n = len(human_scores)
    
    # Observed agreement
    observed = sum(1 for h, l in zip(human_scores, llm_scores) if h == l) / n
    
    # Expected agreement by chance
    human_dist = [human_scores.count(i) / n for i in range(1, num_categories + 1)]
    llm_dist = [llm_scores.count(i) / n for i in range(1, num_categories + 1)]
    expected = sum(h * l for h, l in zip(human_dist, llm_dist))
    
    if expected == 1:
        return 1.0  # Perfect agreement
    
    return (observed - expected) / (1 - expected)


# ==================== ANALYSIS FUNCTIONS ====================

def load_calibration_set() -> List[Dict]:
    """Load human-scored calibration set."""
    if not os.path.exists(CALIBRATION_SET_FILE):
        print(f"❌ Calibration set not found: {CALIBRATION_SET_FILE}")
        print("   Create calibration_set.json with manually scored LOs first.")
        return []
    
    with open(CALIBRATION_SET_FILE, 'r') as f:
        data = json.load(f)
    
    return data.get("calibration_set", [])


def find_llm_scores(lo_id: str, framework: str) -> Dict:
    """Find corresponding LLM evaluation scores."""
    eval_file = os.path.join(LLM_EVALUATION_DIR, f"evaluation_{framework.lower()}.json")
    
    if not os.path.exists(eval_file):
        return None
    
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    
    # Extract LO number from ID (e.g., "ABCD_LO_1" -> 1)
    try:
        lo_num = int(lo_id.split("_")[-1])
    except:
        return None
    
    # Find matching LO in evaluation data
    if framework == "BLOOMS":
        # Blooms has different structure
        for run in eval_data.get("evaluation_runs", []):
            for lo_eval in run.get("individual_evaluations", []):
                if lo_eval["objective_number"] == lo_num:
                    # Use first run's scores
                    return lo_eval
    else:
        # ABCD/SMART structure
        for eval_item in eval_data.get("evaluations", []):
            if eval_item["objective_number"] == lo_num:
                # Use first run's scores
                return eval_item["evaluation_runs"][0]
    
    return None


def compare_scores(calibration_item: Dict) -> Dict:
    """Compare human vs LLM scores for a single LO."""
    framework = calibration_item["framework"]
    lo_id = calibration_item["lo_id"]
    human_scores = calibration_item["human_scores"]
    
    # Find LLM scores
    llm_eval = find_llm_scores(lo_id, framework)
    
    if not llm_eval:
        return {
            "error": "LLM evaluation not found",
            "lo_id": lo_id
        }
    
    # Extract LLM scores based on framework
    if framework in ["ABCD", "SMART"]:
        llm_scores = {}
        for criterion, data in llm_eval.get("overall_scores", {}).items():
            llm_scores[criterion] = data.get("score", 0)
    elif framework == "BLOOMS":
        llm_scores = {}
        for criterion, data in llm_eval.get("scores", {}).items():
            llm_scores[criterion] = data.get("score", 0)
    
    # Calculate differences
    comparison = {
        "lo_id": lo_id,
        "framework": framework,
        "learning_objective": calibration_item["learning_objective"][:100] + "...",
        "criterion_comparisons": {},
        "human_composite": statistics.mean(human_scores.values()),
        "llm_composite": statistics.mean(llm_scores.values()) if llm_scores else 0
    }
    
    # Compare each criterion
    for criterion in human_scores.keys():
        human_score = human_scores[criterion]
        llm_score = llm_scores.get(criterion, 0)
        difference = llm_score - human_score
        
        comparison["criterion_comparisons"][criterion] = {
            "human": human_score,
            "llm": llm_score,
            "difference": difference,
            "agreement": "exact" if difference == 0 else "within_1" if abs(difference) <= 1 else "divergent"
        }
    
    return comparison


def analyze_calibration_set(calibration_set: List[Dict]) -> Dict:
    """Analyze entire calibration set for human-LLM agreement."""
    
    comparisons = []
    all_human_scores = []
    all_llm_scores = []
    
    for item in calibration_set:
        comparison = compare_scores(item)
        
        if "error" in comparison:
            print(f"⚠️  Skipping {comparison['lo_id']}: {comparison['error']}")
            continue
        
        comparisons.append(comparison)
        
        # Collect all scores for aggregate metrics
        for criterion, data in comparison["criterion_comparisons"].items():
            all_human_scores.append(data["human"])
            all_llm_scores.append(data["llm"])
    
    if not comparisons:
        return {"error": "No valid comparisons found"}
    
    # Calculate aggregate metrics
    exact_agreement = calculate_exact_agreement(all_human_scores, all_llm_scores)
    within_one = calculate_within_one_agreement(all_human_scores, all_llm_scores)
    mae = calculate_mean_absolute_error(all_human_scores, all_llm_scores)
    correlation = calculate_correlation(all_human_scores, all_llm_scores)
    kappa = calculate_cohens_kappa(all_human_scores, all_llm_scores)
    
    # LLM bias analysis (is LLM systematically higher/lower?)
    differences = [l - h for h, l in zip(all_human_scores, all_llm_scores)]
    mean_bias = statistics.mean(differences)
    
    return {
        "num_comparisons": len(comparisons),
        "total_criteria_compared": len(all_human_scores),
        "aggregate_metrics": {
            "exact_agreement_pct": round(exact_agreement, 2),
            "within_one_agreement_pct": round(within_one, 2),
            "mean_absolute_error": round(mae, 2),
            "pearson_correlation": round(correlation, 3),
            "cohens_kappa": round(kappa, 3),
            "mean_bias": round(mean_bias, 2)  # Positive = LLM scores higher
        },
        "individual_comparisons": comparisons,
        "interpretation": generate_interpretation(exact_agreement, within_one, kappa, mean_bias)
    }


def generate_interpretation(exact_pct: float, within_one_pct: float, kappa: float, bias: float) -> Dict:
    """Generate interpretation of agreement metrics."""
    
    # Kappa interpretation (Landis & Koch, 1977)
    if kappa < 0:
        kappa_interp = "Poor (worse than chance)"
    elif kappa < 0.20:
        kappa_interp = "Slight"
    elif kappa < 0.40:
        kappa_interp = "Fair"
    elif kappa < 0.60:
        kappa_interp = "Moderate"
    elif kappa < 0.80:
        kappa_interp = "Substantial"
    else:
        kappa_interp = "Almost Perfect"
    
    # Agreement interpretation
    if within_one_pct >= 80:
        agreement_quality = "Excellent (most scores within ±1)"
    elif within_one_pct >= 60:
        agreement_quality = "Good (majority within ±1)"
    elif within_one_pct >= 40:
        agreement_quality = "Moderate (some disagreement)"
    else:
        agreement_quality = "Poor (significant divergence)"
    
    # Bias interpretation
    if abs(bias) < 0.3:
        bias_interp = "No systematic bias"
    elif bias > 0:
        bias_interp = f"LLM scores {abs(bias):.1f} points higher on average (lenient)"
    else:
        bias_interp = f"LLM scores {abs(bias):.1f} points lower on average (strict)"
    
    return {
        "kappa_interpretation": kappa_interp,
        "agreement_quality": agreement_quality,
        "bias_interpretation": bias_interp,
        "suitable_for_paper": within_one_pct >= 60 and kappa >= 0.40
    }


# ==================== REPORT GENERATION ====================

def generate_calibration_report(analysis: Dict) -> str:
    """Generate human-readable calibration report."""
    
    if "error" in analysis:
        return f"❌ Error: {analysis['error']}"
    
    report = []
    report.append("="*80)
    report.append("  CALIBRATION REPORT: HUMAN-LLM AGREEMENT ANALYSIS")
    report.append("="*80)
    
    metrics = analysis["aggregate_metrics"]
    interp = analysis["interpretation"]
    
    report.append(f"\n📊 AGREEMENT METRICS (n={analysis['total_criteria_compared']} criterion scores):")
    report.append(f"   Exact Agreement:        {metrics['exact_agreement_pct']}%")
    report.append(f"   Within ±1 Agreement:    {metrics['within_one_agreement_pct']}%")
    report.append(f"   Mean Absolute Error:    {metrics['mean_absolute_error']}")
    report.append(f"   Pearson Correlation:    {metrics['pearson_correlation']}")
    report.append(f"   Cohen's Kappa:          {metrics['cohens_kappa']} ({interp['kappa_interpretation']})")
    report.append(f"   Mean Bias:              {metrics['mean_bias']:+.2f} ({interp['bias_interpretation']})")
    
    report.append(f"\n💡 INTERPRETATION:")
    report.append(f"   Agreement Quality: {interp['agreement_quality']}")
    report.append(f"   Suitable for Paper: {'✓ YES' if interp['suitable_for_paper'] else '❌ NO - needs improvement'}")
    
    if interp['suitable_for_paper']:
        report.append(f"\n   ✓ Cohen's Kappa ≥ 0.40 (acceptable inter-rater reliability)")
        report.append(f"   ✓ Within-one agreement ≥ 60% (good practical agreement)")
    else:
        report.append(f"\n   ⚠️  Consider: Re-prompting with harder constraints, or manual review of divergent cases")
    
    # Individual comparisons
    report.append(f"\n\n{'='*80}")
    report.append("INDIVIDUAL LO COMPARISONS")
    report.append("="*80)
    
    for comp in analysis["individual_comparisons"]:
        report.append(f"\n\n{'─'*80}")
        report.append(f"LO ID: {comp['lo_id']} ({comp['framework']})")
        report.append(f"{'─'*80}")
        report.append(f"\n{comp['learning_objective']}")
        report.append(f"\nComposite Scores: Human={comp['human_composite']:.2f}, LLM={comp['llm_composite']:.2f}")
        
        report.append(f"\n\nCriterion-by-Criterion:")
        for criterion, data in comp["criterion_comparisons"].items():
            agreement_symbol = "✓" if data["agreement"] == "exact" else "≈" if data["agreement"] == "within_1" else "✗"
            report.append(f"   {agreement_symbol} {criterion.upper():15s}: Human={data['human']}, LLM={data['llm']}, Diff={data['difference']:+d}")
    
    report.append("\n\n" + "="*80)
    
    # Recommendations
    report.append("\n📋 RECOMMENDATIONS:")
    
    if metrics["cohens_kappa"] < 0.40:
        report.append("   • Cohen's Kappa < 0.40: Add harder constraints to rubric prompts")
        report.append("   • Review divergent cases manually to identify systematic issues")
    
    if abs(metrics["mean_bias"]) > 0.5:
        if metrics["mean_bias"] > 0:
            report.append(f"   • LLM is {metrics['mean_bias']:.1f} points too lenient: Tighten rubric language")
        else:
            report.append(f"   • LLM is {abs(metrics['mean_bias']):.1f} points too strict: Add examples of acceptable LOs")
    
    if metrics["within_one_agreement_pct"] < 60:
        report.append("   • Within-one agreement < 60%: Increase calibration set size & iterate")
        report.append("   • Consider using human scores as ground truth and fine-tuning prompts")
    
    if interp["suitable_for_paper"]:
        report.append("\n   ✓ Metrics are acceptable for publication. Report Kappa and within-one agreement.")
    
    report.append("\n" + "="*80)
    report.append("END OF CALIBRATION REPORT")
    report.append("="*80)
    
    return "\n".join(report)


# ==================== MAIN ====================

def main():
    """Run calibration analysis."""
    print("="*70)
    print("  CALIBRATION: HUMAN-LLM AGREEMENT ANALYSIS")
    print("="*70)
    
    # Load calibration set
    calibration_set = load_calibration_set()
    
    if not calibration_set:
        print("\n❌ No calibration set found.")
        print("\nTo use this tool:")
        print("1. Manually score 5-10 LOs using the rubrics")
        print("2. Create calibration_set.json with your scores")
        print("3. Run LLM evaluation on the same LOs")
        print("4. Run this script to compare")
        return
    
    print(f"\n📚 Loaded {len(calibration_set)} manually-scored LOs")
    
    # Analyze
    print("\n🔍 Comparing with LLM evaluation results...")
    analysis = analyze_calibration_set(calibration_set)
    
    if "error" in analysis:
        print(f"\n❌ Analysis failed: {analysis['error']}")
        return
    
    # Generate report
    report = generate_calibration_report(analysis)
    
    # Save report
    os.makedirs(os.path.dirname(CALIBRATION_REPORT), exist_ok=True)
    with open(CALIBRATION_REPORT, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to: {CALIBRATION_REPORT}")


if __name__ == "__main__":
    main()
