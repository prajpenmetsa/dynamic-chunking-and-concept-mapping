"""
Generate Human-Readable Evaluation Reports
Creates formatted reports from LLM-as-judge evaluation results
"""

import json
import os
from typing import Dict, List, Any

# ==================== CONFIGURATION ====================
EVALUATION_DIR = "../../datasets/evaluation"
REPORT_DIR = "../../datasets/evaluation/reports"


# ==================== REPORT GENERATION ====================

def generate_abcd_report(eval_data: Dict) -> str:
    """Generate detailed report for ABCD evaluation."""
    
    report = []
    report.append("="*80)
    report.append(f"  ABCD FRAMEWORK EVALUATION REPORT")
    report.append("="*80)
    report.append(f"\nCourse: {eval_data.get('course_title')} ({eval_data.get('course_code')})")
    report.append(f"Number of Learning Objectives: {eval_data.get('num_objectives')}")
    report.append(f"Evaluation Runs per LO: {eval_data['metadata']['num_runs']}")
    report.append(f"Model: {eval_data['metadata']['model']}")
    report.append("\n" + "="*80)
    
    # For each learning objective
    for eval_item in eval_data.get("evaluations", []):
        lo_num = eval_item["objective_number"]
        lo_text = eval_item["learning_objective"]
        consistency = eval_item["consistency_analysis"]
        
        report.append(f"\n\n{'─'*80}")
        report.append(f"LEARNING OBJECTIVE #{lo_num}")
        report.append(f"{'─'*80}")
        report.append(f"\n{lo_text}")
        
        # Consistency scores
        report.append(f"\n\n📊 CONSISTENCY ANALYSIS (across {len(eval_item['evaluation_runs'])} runs):")
        report.append(f"   Overall Score: {consistency['composite_score_mean']:.2f} ± {consistency['composite_score_stdev']:.2f}")
        report.append(f"   Score Range: [{consistency['composite_score_range'][0]:.2f} - {consistency['composite_score_range'][1]:.2f}]")
        
        report.append("\n   Component Scores:")
        for criterion, stats in consistency.get("criteria_consistency", {}).items():
            status = "✓" if stats["is_consistent"] else "⚠️"
            report.append(f"      {status} {criterion.upper():12s}: {stats['mean']:.2f} ± {stats['stdev']:.2f}  [{stats['range'][0]} - {stats['range'][1]}]")
        
        # Detailed evaluation from first run
        first_run = eval_item["evaluation_runs"][0]
        report.append(f"\n\n📋 DETAILED EVALUATION (Run 1):")
        
        for component, data in first_run.get("overall_scores", {}).items():
            report.append(f"\n   {component.upper()} - Score: {data['score']}/5")
            report.append(f"      Evidence: {data.get('evidence', 'N/A')}")
            if data.get('weakness'):
                report.append(f"      Weakness: {data['weakness']}")
        
        # Granular responses
        report.append(f"\n\n🔍 GRANULAR QUESTION RESPONSES:")
        for response in first_run.get("granular_responses", []):
            report.append(f"\n   • {response['criterion']}: {response['score']}/5")
            report.append(f"     Q: {response['question']}")
            report.append(f"     A: {response['justification']}")
        
        # Overall assessment
        report.append(f"\n\n💡 OVERALL ASSESSMENT:")
        report.append(f"   {first_run.get('overall_assessment', 'N/A')}")
        
        # Improvement suggestions
        if first_run.get("improvement_suggestions"):
            report.append(f"\n\n🎯 IMPROVEMENT SUGGESTIONS:")
            for i, suggestion in enumerate(first_run["improvement_suggestions"], 1):
                report.append(f"   {i}. {suggestion}")
    
    report.append("\n\n" + "="*80)
    report.append("END OF ABCD EVALUATION REPORT")
    report.append("="*80)
    
    return "\n".join(report)


def generate_smart_report(eval_data: Dict) -> str:
    """Generate detailed report for SMART evaluation."""
    
    report = []
    report.append("="*80)
    report.append(f"  SMART FRAMEWORK EVALUATION REPORT")
    report.append("="*80)
    report.append(f"\nCourse: {eval_data.get('course_title')} ({eval_data.get('course_code')})")
    report.append(f"Number of Learning Objectives: {eval_data.get('num_objectives')}")
    report.append(f"Evaluation Runs per LO: {eval_data['metadata']['num_runs']}")
    report.append(f"Model: {eval_data['metadata']['model']}")
    report.append("\n" + "="*80)
    
    # For each learning objective
    for eval_item in eval_data.get("evaluations", []):
        lo_num = eval_item["objective_number"]
        lo_text = eval_item["learning_objective"]
        consistency = eval_item["consistency_analysis"]
        
        report.append(f"\n\n{'─'*80}")
        report.append(f"LEARNING OBJECTIVE #{lo_num}")
        report.append(f"{'─'*80}")
        report.append(f"\n{lo_text}")
        
        # Consistency scores
        report.append(f"\n\n📊 CONSISTENCY ANALYSIS (across {len(eval_item['evaluation_runs'])} runs):")
        report.append(f"   Overall Score: {consistency['composite_score_mean']:.2f} ± {consistency['composite_score_stdev']:.2f}")
        report.append(f"   Score Range: [{consistency['composite_score_range'][0]:.2f} - {consistency['composite_score_range'][1]:.2f}]")
        
        report.append("\n   Component Scores:")
        for criterion, stats in consistency.get("criteria_consistency", {}).items():
            status = "✓" if stats["is_consistent"] else "⚠️"
            report.append(f"      {status} {criterion.upper():12s}: {stats['mean']:.2f} ± {stats['stdev']:.2f}  [{stats['range'][0]} - {stats['range'][1]}]")
        
        # Detailed evaluation from first run
        first_run = eval_item["evaluation_runs"][0]
        report.append(f"\n\n📋 DETAILED EVALUATION (Run 1):")
        
        for component, data in first_run.get("overall_scores", {}).items():
            report.append(f"\n   {component.upper()} - Score: {data['score']}/5")
            report.append(f"      Evidence: {data.get('evidence', 'N/A')}")
            if data.get('weakness'):
                report.append(f"      Weakness: {data['weakness']}")
        
        # Granular responses
        report.append(f"\n\n🔍 GRANULAR QUESTION RESPONSES:")
        for response in first_run.get("granular_responses", []):
            report.append(f"\n   • {response['criterion']}: {response['score']}/5")
            report.append(f"     Q: {response['question']}")
            report.append(f"     A: {response['justification']}")
        
        # Overall assessment
        report.append(f"\n\n💡 OVERALL ASSESSMENT:")
        report.append(f"   {first_run.get('overall_assessment', 'N/A')}")
        
        # Improvement suggestions
        if first_run.get("improvement_suggestions"):
            report.append(f"\n\n🎯 IMPROVEMENT SUGGESTIONS:")
            for i, suggestion in enumerate(first_run["improvement_suggestions"], 1):
                report.append(f"   {i}. {suggestion}")
    
    report.append("\n\n" + "="*80)
    report.append("END OF SMART EVALUATION REPORT")
    report.append("="*80)
    
    return "\n".join(report)


def generate_blooms_report(eval_data: Dict) -> str:
    """Generate detailed report for Bloom's Taxonomy evaluation."""
    
    report = []
    report.append("="*80)
    report.append(f"  BLOOM'S TAXONOMY EVALUATION REPORT")
    report.append("="*80)
    report.append(f"\nCourse: {eval_data.get('course_title')} ({eval_data.get('course_code')})")
    report.append(f"Number of Learning Objectives: {eval_data.get('num_objectives')}")
    report.append(f"Evaluation Runs: {eval_data['metadata']['num_runs']}")
    report.append(f"Model: {eval_data['metadata']['model']}")
    report.append("\n" + "="*80)
    
    # Use first run for detailed analysis
    first_run = eval_data.get("evaluation_runs", [{}])[0]
    consistency = eval_data.get("consistency_analysis", {})
    
    # Overall consistency
    report.append(f"\n\n📊 CONSISTENCY ANALYSIS (across {eval_data['metadata']['num_runs']} runs):")
    report.append(f"   Overall Average Score: {consistency.get('overall_mean', 0):.2f}")
    report.append(f"   Individual LOs StdDev: {consistency.get('individual_stdev', 0):.2f}")
    report.append(f"   Set-Level StdDev: {consistency.get('set_level_stdev', 0):.2f}")
    
    # Individual LO evaluations
    report.append(f"\n\n{'='*80}")
    report.append("INDIVIDUAL LEARNING OBJECTIVE EVALUATIONS")
    report.append("="*80)
    
    for lo_eval in first_run.get("individual_evaluations", []):
        lo_num = lo_eval["objective_number"]
        lo_text = lo_eval["objective_text"]
        identified_level = lo_eval["identified_level"]
        composite = lo_eval["composite_score"]
        
        report.append(f"\n\n{'─'*80}")
        report.append(f"LEARNING OBJECTIVE #{lo_num}")
        report.append(f"{'─'*80}")
        report.append(f"\n{lo_text}")
        report.append(f"\n📌 Identified Bloom's Level: {identified_level}")
        report.append(f"📊 Composite Score: {composite:.2f}/5.0")
        
        # Component scores
        report.append(f"\n\n📋 COMPONENT SCORES:")
        for component, data in lo_eval.get("scores", {}).items():
            report.append(f"\n   {component.upper().replace('_', ' ')} - Score: {data['score']}/5")
            report.append(f"      Evidence: {data.get('evidence', 'N/A')}")
            if data.get('weakness'):
                report.append(f"      Weakness: {data['weakness']}")
        
        # Granular responses
        if lo_eval.get("granular_responses"):
            report.append(f"\n\n🔍 GRANULAR QUESTION RESPONSES:")
            for response in lo_eval["granular_responses"]:
                report.append(f"\n   • {response['criterion']}: {response['score']}/5")
                report.append(f"     Q: {response['question']}")
                report.append(f"     A: {response['justification']}")
        
        # Improvement suggestions
        if lo_eval.get("improvement_suggestions"):
            report.append(f"\n\n🎯 IMPROVEMENT SUGGESTIONS:")
            for i, suggestion in enumerate(lo_eval["improvement_suggestions"], 1):
                report.append(f"   {i}. {suggestion}")
    
    # Set-level evaluation
    set_eval = first_run.get("set_evaluation", {})
    report.append(f"\n\n{'='*80}")
    report.append("SET-LEVEL EVALUATION (Progression & Coverage)")
    report.append("="*80)
    
    # Level distribution
    level_dist = set_eval.get("level_distribution", {})
    report.append(f"\n📊 BLOOM'S LEVEL DISTRIBUTION:")
    for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
        count = level_dist.get(level, 0)
        bar = "█" * count
        report.append(f"   {level:12s}: {bar} ({count})")
    
    # Progression score
    prog_score = set_eval.get("progression_score", {})
    report.append(f"\n📈 PROGRESSION SCORE: {prog_score.get('score', 0)}/5")
    report.append(f"   Evidence: {prog_score.get('evidence', 'N/A')}")
    if prog_score.get('weakness'):
        report.append(f"   Weakness: {prog_score['weakness']}")
    
    # Overall assessment
    report.append(f"\n\n💡 OVERALL SET ASSESSMENT:")
    report.append(f"   {set_eval.get('overall_assessment', 'N/A')}")
    
    # Recommendations
    if first_run.get("recommendations"):
        report.append(f"\n\n🎯 RECOMMENDATIONS FOR IMPROVEMENT:")
        for i, rec in enumerate(first_run["recommendations"], 1):
            report.append(f"   {i}. {rec}")
    
    report.append("\n\n" + "="*80)
    report.append("END OF BLOOM'S TAXONOMY EVALUATION REPORT")
    report.append("="*80)
    
    return "\n".join(report)


def generate_summary_report(abcd_data: Dict, smart_data: Dict, blooms_data: Dict) -> str:
    """Generate cross-framework comparison summary."""
    
    report = []
    report.append("="*80)
    report.append("  CROSS-FRAMEWORK EVALUATION SUMMARY")
    report.append("="*80)
    
    # Get average scores from each framework
    abcd_avg = 0
    if abcd_data.get("evaluations"):
        scores = [e["consistency_analysis"]["composite_score_mean"] for e in abcd_data["evaluations"]]
        abcd_avg = sum(scores) / len(scores)
    
    smart_avg = 0
    if smart_data.get("evaluations"):
        scores = [e["consistency_analysis"]["composite_score_mean"] for e in smart_data["evaluations"]]
        smart_avg = sum(scores) / len(scores)
    
    blooms_avg = blooms_data.get("consistency_analysis", {}).get("overall_mean", 0)
    
    report.append(f"\n📊 AVERAGE SCORES ACROSS FRAMEWORKS:")
    report.append(f"   ABCD Framework:     {abcd_avg:.2f}/5.0")
    report.append(f"   SMART Framework:    {smart_avg:.2f}/5.0")
    report.append(f"   Bloom's Taxonomy:   {blooms_avg:.2f}/5.0")
    
    # Consistency analysis
    report.append(f"\n\n📈 CONSISTENCY METRICS:")
    
    if abcd_data.get("evaluations"):
        abcd_stdevs = [e["consistency_analysis"]["composite_score_stdev"] for e in abcd_data["evaluations"]]
        abcd_consistency = sum(abcd_stdevs) / len(abcd_stdevs)
        report.append(f"   ABCD Average StdDev:    {abcd_consistency:.2f}")
    
    if smart_data.get("evaluations"):
        smart_stdevs = [e["consistency_analysis"]["composite_score_stdev"] for e in smart_data["evaluations"]]
        smart_consistency = sum(smart_stdevs) / len(smart_stdevs)
        report.append(f"   SMART Average StdDev:   {smart_consistency:.2f}")
    
    blooms_consistency = blooms_data.get("consistency_analysis", {}).get("individual_stdev", 0)
    report.append(f"   Bloom's Average StdDev: {blooms_consistency:.2f}")
    
    # Interpretation
    report.append(f"\n\n💡 INTERPRETATION:")
    report.append(f"   • Lower standard deviation (<0.5) indicates consistent evaluation across runs")
    report.append(f"   • Scores 4.0+ indicate strong alignment with framework criteria")
    report.append(f"   • Scores 2.5-4.0 indicate partial alignment with room for improvement")
    report.append(f"   • Scores <2.5 indicate significant gaps in framework adherence")
    
    # Recommendations
    report.append(f"\n\n🎯 KEY FINDINGS & RECOMMENDATIONS:")
    
    # ABCD analysis
    if abcd_avg >= 4.0:
        report.append(f"   ✓ ABCD: Strong adherence to framework. LOs are well-structured.")
    elif abcd_avg >= 3.0:
        report.append(f"   ⚠ ABCD: Moderate adherence. Review weak components (A/B/C/D).")
    else:
        report.append(f"   ❌ ABCD: Significant gaps. Major restructuring needed.")
    
    # SMART analysis
    if smart_avg >= 4.0:
        report.append(f"   ✓ SMART: Highly specific and measurable objectives.")
    elif smart_avg >= 3.0:
        report.append(f"   ⚠ SMART: Partially SMART. Improve specificity and measurability.")
    else:
        report.append(f"   ❌ SMART: Vague objectives. Add concrete criteria and metrics.")
    
    # Bloom's analysis
    if blooms_avg >= 4.0:
        report.append(f"   ✓ BLOOM'S: Excellent alignment with taxonomy and progression.")
    elif blooms_avg >= 3.0:
        report.append(f"   ⚠ BLOOM'S: Acceptable taxonomy use. Verify verb accuracy.")
    else:
        report.append(f"   ❌ BLOOM'S: Poor taxonomy alignment. Review verb selection.")
    
    report.append("\n" + "="*80)
    report.append("END OF SUMMARY REPORT")
    report.append("="*80)
    
    return "\n".join(report)


# ==================== MAIN ====================

def main():
    """Generate all evaluation reports."""
    print("="*70)
    print("  GENERATING EVALUATION REPORTS")
    print("="*70)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Load evaluation data
    abcd_file = os.path.join(EVALUATION_DIR, "evaluation_abcd.json")
    smart_file = os.path.join(EVALUATION_DIR, "evaluation_smart.json")
    blooms_file = os.path.join(EVALUATION_DIR, "evaluation_blooms.json")
    
    data_loaded = {}
    
    # ABCD Report
    if os.path.exists(abcd_file):
        print("\n📄 Generating ABCD report...")
        with open(abcd_file, 'r') as f:
            abcd_data = json.load(f)
            data_loaded['abcd'] = abcd_data
        
        report = generate_abcd_report(abcd_data)
        output_path = os.path.join(REPORT_DIR, "report_abcd.txt")
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"   ✓ Saved: {output_path}")
    else:
        print(f"\n⚠️  ABCD evaluation file not found: {abcd_file}")
    
    # SMART Report
    if os.path.exists(smart_file):
        print("\n📄 Generating SMART report...")
        with open(smart_file, 'r') as f:
            smart_data = json.load(f)
            data_loaded['smart'] = smart_data
        
        report = generate_smart_report(smart_data)
        output_path = os.path.join(REPORT_DIR, "report_smart.txt")
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"   ✓ Saved: {output_path}")
    else:
        print(f"\n⚠️  SMART evaluation file not found: {smart_file}")
    
    # Bloom's Report
    if os.path.exists(blooms_file):
        print("\n📄 Generating Bloom's Taxonomy report...")
        with open(blooms_file, 'r') as f:
            blooms_data = json.load(f)
            data_loaded['blooms'] = blooms_data
        
        report = generate_blooms_report(blooms_data)
        output_path = os.path.join(REPORT_DIR, "report_blooms.txt")
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"   ✓ Saved: {output_path}")
    else:
        print(f"\n⚠️  Bloom's evaluation file not found: {blooms_file}")
    
    # Summary Report
    if len(data_loaded) >= 2:
        print("\n📄 Generating cross-framework summary...")
        summary = generate_summary_report(
            data_loaded.get('abcd', {}),
            data_loaded.get('smart', {}),
            data_loaded.get('blooms', {})
        )
        output_path = os.path.join(REPORT_DIR, "report_summary.txt")
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"   ✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("  ✅ REPORT GENERATION COMPLETE")
    print("="*70)
    print(f"\n📁 Reports saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
