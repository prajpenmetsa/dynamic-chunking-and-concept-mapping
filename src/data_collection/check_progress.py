"""
Quick Statistics Generator for Dataset Collection

Shows current progress across all data sources.

Usage:
    python check_progress.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Paths
BASE_DIR = Path("../../datasets/large_scale_collection/raw")
COURSERA_DIR = BASE_DIR / "coursera"
UDEMY_DIR = BASE_DIR / "udemy"
UNIVERSITIES_DIR = BASE_DIR / "universities"

def load_json_files(directory: Path) -> List[Dict]:
    """Load all JSON files from a directory"""
    courses = []
    
    if not directory.exists():
        return courses
    
    for file_path in directory.glob("*.json"):
        if file_path.name.endswith("_stats.json"):
            continue  # Skip stats files
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    courses.extend(data)
                elif isinstance(data, dict) and 'courses' in data:
                    courses.extend(data['courses'])
                    
        except Exception as e:
            print(f"⚠️ Error loading {file_path.name}: {e}")
    
    return courses

def calculate_stats(courses: List[Dict]) -> Dict:
    """Calculate statistics for a list of courses"""
    if not courses:
        return {
            'total': 0,
            'with_los': 0,
            'with_description': 0,
            'lo_coverage': 0,
            'avg_los_per_course': 0
        }
    
    total = len(courses)
    with_los = sum(1 for c in courses if c.get('learning_outcomes'))
    with_description = sum(1 for c in courses if len(c.get('description', '')) > 50)
    
    total_los = sum(len(c.get('learning_outcomes', [])) for c in courses)
    avg_los = total_los / total if total > 0 else 0
    
    return {
        'total': total,
        'with_los': with_los,
        'with_description': with_description,
        'lo_coverage': round(with_los / total * 100, 2) if total > 0 else 0,
        'avg_los_per_course': round(avg_los, 2)
    }

def print_banner(text: str):
    """Print a nice banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def main():
    print_banner("📊 DATASET COLLECTION PROGRESS")
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data from each source
    print("\n🔄 Loading data...")
    coursera_courses = load_json_files(COURSERA_DIR)
    udemy_courses = load_json_files(UDEMY_DIR)
    university_courses = load_json_files(UNIVERSITIES_DIR)
    
    # Calculate statistics
    coursera_stats = calculate_stats(coursera_courses)
    udemy_stats = calculate_stats(udemy_courses)
    university_stats = calculate_stats(university_courses)
    
    # Overall statistics
    all_courses = coursera_courses + udemy_courses + university_courses
    overall_stats = calculate_stats(all_courses)
    
    # Display results
    print_banner("OVERALL STATISTICS")
    print(f"  📚 Total Courses:              {overall_stats['total']:,}")
    print(f"  ✅ With Learning Outcomes:     {overall_stats['with_los']:,} ({overall_stats['lo_coverage']:.1f}%)")
    print(f"  📝 With Descriptions:          {overall_stats['with_description']:,}")
    print(f"  📊 Avg LOs per Course:         {overall_stats['avg_los_per_course']:.2f}")
    
    print_banner("BY SOURCE")
    
    # Coursera
    print(f"\n  🎓 COURSERA:")
    print(f"     Total:            {coursera_stats['total']:,}")
    print(f"     With LOs:         {coursera_stats['with_los']:,} ({coursera_stats['lo_coverage']:.1f}%)")
    print(f"     Avg LOs/Course:   {coursera_stats['avg_los_per_course']:.2f}")
    
    # Udemy
    print(f"\n  💻 UDEMY:")
    print(f"     Total:            {udemy_stats['total']:,}")
    print(f"     With LOs:         {udemy_stats['with_los']:,} ({udemy_stats['lo_coverage']:.1f}%)")
    print(f"     Avg LOs/Course:   {udemy_stats['avg_los_per_course']:.2f}")
    
    # Universities
    print(f"\n  🏫 UNIVERSITIES:")
    print(f"     Total:            {university_stats['total']:,}")
    print(f"     With LOs:         {university_stats['with_los']:,} ({university_stats['lo_coverage']:.1f}%)")
    print(f"     Avg LOs/Course:   {university_stats['avg_los_per_course']:.2f}")
    
    # List individual university files
    if UNIVERSITIES_DIR.exists():
        uni_files = [f for f in UNIVERSITIES_DIR.glob("*.json") if not f.name.endswith("_stats.json")]
        if uni_files:
            print(f"\n     Universities collected:")
            for file in sorted(uni_files):
                file_courses = load_json_files(UNIVERSITIES_DIR.parent / UNIVERSITIES_DIR.name)
                # Count courses in this specific file
                with open(file, 'r') as f:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else len(data.get('courses', []))
                print(f"       • {file.stem}: {count:,} courses")
    
    # Progress towards goals
    print_banner("PROGRESS TOWARDS GOALS")
    
    goals = [
        ("Week 1 Target", 10_000),
        ("Week 2 Target", 50_000),
        ("Week 4 Target", 100_000),
        ("Week 8 Target", 500_000),
        ("Week 12 Target", 1_000_000)
    ]
    
    current = overall_stats['total']
    
    for goal_name, goal_value in goals:
        progress = (current / goal_value * 100) if goal_value > 0 else 0
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        status = "✅" if progress >= 100 else "🔄" if progress > 0 else "⏳"
        print(f"  {status} {goal_name:20s} [{bar}] {progress:5.1f}% ({current:,}/{goal_value:,})")
    
    # Data quality assessment
    print_banner("DATA QUALITY ASSESSMENT")
    
    lo_coverage = overall_stats['lo_coverage']
    
    if lo_coverage >= 80:
        quality = "🟢 EXCELLENT"
    elif lo_coverage >= 60:
        quality = "🟡 GOOD"
    elif lo_coverage >= 40:
        quality = "🟠 FAIR"
    else:
        quality = "🔴 NEEDS IMPROVEMENT"
    
    print(f"  LO Coverage:     {quality} ({lo_coverage:.1f}%)")
    print(f"  Target:          🎯 80%+")
    
    if lo_coverage < 80:
        print(f"\n  ⚠️  Action Required: Improve LO extraction/collection")
        print(f"      Need {int(overall_stats['total'] * 0.8 - overall_stats['with_los']):,} more courses with LOs")
    
    # Recommendations
    print_banner("RECOMMENDATIONS")
    
    if current < 10_000:
        print("  📌 Priority: Reach 10,000 courses (Week 1 goal)")
        print("     • Focus on Coursera (fastest collection)")
        print("     • Add MIT, Stanford if not done")
        print("     • Run collections in parallel")
    elif current < 50_000:
        print("  📌 Priority: Scale to 50,000 courses (Week 2 goal)")
        print("     • Complete Udemy collection")
        print("     • Add 5 more universities")
        print("     • Automate nightly collections")
    elif current < 100_000:
        print("  📌 Priority: Reach 100,000 courses (Week 4 goal)")
        print("     • Expand university coverage (20+ institutions)")
        print("     • Quality validation pipeline")
        print("     • Start course content collection")
    else:
        print("  🎉 Great progress! Continue expanding sources")
        print("     • Maintain collection momentum")
        print("     • Focus on quality improvements")
        print("     • Prepare for LO generation experiments")
    
    # Save statistics to file
    stats_output = {
        'timestamp': datetime.now().isoformat(),
        'overall': overall_stats,
        'by_source': {
            'coursera': coursera_stats,
            'udemy': udemy_stats,
            'universities': university_stats
        },
        'files': {
            'coursera': len(list(COURSERA_DIR.glob("*.json"))) if COURSERA_DIR.exists() else 0,
            'udemy': len(list(UDEMY_DIR.glob("*.json"))) if UDEMY_DIR.exists() else 0,
            'universities': len(list(UNIVERSITIES_DIR.glob("*.json"))) if UNIVERSITIES_DIR.exists() else 0
        }
    }
    
    stats_file = BASE_DIR.parent / "statistics" / "latest_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2)
    
    print(f"\n💾 Statistics saved to: {stats_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
