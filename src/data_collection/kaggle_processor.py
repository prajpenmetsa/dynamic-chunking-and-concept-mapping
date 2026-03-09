"""
Kaggle Dataset Processor

Standardizes downloaded Kaggle course datasets into the project schema.

Processes:
- combined_dataset.json  (13,793 records - Coursera/edX/multi-source with descriptions + skills)
- edx_courses.json       (1,000 records - edX with HTML descriptions + skills)
- Oxford.csv             (1,049 records - Oxford with descriptions)
- Harvard_university.csv (494 records - Harvard with descriptions)
- alison.csv             (5,725 records - Alison with skills)

Usage:
    python kaggle_processor.py
"""

import json
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

SCRIPT_DIR = Path(__file__).parent
KAGGLE_DIR = SCRIPT_DIR / "../../datasets/large_scale_collection/raw/kaggle"
OUT_DIR = SCRIPT_DIR / "../../datasets/large_scale_collection/raw"
TODAY = datetime.now().isoformat()


def strip_html(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()


def make_id(source: str, title: str) -> str:
    slug = hashlib.md5(f"{source}_{title}".encode()).hexdigest()[:10]
    return f"{source}_{slug}"


def parse_skills(skills_val) -> List[str]:
    """Handle skills in list, string, or dict formats"""
    if not skills_val:
        return []
    if isinstance(skills_val, list):
        result = []
        for s in skills_val:
            if isinstance(s, dict):
                result.append(s.get("skill", ""))
            elif isinstance(s, str) and s.lower() not in ("nan", "none", ""):
                result.append(s)
        return [s for s in result if s]
    if isinstance(skills_val, str):
        cleaned = skills_val.strip("[]").replace("'", "")
        parts = [p.strip() for p in cleaned.split(",") if p.strip().lower() not in ("nan", "none", "")]
        return parts
    return []


def make_record(source, title, description, learning_outcomes, institution=None,
                level=None, category=None, language="English", url=None) -> Optional[Dict]:
    title = (title or "").strip()
    description = strip_html(description or "")
    if not title or len(description) < 50:
        return None
    return {
        "id": make_id(source, title),
        "source": source,
        "course_code": None,
        "title": title,
        "description": description,
        "learning_outcomes": [lo for lo in learning_outcomes if lo and isinstance(lo, str)],
        "metadata": {
            "institution": institution or source,
            "level": level,
            "category": category,
            "language": language,
            "date_collected": TODAY,
            "url": url
        }
    }


def process_combined_dataset() -> List[Dict]:
    print("Processing combined_dataset.json...")
    with open(KAGGLE_DIR / "combined_dataset.json") as f:
        data = json.load(f)

    courses = []
    for r in data:
        description = strip_html(r.get("description", ""))
        # Remove leading "Description:" prefix if present
        if description.lower().startswith("description:"):
            description = description[12:].strip()

        skills = parse_skills(r.get("skills"))
        rec = make_record(
            source=r.get("provider", "mooc").lower().replace(" ", "_"),
            title=r.get("course_name"),
            description=description,
            learning_outcomes=skills,
            institution=r.get("organization"),
            level=r.get("level"),
            category=r.get("subject"),
            url=r.get("url")
        )
        if rec:
            courses.append(rec)

    print(f"  -> {len(courses)} valid records")
    return courses


def process_edx_courses() -> List[Dict]:
    print("Processing edx_courses.json...")
    with open(KAGGLE_DIR / "edx_courses.json") as f:
        data = json.load(f)

    courses = []
    for r in data:
        # Combine primary + secondary descriptions
        desc_parts = [
            strip_html(r.get("primary_description", "")),
            strip_html(r.get("secondary_description", "")),
        ]
        description = " ".join(p for p in desc_parts if p)

        skills = parse_skills(r.get("skills"))

        partners = r.get("partner", [])
        institution = partners[0] if partners else "edX"

        rec = make_record(
            source="edx",
            title=r.get("title"),
            description=description,
            learning_outcomes=skills,
            institution=institution,
            level=r.get("level"),
            category=r.get("subject"),
            language=r.get("language", "English"),
            url=r.get("marketing_url")
        )
        if rec:
            courses.append(rec)

    print(f"  -> {len(courses)} valid records")
    return courses


def process_oxford() -> List[Dict]:
    print("Processing Oxford.csv...")
    df = pd.read_csv(KAGGLE_DIR / "Oxford.csv", on_bad_lines="skip")
    courses = []
    for _, r in df.iterrows():
        rec = make_record(
            source="oxford",
            title=r.get("Name"),
            description=str(r.get("About Course", "")),
            learning_outcomes=[],
            institution="University of Oxford",
            url=str(r.get("Link", "")) or None
        )
        if rec:
            courses.append(rec)
    print(f"  -> {len(courses)} valid records")
    return courses


def process_harvard() -> List[Dict]:
    print("Processing Harvard_university.csv...")
    df = pd.read_csv(KAGGLE_DIR / "Harvard_university.csv", on_bad_lines="skip")
    courses = []
    for _, r in df.iterrows():
        rec = make_record(
            source="harvard",
            title=r.get("Name"),
            description=str(r.get("About", "")),
            learning_outcomes=[],
            institution="Harvard University",
            category=str(r.get("subject ", "")) or None,
            url=str(r.get("Link to course", "")) or None
        )
        if rec:
            courses.append(rec)
    print(f"  -> {len(courses)} valid records")
    return courses


def process_alison() -> List[Dict]:
    print("Processing alison.csv...")
    df = pd.read_csv(KAGGLE_DIR / "alison.csv", on_bad_lines="skip")
    courses = []
    for _, r in df.iterrows():
        skills_raw = str(r.get("Skills", ""))
        skills = [s.strip() for s in skills_raw.split(",") if s.strip() and s.strip().lower() not in ("nan", "none")]
        # Alison has no description — use skills as a proxy description only if rich enough
        if len(skills) < 3:
            continue
        desc = f"This course covers: {', '.join(skills)}."
        rec = make_record(
            source="alison",
            title=r.get("Name Of The Course "),
            description=desc,
            learning_outcomes=skills,
            institution=str(r.get("Institute", "Alison")),
            category=str(r.get("Category", "")) or None,
            url=str(r.get("Link", "")) or None
        )
        if rec:
            courses.append(rec)
    print(f"  -> {len(courses)} valid records")
    return courses


def deduplicate(courses: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for c in courses:
        key = c["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def validate(courses: List[Dict]) -> Dict:
    total = len(courses)
    with_los = sum(1 for c in courses if c["learning_outcomes"])
    with_desc = sum(1 for c in courses if len(c.get("description", "")) > 50)
    avg_los = sum(len(c["learning_outcomes"]) for c in courses) / total if total else 0
    return {
        "total_courses": total,
        "with_learning_outcomes": with_los,
        "with_description": with_desc,
        "lo_coverage": round(with_los / total * 100, 2) if total else 0,
        "avg_los_per_course": round(avg_los, 2)
    }


def main():
    print("=" * 60)
    print("KAGGLE DATASET PROCESSOR")
    print("=" * 60)

    all_courses = []
    all_courses.extend(process_combined_dataset())
    all_courses.extend(process_edx_courses())
    all_courses.extend(process_oxford())
    all_courses.extend(process_harvard())
    all_courses.extend(process_alison())

    print(f"\nTotal before dedup: {len(all_courses)}")
    all_courses = deduplicate(all_courses)
    print(f"Total after dedup:  {len(all_courses)}")

    # Split by source and save
    sources = {}
    for c in all_courses:
        src = c["source"]
        sources.setdefault(src, []).append(c)

    print("\nBy source:")
    for src, courses in sorted(sources.items()):
        print(f"  {src}: {len(courses)}")

    # Save one combined kaggle file
    out_path = OUT_DIR / "kaggle" / f"kaggle_processed_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_courses, f, indent=2, ensure_ascii=False)

    stats = validate(all_courses)
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    stats_path = out_path.with_suffix("").with_name(out_path.stem + "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
