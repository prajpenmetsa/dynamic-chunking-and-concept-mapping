"""
Slide-based Learning Outcomes (LOs) generation using ABCD Framework
MAP-REDUCE approach:
  1) For each PDF deck: extract ALL text (text-only; ignores images) and summarize into compact JSON.
  2) Combine all deck summaries and generate 6–7 course LOs in ABCD format.

This ensures ALL PDFs are "taken in" (influence the final LOs) without hitting context limits.

Requirements:
  pip install pdfplumber python-dotenv tqdm requests
  Sign in to Ollama Cloud: ollama signin
  Pull cloud model: ollama pull gpt-oss:20b-cloud
  Create API key at: https://ollama.com/settings/keys
  Add to .env file: OLLAMA_API_KEY=your_api_key_here
"""

import json
import os
import glob
import time
from typing import Dict, Any, List

import pdfplumber
from tqdm import tqdm
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================
load_dotenv()

# Ollama API Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY not found in environment. Add it to your .env file.")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com")  # Default to Ollama cloud for cloud models
MODEL_NAME = "gpt-oss:20b-cloud"  # Cloud-hosted model

# Course Configuration - UPDATE THESE VALUES
COURSE_TITLE = "Operating Systems and Networks"
COURSE_CODE = "CS3.301"
SLIDE_DECKS_FOLDER = "/Users/lakshmiprajnapenmetsa/Desktop/iiith/research/irel/honors/dynamic-chunking-and-concept-mapping/raw-data/osn_lecs"
OUTPUT_FILE = "/Users/lakshmiprajnapenmetsa/Desktop/iiith/research/irel/honors/dynamic-chunking-and-concept-mapping/datasets/slide_based_los_abcd.json"

# Safety knobs
MIN_EXTRACTED_CHARS_PER_PDF = 200        # If below this, we treat as "no usable text"
MAX_DECK_TEXT_CHARS = 45000              # Per-deck truncation to avoid huge prompts (still "all PDFs" overall)
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2
RATE_LIMIT_DELAY = 5                     # Seconds to wait between API calls


# ==================== PDF TEXT EXTRACTION (TEXT ONLY) ====================
def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all TEXT from all pages in a single PDF.
    (Ignores any non-text content like images; no OCR.)
    """
    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_text.append(text)
    return "\n".join(pages_text)


def safe_truncate(text: str, max_chars: int) -> str:
    """
    Truncate text to max_chars safely.
    We keep the START and END of the document because:
    - starts often include title/agenda
    - ends often include summaries/exercises
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


# ==================== PROMPTS ====================
def create_deck_summary_prompt(course_title: str, filename: str, deck_text: str) -> str:
    """
    Per-deck compression step: turn raw deck text into compact structured JSON
    that is small enough to combine across all decks later.
    """
    return f"""
You are an expert instructor for an advanced Operating Systems course.

Task: Convert this lecture deck into a compact structured summary that captures what the deck teaches.

Course: {course_title}
Deck file: {filename}

Deck text (extracted from PDF; text-only):
{deck_text}

Return ONLY valid JSON in exactly this schema (no markdown, no extra keys):
{{
  "deck": "{filename}",
  "topics": ["..."],
  "key_concepts": ["..."],
  "skills": ["..."],
  "important_terms": ["..."]
}}

Guidelines:
- "topics": 5–12 high-level topics taught in this deck.
- "key_concepts": 8–20 concrete concepts/mechanisms/algorithms.
- "skills": 5–12 student skills (e.g., analyze deadlock scenarios, implement paging simulator).
- "important_terms": 10–25 technical terms/acronyms.
- Keep strings concise and specific to OS.
"""


def create_final_lo_prompt_abcd(course_title: str, course_code: str, all_deck_summaries_json: str) -> str:
    """
    Final synthesis step: generate course outcomes using ABCD framework from all deck summaries.
    """
    return f"""
You are an expert Educational Curriculum Designer.

Task: Create 6–7 comprehensive Learning Outcomes (LOs) for this university course using the ABCD Framework.

Course:
- Title: {course_title}
- Code: {course_code}

You are given structured summaries for ALL lecture decks (JSON list):
{all_deck_summaries_json}

ABCD Framework requirements (MUST be present in EACH outcome):
A = Audience: Students
B = Behavior: measurable verb (Analyze, Design, Implement, Evaluate, Compare, etc.)
C = Condition: given/using/under constraints (e.g., given workload traces, using xv6, under memory limits)
D = Degree: performance criteria (e.g., with ≥95% correctness, within X latency, meeting specified constraints)

Format template:
"CO-i: Students will [Behavior] [topic/content] [Condition] [Degree]."

Rules:
- Return ONLY a JSON array of strings (no markdown).
- Ensure each outcome clearly contains A, B, C, and D.
- Cover breadth across the full set of deck summaries (not just one topic).
- Outcomes should progress from foundational to advanced capabilities.
"""



# ==================== OLLAMA API CALL HELPER ====================
def parse_json_response(text: str) -> Any:
    """
    Parse JSON from API response, handling markdown code blocks and preamble text.
    """
    text = text.strip()
    
    # If there's a markdown code block, extract it
    if "```json" in text or "```" in text:
        # Find the start of the code block
        start_markers = ["```json", "```"]
        start_idx = -1
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                start_idx = idx
                break
        
        if start_idx != -1:
            # Skip past the opening marker
            text = text[start_idx:]
            lines = text.split("\n")
            lines = lines[1:] if lines else lines  # Remove first line with ```
            
            # Find closing ``` and remove it
            json_lines = []
            for line in lines:
                if line.strip() == "```":
                    break
                json_lines.append(line)
            text = "\n".join(json_lines)
    
    # If still has issues, try to find JSON object boundaries
    text = text.strip()
    if not text.startswith("{") and not text.startswith("["):
        # Find first { or [
        for i, char in enumerate(text):
            if char in ("{", "["):
                text = text[i:]
                break
    
    return json.loads(text.strip())


def call_ollama_api(prompt: str) -> Any:
    """
    Call the Ollama API to generate a response based on the given prompt.
    Uses Ollama's generate endpoint format.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 4096
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"      [API attempt {attempt}/{MAX_RETRIES}]")
            response = requests.post(f"{OLLAMA_API_URL}/api/generate", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # Extract the generated text from Ollama's response format
            generated_text = result.get("response", "")
            if not generated_text:
                raise ValueError("Empty response from Ollama API")

            # Parse JSON, handling markdown code blocks and preamble text
            return parse_json_response(generated_text)
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            wait_time = RETRY_SLEEP_SECONDS * attempt
            try:
                error_json = response.json()
                error_detail = f" - {error_json}"
                # If rate limit, extract the suggested wait time
                if response.status_code == 429 and "error" in error_json:
                    error_msg = error_json["error"].get("message", "")
                    if "try again in" in error_msg.lower():
                        # Extract wait time like "13.335s" from message
                        import re
                        match = re.search(r'try again in ([\d.]+)s', error_msg)
                        if match:
                            wait_time = max(wait_time, float(match.group(1)) + 1)
            except:
                error_detail = f" - {response.text[:300]}"
            print(f"      ⚠️  Error: {type(e).__name__}: {str(e)[:200]}{error_detail}")
            if attempt < MAX_RETRIES:
                print(f"      → Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"      ⚠️  Error: {type(e).__name__}: {str(e)[:200]}")
            if attempt < MAX_RETRIES:
                print(f"      → Retrying in {RETRY_SLEEP_SECONDS * attempt}s...")
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
            else:
                raise


# ==================== MAIN EXECUTION ====================
def main():
    print("=" * 70)
    print("  ABCD FRAMEWORK: SLIDE-BASED LO GENERATION (ALL PDFS)")
    print("  TEXT-ONLY INPUTS | MAP-REDUCE (DECK SUMMARY -> FINAL LOs)")
    print("=" * 70)

    # Test API connectivity
    print("\n🔑 Testing Ollama API connection...")
    try:
        # Test with a simple generation
        test_prompt = 'Reply with valid JSON: {"status": "OK"}'
        test_response = call_ollama_api(test_prompt)
        print(f"   ✓ API connection successful (response: {test_response})")
    except Exception as e:
        print(f"   ❌ API connection failed: {type(e).__name__}: {str(e)}")
        print("   Ensure OLLAMA_API_KEY is set correctly in your .env file.")
        return

    pdf_files = sorted(glob.glob(os.path.join(SLIDE_DECKS_FOLDER, "*.pdf")))
    print(f"\n📚 Found {len(pdf_files)} PDF files in: {SLIDE_DECKS_FOLDER}")
    if not pdf_files:
        print("❌ No PDFs found. Check folder path.")
        return

    # -------- MAP: summarize each deck --------
    deck_summaries: List[Dict[str, Any]] = []
    extraction_report: List[Dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Extracting + summarizing PDFs"):
        filename = os.path.basename(pdf_path)
        print(f"\n🔍 Processing: {filename}")

        try:
            print(f"   → Extracting text...")
            raw_text = extract_pdf_text(pdf_path)
            raw_len = len(raw_text)
            print(f"   → Extracted {raw_len} chars")

            if raw_len < MIN_EXTRACTED_CHARS_PER_PDF:
                print(f"   ⚠️  Skipped: too little text ({raw_len} chars)")
                extraction_report.append({
                    "deck": filename,
                    "extracted_chars": raw_len,
                    "status": "skipped_low_text"
                })
                continue

            # Per-deck truncation to keep prompts safe; still uses all PDFs overall
            deck_text = safe_truncate(raw_text, MAX_DECK_TEXT_CHARS)

            print(f"   → Calling Ollama API...")
            prompt = create_deck_summary_prompt(COURSE_TITLE, filename, deck_text)
            deck_summary = call_ollama_api(prompt)
            print(f"   ✓ Received summary")
            
            # Rate limit protection: wait between requests
            print(f"   → Waiting {RATE_LIMIT_DELAY}s to avoid rate limits...")
            time.sleep(RATE_LIMIT_DELAY)

            # Minimal validation (avoid crashes later)
            if not isinstance(deck_summary, dict) or "deck" not in deck_summary:
                extraction_report.append({
                    "deck": filename,
                    "extracted_chars": raw_len,
                    "status": "bad_summary_format"
                })
                continue

            deck_summaries.append(deck_summary)
            extraction_report.append({
                "deck": filename,
                "extracted_chars": raw_len,
                "status": "summarized"
            })

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {str(e)}")
            extraction_report.append({
                "deck": filename,
                "extracted_chars": None,
                "status": f"error: {str(e)}"
            })
            continue

    if not deck_summaries:
        print("\n❌ No deck summaries produced.")
        print("   If PDFs are image-based, pdfplumber will extract little/no text (OCR would be needed).")
        return

    print(f"\n✓ Summarized {len(deck_summaries)} / {len(pdf_files)} PDFs into compact JSON.")

    # -------- REDUCE: generate final learning outcomes --------
    print("\n📝 Generating final Learning Outcomes from ALL deck summaries...")

    # If the combined summaries are still huge, do a second compression pass
    all_summaries_json = json.dumps(deck_summaries, ensure_ascii=False)
    # Soft cap; if too long, compress again by asking Ollama to merge summaries
    if len(all_summaries_json) > 120000:
        compress_prompt = f"""
You are an expert OS instructor.
Compress the following JSON list of deck summaries into a smaller JSON list.
Do NOT lose major topics; merge duplicates. Keep each field concise.

Return ONLY JSON list with same schema items:
[
  {{"deck":"MERGED-x","topics":[...],"key_concepts":[...],"skills":[...],"important_terms":[...]}},
  ...
]

INPUT:
{all_summaries_json}
"""
        deck_summaries = call_ollama_api(compress_prompt)
        all_summaries_json = json.dumps(deck_summaries, ensure_ascii=False)

    final_prompt = create_final_lo_prompt_abcd(COURSE_TITLE, COURSE_CODE, all_summaries_json)
    learning_objectives = call_ollama_api(final_prompt)

    if not isinstance(learning_objectives, list) or not all(isinstance(x, str) for x in learning_objectives):
        print("❌ Final LOs came back in an invalid format.")
        return

    # -------- SAVE OUTPUT --------
    output = {
        "course_title": COURSE_TITLE,
        "course_code": COURSE_CODE,
        "metadata": {
            "source": "All lecture slide decks (text-only)",
            "folder": SLIDE_DECKS_FOLDER,
            "framework": "ABCD (Audience, Behavior, Condition, Degree)",
            "model": MODEL_NAME,
            "num_pdfs_found": len(pdf_files),
            "num_decks_summarized": len(deck_summaries),
            "per_deck_text_truncation_chars": MAX_DECK_TEXT_CHARS,
            "note": "Per-deck prompts may truncate very long extracted text, but every PDF contributes via per-deck summaries."
        },
        "extraction_report": extraction_report,
        "deck_summaries": deck_summaries,
        "learning_objectives": learning_objectives
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # -------- DISPLAY RESULTS --------
    print(f"\n✓ Saved output to: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("  GENERATED LEARNING OUTCOMES (ABCD)")
    print("=" * 70)
    for lo in learning_objectives:
        print(f"\n{lo}")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
