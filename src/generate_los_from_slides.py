"""
Simple approach: Extract all slides → Generate LOs in ONE API call
Avoids rate limiting by making only 1 request
"""

import json
import os
import glob
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import pdfplumber
from tqdm import tqdm

# ==================== CONFIGURATION ====================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found!")

genai.configure(api_key=API_KEY)

SLIDE_DECKS_FOLDER = "../raw-data/osn_lecs"
COURSE_TITLE = "Advanced Operating Systems"
COURSE_CODE = "CS3.304"
OUTPUT_FILE = "../datasets/slide_based_los_simple.json"

# Use model with highest free tier quota
MODEL_NAME = "gemini-2.5-flash"


# ==================== SLIDE TEXT EXTRACTION ====================
def extract_all_slides_text(folder_path: str, max_chars: int = 50000) -> str:
    """
    Extract text from ALL PDFs in folder.
    Truncate to max_chars to fit in context window.
    """
    all_text = []
    total_chars = 0
    
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    
    print(f"\n📚 Extracting text from {len(pdf_files)} PDF files...")
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        filename = os.path.basename(pdf_path)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                file_text = []
                
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        file_text.append(text.strip())
                
                # Add section header for this lecture
                lecture_text = f"\n\n=== {filename} ===\n" + "\n".join(file_text)
                
                # Check if adding this would exceed limit
                if total_chars + len(lecture_text) > max_chars:
                    print(f"\n⚠️  Reached {max_chars} char limit at {filename}")
                    break
                
                all_text.append(lecture_text)
                total_chars += len(lecture_text)
        
        except Exception as e:
            print(f"\n⚠️  Error reading {filename}: {e}")
            continue
    
    combined_text = "\n".join(all_text)
    print(f"\n✓ Extracted {total_chars:,} characters from {len(all_text)} files")
    
    return combined_text


# ==================== LO GENERATION PROMPT ====================
def create_lo_generation_prompt(course_title: str, course_code: str, slide_text: str) -> str:
    """
    Single prompt that generates LOs directly from slide text.
    Based on your Iteration 1 approach.
    """
    return f"""
You are an expert Educational Curriculum Designer.

**Task**: Create 6-7 comprehensive Learning Outcomes (LOs) for this university course based on lecture slide content.

**Course Information**:
- Title: {course_title}
- Code: {course_code}

**Lecture Slide Content** (from multiple lecture decks):
{slide_text}

**Instructions**:
1. Analyze the slide content to identify major topics and concepts
2. Generate 6-7 learning outcomes that:
   - Cover the breadth of topics in the slides
   - Follow Bloom's Taxonomy progression (foundational → advanced)
   - Are specific and measurable
   - Use appropriate Bloom's action verbs

**Bloom's Taxonomy Action Verbs**:
- Remember: Define, List, Identify, Recall, State
- Understand: Explain, Describe, Summarize, Interpret, Compare
- Apply: Apply, Implement, Execute, Demonstrate, Use
- Analyze: Analyze, Examine, Differentiate, Investigate, Compare
- Evaluate: Evaluate, Assess, Critique, Judge, Justify
- Create: Design, Construct, Develop, Formulate, Synthesize

**Output Format** (JSON array ONLY, no markdown):
[
  "CO-1: [Bloom Verb] [specific technical content related to slides]",
  "CO-2: [Bloom Verb] [specific technical content related to slides]",
  "CO-3: [Bloom Verb] [specific technical content related to slides]",
  "CO-4: [Bloom Verb] [specific technical content related to slides]",
  "CO-5: [Bloom Verb] [specific technical content related to slides]",
  "CO-6: [Bloom Verb] [specific technical content related to slides]",
  "CO-7: [Bloom Verb] [specific technical content related to slides]"
]

**Quality Requirements**:
✓ Each LO must start with a Bloom's Taxonomy verb
✓ Be specific to the technical content in the slides
✓ Progress from lower-order to higher-order thinking
✓ Cover diverse major topics from the lectures
✓ Avoid redundancy between outcomes
✓ Be measurable and actionable

Return ONLY the JSON array, no other text.
"""


# ==================== MAIN EXECUTION ====================
def main():
    print("="*70)
    print("  SIMPLE SLIDE-BASED LO GENERATION")
    print("  Single API call approach - avoids rate limiting")
    print("="*70)
    
    # Step 1: Extract all slide text
    slide_text = extract_all_slides_text(SLIDE_DECKS_FOLDER, max_chars=50000)
    
    if not slide_text or len(slide_text) < 500:
        print("\n❌ Insufficient slide text extracted!")
        return
    
    # Step 2: Generate LOs with single API call
    print("\n📝 Generating learning objectives...")
    print("   (This uses only 1 API call)")
    
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )
    
    prompt = create_lo_generation_prompt(COURSE_TITLE, COURSE_CODE, slide_text)
    
    try:
        response = model.generate_content(prompt)
        learning_objectives = json.loads(response.text)
        
        if not isinstance(learning_objectives, list):
            print("❌ Invalid response format!")
            return
        
        print(f"\n✓ Successfully generated {len(learning_objectives)} learning objectives!")
        
    except Exception as e:
        print(f"\n❌ Error generating LOs: {e}")
        return
    
    # Step 3: Save output
    output = {
        "course_title": COURSE_TITLE,
        "course_code": COURSE_CODE,
        "metadata": {
            "source": "Lecture slide decks",
            "folder": SLIDE_DECKS_FOLDER,
            "extraction_method": "Single API call with slide text",
            "total_chars_processed": len(slide_text),
            "generation_method": "Direct LO generation from slides (similar to Iteration 1)"
        },
        "learning_objectives": learning_objectives
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved output to: {OUTPUT_FILE}")
    
    # Display results
    print("\n" + "="*70)
    print("  GENERATED LEARNING OBJECTIVES")
    print("="*70)
    for lo in learning_objectives:
        print(f"\n{lo}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()