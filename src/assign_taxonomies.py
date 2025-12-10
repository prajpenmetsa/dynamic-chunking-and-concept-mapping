import json
import time
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import retry

# --- 1. SETUP ---
load_dotenv()

# Input File (from previous step)
INPUT_FILE = '../datasets/iiit_course_descriptions_generated.json'

# Output Files
OUT_BLOOMS = '../datasets/iiit_taxonomy_blooms.json'
OUT_ABCD = '../datasets/iiit_taxonomy_abcd.json'
OUT_SMART = '../datasets/iiit_taxonomy_smart.json'

# Configure Gemini
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # Using the specific Gemma model you requested
    model = genai.GenerativeModel('models/gemma-3-27b-it')
else:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

# --- 2. THE PROMPT ---
SYSTEM_PROMPT = """
Role: Expert Pedagogue and Educational Data Scientist.
Task: Analyze a list of Learning Objectives (LOs) using three specific frameworks: Bloom's Taxonomy, the ABCD Model, and the SMART Framework.

Input Data:
- Course Title
- List of Learning Objectives (LOs)

Instructions:
For EACH LO in the input list, generate a JSON object with the following fields. Always base your analysis ONLY on the text of the LO itself.

1. "blooms_taxonomy":
   - "level": One of [Remember, Understand, Apply, Analyze, Evaluate, Create].
     - Remember: recalling or recognizing facts, terms, or basic concepts.
     - Understand: explaining ideas or concepts, interpreting, summarizing, classifying.
     - Apply: using procedures or concepts in new situations, solving problems with known methods.
     - Analyze: breaking material into parts, comparing, organizing, attributing, finding relationships.
     - Evaluate: making judgments, critiquing, justifying decisions using criteria or standards.
     - Create: generating new products, designs, or original work by combining elements.
   - Choose the single dominant level based on the main observable action verb in the LO.
   - "justification": Brief reason (1–2 sentences) explicitly linking the LO’s main verb(s) to the chosen level.

2. "abcd_model":
   - Interpret the LO according to the ABCD model:
     - "audience": Who is the learner? If not explicitly stated, use a generic term like "Student".
     - "behavior": The main observable action the learner should perform, written as a short verb phrase (e.g., "design a compiler", "classify images using CNNs").
     - "condition": The context, tools, or given resources (e.g., "given a dataset", "using Python", "in the lab"). If the LO does not clearly specify a condition, return null.
     - "degree": The explicit standard of performance (e.g., "with 90% accuracy", "with no syntax errors"). If the LO does not specify any performance criterion, return null.
   - Do NOT invent conditions or degrees that are not clearly implied by the LO text.

3. "smart_framework":
   - Focus on two SMART properties:
     - "is_measurable": boolean
       - true if the LO describes an observable behavior that could be assessed.
       - false if the LO is too vague or affective.
     - "is_time_bound": boolean
       - true only if the LO explicitly includes a time frame.
       - false otherwise.
   - "critique": A single sentence critiquing whether the LO meets SMART principles overall.

Output Format: Return valid JSON only. NO Markdown. Structure:
{
  "analysis": [
    { "original_text": "...", "blooms_taxonomy": {...}, "abcd_model": {...}, "smart_framework": {...} },
    ...
  ]
}
"""

# --- 3. HELPER FUNCTIONS ---
def get_user_content(course):
    return f"""
    Title: {course.get('Course Title', 'Unknown')}
    LOs: {json.dumps(course.get('LOs/COs', []))}
    """

def extract_json(text):
    """Clean markdown code blocks from response."""
    try:
        # If response is wrapped in ```json ... ```, strip it
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception:
        return None

@retry.Retry(predicate=retry.if_exception_type(Exception))
def generate_taxonomies(course):
    try:
        # Removed generation_config={"response_mime_type": "application/json"}
        response = model.generate_content(
            contents=[SYSTEM_PROMPT, get_user_content(course)]
        )
        return extract_json(response.text)
    except Exception as e:
        print(f"  [Error] {e}")
        return None

def save_files(blooms_data, abcd_data, smart_data):
    with open(OUT_BLOOMS, 'w') as f: json.dump(blooms_data, f, indent=2)
    with open(OUT_ABCD, 'w') as f: json.dump(abcd_data, f, indent=2)
    with open(OUT_SMART, 'w') as f: json.dump(smart_data, f, indent=2)

# --- 4. MAIN PIPELINE ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        courses = json.load(f)

    # Initialize separate lists for each output file
    all_blooms = []
    all_abcd = []
    all_smart = []

    print(f"--- Starting Taxonomy Analysis for {len(courses)} courses ---")

    for i, course in enumerate(courses):
        title = course.get('Course Title', 'Unknown')
        print(f"[{i+1}/{len(courses)}] Analyzing: {title}")

        # Call Gemini
        result = generate_taxonomies(course)

        if result and 'analysis' in result:
            # Process the results into separate structures
            blooms_entry = {"Course Title": title, "LO_Analysis": []}
            abcd_entry = {"Course Title": title, "LO_Analysis": []}
            smart_entry = {"Course Title": title, "LO_Analysis": []}

            for item in result['analysis']:
                lo_text = item.get('original_text', '')
                
                # Split the data
                blooms_entry["LO_Analysis"].append({
                    "original_lo": lo_text,
                    "blooms_taxonomy": item.get('blooms_taxonomy')
                })
                abcd_entry["LO_Analysis"].append({
                    "original_lo": lo_text,
                    "abcd_model": item.get('abcd_model')
                })
                smart_entry["LO_Analysis"].append({
                    "original_lo": lo_text,
                    "smart_framework": item.get('smart_framework')
                })

            # Append to main lists
            all_blooms.append(blooms_entry)
            all_abcd.append(abcd_entry)
            all_smart.append(smart_entry)
            
            print("  > Success")
        else:
            print("  > Failed or Empty Response")

        # Save periodically (every 5 courses)
        if i % 5 == 0:
            save_files(all_blooms, all_abcd, all_smart)
        
        # Rate Limit Safety
        time.sleep(2)

    # Final Save
    save_files(all_blooms, all_abcd, all_smart)
    print("\nProcessing Complete!")
    print(f"1. Bloom's Data saved to: {OUT_BLOOMS}")
    print(f"2. ABCD Data saved to:   {OUT_ABCD}")
    print(f"3. SMART Data saved to:  {OUT_SMART}")

if __name__ == "__main__":
    main()