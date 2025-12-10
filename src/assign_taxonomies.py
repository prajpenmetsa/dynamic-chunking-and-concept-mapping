import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from google.api_core import retry

# --- 1. SETUP ---
load_dotenv()

# We use the OUTPUT of Script 1 as the INPUT here to keep the pipeline moving
INPUT_FILE = 'iiit_course_descriptions_generated.json' 
OUTPUT_FILE = 'iiit_gold_standard_complete.json'

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemma-3-27b-it')

grok_client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# --- 2. PROMPT FOR PEDAGOGY ---
SYSTEM_PROMPT_TAXONOMY = """
Role: Expert Pedagogue and Educational Data Scientist.
Task: Analyze a list of Learning Objectives (LOs) using three specific frameworks: Bloom's Taxonomy, the ABCD Model, and the SMART Framework.

Input Data:
- Course Title
- List of Learning Objectives (LOs)

Instructions:
For EACH LO in the input list, generate a JSON object with the following fields:

1. "blooms_taxonomy":
   - "level": One of [Remember, Understand, Apply, Analyze, Evaluate, Create].
   - "justification": Brief reason based on the active verb.

2. "abcd_model":
   - "audience": Who is the learner? (e.g. "Student")
   - "behavior": What is the action? (e.g. "Design a compiler")
   - "condition": Context/Tools (e.g. "using C++", "given a dataset") - Return null if absent.
   - "degree": Standard of performance (e.g. "efficiently", "with 90% accuracy") - Return null if absent.

3. "smart_framework":
   - "is_measurable": boolean
   - "is_time_bound": boolean
   - "critique": A single sentence critiquing if the LO meets SMART goals.

Output Format: Return valid JSON only. Structure:
{
  "analysis": [
    { "original_text": "...", "blooms_taxonomy": {...}, "abcd_model": {...}, "smart_framework": {...} },
    ...
  ]
}
"""

# --- 3. HELPER FUNCTIONS ---

def get_pedagogy_input(course):
    return f"""
    Title: {course['Course Title']}
    LOs: {json.dumps(course['LOs/COs'])}
    """

@retry.Retry(predicate=retry.if_exception_type(Exception))
def analyze_gemini(course):
    try:
        response = gemini_model.generate_content(
            contents=[SYSTEM_PROMPT_TAXONOMY, get_pedagogy_input(course)],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [Gemini Error] {e}")
        return None

def analyze_grok(course):
    try:
        response = grok_client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TAXONOMY},
                {"role": "user", "content": get_pedagogy_input(course)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [Grok Error] {e}")
        return None

# --- 4. MAIN LOOP ---
def main():
    try:
        with open(INPUT_FILE, 'r') as f:
            courses = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run Script 1 first.")
        return

    print(f"--- Starting Taxonomy Assignment for {len(courses)} courses ---")

    for i, course in enumerate(courses):
        print(f"[{i+1}/{len(courses)}] {course['Course Title']}")
        
        if 'Pedagogy_Candidates' not in course:
            course['Pedagogy_Candidates'] = {}

        # 1. Gemini Analysis
        print("  > Gemini Analysis...")
        gemini_res = analyze_gemini(course)
        if gemini_res:
            course['Pedagogy_Candidates']['Gemini'] = gemini_res.get('analysis', [])

        # 2. Grok Analysis
        print("  > Grok Analysis...")
        grok_res = analyze_grok(course)
        if grok_res:
            course['Pedagogy_Candidates']['Grok'] = grok_res.get('analysis', [])
            
        time.sleep(1)

    # Save Final Gold Standard
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(courses, f, indent=2)
    print(f"\nSuccess! Full dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()