import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions, retry

# --- 1. CONFIGURATION ---
load_dotenv()
SOURCE_FILE = '../datasets/iiit_poc_monsoon_2024.json'
OUTPUT_FILE = '../datasets/iiit_course_descriptions_generated.json'

MODEL_NAME = 'models/gemini-flash-latest'

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(MODEL_NAME)
else:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

# --- 2. PROMPT ---
SYSTEM_PROMPT = """
Role: Academic Curriculum Designer.
Task: Write a professional Course Description (80-100 words).
Input: Course Title, Learning Objectives (LOs), Detailed Syllabus.

Instructions:
1. Opening: State the core domain and primary focus.
2. Body: Synthesize key topics from the Syllabus units, connecting them logically.
3. Closing: Mention practical outcomes derived from the LOs.
4. Tone: Academic, engaging, professional. No marketing fluff.

Output Format: Return ONLY the description text string.
"""

def format_user_prompt(course):
    syllabus_content = course.get('Detailed Syllabus', '')
    if isinstance(syllabus_content, list):
        syllabus_content = "\n".join(syllabus_content)
        
    return f"""
    Title: {course.get('Course Title', 'Unknown')}
    LOs: {json.dumps(course.get('LOs/COs', []))}
    Syllabus: {syllabus_content}
    """

def generate_description(course):
    try:
        response = model.generate_content(
            contents=[SYSTEM_PROMPT, format_user_prompt(course)]
        )
        return response.text.strip()
    except Exception as e:
        print(f"  [Error] {e}")
        return None

# --- 3. MAIN LOGIC ---
def main():
    # Load Data: Try to load the "work in progress" file first
    if os.path.exists(DATA_FILE):
        print(f"Resuming from {DATA_FILE}...")
        with open(DATA_FILE, 'r') as f:
            courses = json.load(f)
    elif os.path.exists(SOURCE_FILE):
        print(f"Starting fresh from {SOURCE_FILE}...")
        with open(SOURCE_FILE, 'r') as f:
            courses = json.load(f)
    else:
        print("Error: No data file found.")
        return

    print(f"Total Courses: {len(courses)}")

    # Loop through courses
    for i, course in enumerate(courses):
        title = course.get('Course Title', 'Unknown')
        
        # --- FIX 1: FLATTEN STRUCTURE ---
        # If the script previously made "Description_Candidates", move it to main field
        if 'Description_Candidates' in course:
            candidates = course['Description_Candidates']
            if 'Gemini' in candidates and candidates['Gemini']:
                course['Course Description'] = candidates['Gemini']
            del course['Description_Candidates'] # Clean up the mess

        # --- FIX 2: RESUME LOGIC ---
        # If we already have a description, skip this course
        current_desc = course.get('Course Description')
        if current_desc and len(current_desc) > 20:
            print(f"[{i+1}/{len(courses)}] Skipping '{title}' (Already done)")
            continue

        print(f"[{i+1}/{len(courses)}] Generating: '{title}'...")

        # Generate
        try:
            desc = generate_description(course)
            if desc:
                course['Course Description'] = desc # Direct assignment (Fix 1)
                print("  > Success")
                
                # Save Progress IMMEDIATELY (Fix 2)
                with open(DATA_FILE, 'w') as f:
                    json.dump(courses, f, indent=2)
            
            # Smart Sleep: 10s is usually safe for Flash tier
            time.sleep(10)

        except exceptions.ResourceExhausted:
            print("  > Quota Exceeded! Cooling down for 60 seconds...")
            time.sleep(60)
            # Retry once logic could go here, or just let the loop continue and fix it on next run
        except Exception as e:
            print(f"  > Unexpected Error: {e}")
            time.sleep(5)

    print("\nBatch Complete!")

if __name__ == "__main__":
    main()