import json
import time
import os
import re
from huggingface_hub import InferenceClient
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not API_KEY:
    raise ValueError("Hugging Face API Key not found! Please check your .env file.")

# --- CONFIGURATION ---
INPUT_FILE = '../datasets/iiit_courses_without_los_iteration_1.json'
OUTPUT_FILE = '../datasets/iiit_courses_generated_los_iteration_1.json'

MODEL_NAME = "gemini-2.5-lite"

def create_prompt(course_title, description, syllabus):
    return f"""You are an expert Educational Curriculum Designer.

Task: Create a set of 5-7 Learning Outcomes (LOs) for a university course.

Input Data:
- Course Title: {course_title}
- Description: {description}
- Syllabus Topics: {json.dumps(syllabus)}

Constraints:
1. Output MUST be a raw JSON list of strings. 
2. Format: ["CO1: ...", "CO2: ...", "CO3: ..."]
3. Use Bloom's Taxonomy action verbs (Demonstrate, Analyze, Design, Develop, Apply, Evaluate, Create, etc.).
4. Each LO should be specific, measurable, and aligned with the course content.
5. No markdown formatting (no ```json), just the list.

Respond with ONLY the JSON list, nothing else."""

def load_existing_progress():
    """Checks if output file exists and loads it to resume progress."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def generate_learning_objectives():
    # 1. Load Input Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_courses = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2. Resume Logic (Don't re-do work!)
    processed_courses = load_existing_progress()
    processed_titles = {c.get("Course Title") for c in processed_courses}
    
    courses_to_process = [c for c in all_courses if c.get("Course Title") not in processed_titles]
    
    print(f"Total Courses: {len(all_courses)}")
    print(f"Already Done: {len(processed_courses)}")
    print(f"Remaining: {len(courses_to_process)}")
    print(f"Using Model: {MODEL_NAME}")

    # 3. Initialize Hugging Face Inference Client
    client = InferenceClient(api_key=API_KEY)
    
    # 4. Processing Loop
    results = processed_courses # Start with what we already have
    
    for course in tqdm(courses_to_process):
        title = course.get("Course Title", "Unknown")
        desc = course.get("Course Description", "")
        syllabus = course.get("Detailed Syllabus", [])
        
        if not desc:
            course["Generated_LOs"] = []
            results.append(course)
            continue

        prompt = create_prompt(title, desc, syllabus)
        
        try:
            # Call Hugging Face Inference API
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks if present)
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                generated_los = json.loads(json_str)
            else:
                generated_los = json.loads(response_text)
            
            course["Generated_LOs"] = generated_los
            course["Generation_Method"] = "Llama3_70B_ZeroShot_v1"
            
        except Exception as e:
            print(f"Error for {title}: {e}")
            course["Generated_LOs"] = [] # Empty list on failure
            if "429" in str(e) or "400" in str(e):
                print("Quota hit or API Error. Saving progress and stopping.")
                break
        
        results.append(course)
        
        # Save every 5 items so you don't lose progress if it crashes
        if len(results) % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        time.sleep(4) # Respect rate limits

    # Final Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done! Final dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_learning_objectives()