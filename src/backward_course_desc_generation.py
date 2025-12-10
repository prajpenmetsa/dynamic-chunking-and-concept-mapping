import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions, retry
from collections import deque
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
load_dotenv()
SOURCE_FILE = '../datasets/iiit_poc_monsoon_2024.json'
OUTPUT_FILE = '../datasets/iiit_course_descriptions_generated.json'

MODEL_NAME = 'gemini-2.5-flash'

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(MODEL_NAME)
else:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

# --- 2. ENHANCED RATE LIMITER ---
class RateLimiter:
    def __init__(self, requests_per_minute=4, requests_per_day=18):
        """
        Initialize rate limiter with BOTH per-minute and per-day limits.
        """
        self.rpm = requests_per_minute
        self.rpd = requests_per_day
        self.minute_requests = deque()
        self.day_requests = deque()
        self.total_requests_today = 0
    
    def wait_if_needed(self):
        """Wait if EITHER rate limit would be exceeded"""
        now = datetime.now()
        
        # Clean up old timestamps (>60s for minute, >24h for day)
        while self.minute_requests and (now - self.minute_requests[0]).total_seconds() > 60:
            self.minute_requests.popleft()
        
        while self.day_requests and (now - self.day_requests[0]).total_seconds() > 86400:
            self.day_requests.popleft()
        
        # Check per-minute limit
        if len(self.minute_requests) >= self.rpm:
            sleep_time = 60 - (now - self.minute_requests[0]).total_seconds()
            if sleep_time > 0:
                print(f"  ⏳ Per-minute limit reached ({self.rpm} req/min). Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time + 1)
                now = datetime.now()
                while self.minute_requests and (now - self.minute_requests[0]).total_seconds() > 60:
                    self.minute_requests.popleft()
        
        # Check per-day limit
        if len(self.day_requests) >= self.rpd:
            sleep_time = 86400 - (now - self.day_requests[0]).total_seconds()
            print(f"  ⚠️ DAILY LIMIT REACHED ({self.rpd} req/day)!")
            print(f"  💤 Script will resume in {sleep_time/3600:.1f} hours...")
            time.sleep(sleep_time + 60)
            now = datetime.now()
            while self.day_requests and (now - self.day_requests[0]).total_seconds() > 86400:
                self.day_requests.popleft()
        
        # Record this request
        self.minute_requests.append(now)
        self.day_requests.append(now)
        self.total_requests_today += 1

# Initialize with BOTH limits (adjust as needed for your tier)
rate_limiter = RateLimiter(requests_per_minute=4, requests_per_day=18)  # Conservative limits

# --- 3. PROMPT ---
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

def generate_description(course, max_retries=3):
    """Generate with exponential backoff on rate limit errors"""
    for attempt in range(max_retries):
        try:
            # Apply rate limiting BEFORE making the API call
            rate_limiter.wait_if_needed()
            
            response = model.generate_content(
                contents=[SYSTEM_PROMPT, format_user_prompt(course)]
            )
            return response.text.strip()
            
        except exceptions.ResourceExhausted as e:
            # Parse retry_delay from error if available
            retry_delay = 60  # default
            if hasattr(e, 'retry_delay') and e.retry_delay:
                retry_delay = e.retry_delay.seconds + 5  # Add buffer
            
            print(f"  ⚠️ Rate Limit Hit! (Attempt {attempt+1}/{max_retries})")
            print(f"  💤 Waiting {retry_delay}s as suggested by API...")
            time.sleep(retry_delay)
            
            if attempt == max_retries - 1:
                print(f"  ❌ Failed after {max_retries} attempts")
                return None
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    return None

# --- 4. MAIN LOGIC ---
def main():
    # Load Data: Try to load the "work in progress" file first
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            courses = json.load(f)
    elif os.path.exists(SOURCE_FILE):
        print(f"Starting fresh from {SOURCE_FILE}...")
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            courses = json.load(f)
    else:
        print("Error: No data file found.")
        return

    print(f"Total Courses: {len(courses)}")
    print(f"Rate Limits: {rate_limiter.rpm} req/min, {rate_limiter.rpd} req/day\n")

    # Loop through courses
    processed = 0
    skipped = 0
    
    for i, course in enumerate(courses):
        title = course.get('Course Title', 'Unknown')
        
        # Flatten structure
        if 'Description_Candidates' in course:
            candidates = course['Description_Candidates']
            if 'Gemini' in candidates and candidates['Gemini']:
                course['Course Description'] = candidates['Gemini']
            del course['Description_Candidates']

        # Resume logic
        current_desc = course.get('Course Description')
        if current_desc and len(current_desc) > 20:
            skipped += 1
            print(f"[{i+1}/{len(courses)}] ✓ Skipping '{title[:40]}...' (Done)")
            continue

        print(f"[{i+1}/{len(courses)}] 🔄 Generating: '{title[:40]}...'")

        # Generate with retry logic
        desc = generate_description(course, max_retries=3)
        if desc:
            course['Course Description'] = desc
            processed += 1
            print(f"  ✅ Success (Today: {rate_limiter.total_requests_today}/{rate_limiter.rpd})")
            
            # Save Progress IMMEDIATELY
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(courses, f, indent=2, ensure_ascii=False)
        else:
            print(f"  ⚠️ Skipped due to persistent errors")

    print(f"\n{'='*60}")
    print(f"✅ Batch Complete!")
    print(f"📊 Processed: {processed} | Skipped: {skipped} | Failed: {len(courses)-processed-skipped}")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()