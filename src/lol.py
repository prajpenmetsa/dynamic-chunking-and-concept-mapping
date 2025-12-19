import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

    print("--- Available Models for Content Generation ---")
    try:
        # 2. List all models
        for m in genai.list_models():
            # 3. Filter for models that support text generation ('generateContent')
            if 'generateContent' in m.supported_generation_methods:
                print(f"Name: {m.name}")
                print(f"  Display Name: {m.display_name}")
                print(f"  Description: {m.description[:100]}...") # Truncated description
                print("-" * 40)
    except Exception as e:
        print(f"Error listing models: {e}")