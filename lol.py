import google.generativeai as genai
genai.configure(api_key="AIzaSyD_s2xGBHYoS11jF8GTLsIbBvjuwv2xkTY")

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)