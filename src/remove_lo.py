# iteration 1: generate LOs using course descriptions as input alone
# create json file with only course details without LOs/COs

import json

def create_input_dataset(input_file, output_file):
    try:
        # Load the original dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each course entry
        cleaned_data = []
        for course in data:
            # Create a new dictionary excluding the 'LOs/COs' key
            new_entry = {key: value for key, value in course.items() if key != 'LOs/COs'}
            cleaned_data.append(new_entry)
            
        # Save the new dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
        print(f"Success! Created '{output_file}' with {len(cleaned_data)} courses.")
        print("The 'LOs/COs' field has been removed from all entries.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'.")

# Execute the function
input_filename = '../datasets/iiit_course_descriptions_generated.json'
output_filename = '../datasets/iiit_courses_input_only.json'

create_input_dataset(input_filename, output_filename)