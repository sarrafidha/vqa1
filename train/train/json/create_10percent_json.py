import json

# Path to the original JSON file
input_path = 'C:\\Users\\sarra\\Downloads\\changechat\\train\\train\\json\\Train_all.json'
# Path to the output JSON file
output_path = 'C:\\Users\\sarra\\Downloads\\changechat\\train\\train\\json\\Train_50images.json'

# Load the original data
with open(input_path, 'r') as f:
    data = json.load(f)

questions = data['question']

# Select only the first 50 questions
selected = questions[:50]

# Write the selected questions to the new JSON file
with open(output_path, 'w') as f:
    json.dump({'question': selected}, f, indent=2)

print(f"Selected the first 50 questions out of {len(questions)}. Saved to {output_path}.") 