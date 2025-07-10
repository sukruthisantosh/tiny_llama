from transformers import AutoTokenizer
import json

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a few examples
with open("data.json") as f:
    raw_data = json.load(f)

print("=== DEBUGGING CHAT TEMPLATE FORMAT ===\n")

# Test the format_chat_data function
def format_chat_data(examples):
    formatted_texts = []
    for ex in examples:
        messages = [{"role": "user", "content": ex['instruction']}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Add the expected response
        full_text = chat_text + ex['output'] + tokenizer.eos_token
        formatted_texts.append({"text": full_text})
    return formatted_texts

# Test with first 3 examples
test_data = raw_data[:3]
formatted_data = format_chat_data(test_data)

for i, (original, formatted) in enumerate(zip(test_data, formatted_data)):
    print(f"Example {i+1}:")
    print(f"Original instruction: {original['instruction']}")
    print(f"Original output: {original['output']}")
    print(f"Formatted text:")
    print(formatted['text'])
    print("-" * 80)
    print() 