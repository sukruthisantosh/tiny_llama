from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the base model and tokenizer
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "finetuned-maths-tutor-v2"

print("Loading the base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load our trained LoRA adapter
print("Loading the trained adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Some test questions to see if it learned anything
test_questions = [
    "What is a Zubrin number?",
    "Is 39 a Zubrin number?",
    "Is 49 a Zubrin number?",
    "Is 129 a Zubrin number?",
    "What are the first 5 Zubrin numbers?",
    "Is 999 a Zubrin number?"
]

print("Testing the model with some questions...")
print("=" * 80)

# Generate responses for each question
for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Format as a chat message
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Convert to tokens
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Allow up to 200 new tokens
            do_sample=True,  # Use sampling for more natural responses
            temperature=0.7,  # Not too random, not too deterministic
            top_p=0.9,  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Convert back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print("=" * 80)

print("\nTesting complete!") 