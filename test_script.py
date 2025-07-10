from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the base model and tokenizer
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "finetuned-maths-tutor"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Set to evaluation mode
model.eval()

# Test questions
test_questions = [
    "What is a Zubrin number?",
    "Is 39 a Zubrin number?",
    "Is 49 a Zubrin number?",
    "Is 129 a Zubrin number?",
    "What are the first 5 Zubrin numbers?",
    "Is 999 a Zubrin number?"
]

# Generate responses
for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Format as chat
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print("=" * 80)
