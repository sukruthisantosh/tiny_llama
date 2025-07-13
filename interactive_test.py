from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the base model and tokenizer
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "finetuned-maths-tutor-v2"

print("Loading your trained Zubrin number expert...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print("✅ Your Zubrin number expert is ready!")
print("Ask me anything about Zubrin numbers!")
print("Type 'quit' to exit\n")

# Interactive chat loop
while True:
    # Get user input
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye! 👋")
        break
    
    if not user_input:
        continue
    
    print("AI: ", end="", flush=True)
    
    try:
        # Format as chat
        messages = [{"role": "user", "content": user_input}]
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
        
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            assistant_response = response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        else:
            assistant_response = response
        
        print(assistant_response)
        
    except Exception as e:
        print(f"Sorry, I encountered an error: {e}")
    
    print()  # Empty line for readability 