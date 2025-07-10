from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json

# Load your dataset
with open("data.json") as f:
    raw_data = json.load(f)

# Load tokenizer and base model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA with more conservative settings
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,  # Increased from 8
    lora_alpha=64,  # Increased from 32
    lora_dropout=0.05,  # Reduced from 0.1
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # More specific targeting
)
model = get_peft_model(model, peft_config)

# Convert to chat format for training
def format_chat_data(examples):
    formatted_texts = []
    for ex in examples:
        messages = [{"role": "user", "content": ex['instruction']}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Add the expected response
        full_text = chat_text + ex['output'] + tokenizer.eos_token
        formatted_texts.append({"text": full_text})
    return formatted_texts

# Format data using chat template
formatted_data = format_chat_data(raw_data)
data = Dataset.from_list(formatted_data)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized = data.map(tokenize, batched=True)

# Set up training with better parameters
training_args = TrainingArguments(
    output_dir="./finetuned-maths-tutor-v2",
    per_device_train_batch_size=1,  # Reduced for stability
    num_train_epochs=20,  # More epochs
    save_steps=10,
    logging_steps=10,
    save_total_limit=3,
    learning_rate=1e-5,  # Lower learning rate for stability
    warmup_steps=10,  # More warmup
    weight_decay=0.01,  # Add weight decay
    gradient_accumulation_steps=4,  # Effective batch size = 1 * 4 = 4
    fp16=True,  # Use mixed precision
    dataloader_pin_memory=False,  # Avoid MPS issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train!
print("Starting improved training...")
trainer.train()

# Save
model.save_pretrained("./finetuned-maths-tutor-v2")
tokenizer.save_pretrained("./finetuned-maths-tutor-v2")
print("Training complete! Model saved to ./finetuned-maths-tutor-v2") 