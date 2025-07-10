from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json

# Load your combined dataset
with open("data_combined.json") as f:
    raw_data = json.load(f)

print(f"📊 Loaded {len(raw_data)} training examples")

# Load tokenizer and base model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA with optimized settings
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,  # Increased rank
    lora_alpha=64,  # Increased alpha
    lora_dropout=0.05,  # Reduced dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Target specific modules
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

# Set up training with optimized parameters
training_args = TrainingArguments(
    output_dir="./finetuned-maths-tutor-improved",
    per_device_train_batch_size=1,  # Small batch for stability
    num_train_epochs=15,  # More epochs for better learning
    save_steps=20,  # Save every 20 steps
    logging_steps=5,  # Log every 5 steps
    save_total_limit=3,
    learning_rate=8e-6,  # Very conservative learning rate
    warmup_steps=20,  # More warmup
    weight_decay=0.01,  # Regularization
    gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
    fp16=True,  # Mixed precision
    dataloader_pin_memory=False,  # Avoid MPS issues
    evaluation_strategy="steps",  # Evaluate during training
    eval_steps=50,  # Evaluate every 50 steps
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="eval_loss",  # Use eval loss as metric
    greater_is_better=False,  # Lower loss is better
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train!
print("🚀 Starting improved training with combined dataset...")
print(f"📈 Training for {training_args.num_train_epochs} epochs")
print(f"🎯 Target: Better Zubrin number understanding")

trainer.train()

# Save
model.save_pretrained("./finetuned-maths-tutor-improved")
tokenizer.save_pretrained("./finetuned-maths-tutor-improved")
print("✅ Training complete! Model saved to ./finetuned-maths-tutor-improved") 