from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json

# Load your dataset - handle JSON array format
with open("data.json") as f:
    raw_data = json.load(f)

# Load tokenizer and base model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
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

# Set up training with more epochs
training_args = TrainingArguments(
    output_dir="./finetuned-maths-tutor",
    per_device_train_batch_size=2,
    num_train_epochs=10,  # Increased from 3 to 10
    save_steps=5,  # Save more frequently
    logging_steps=1,  # Log every step
    save_total_limit=3,
    learning_rate=5e-5,  # Slightly higher learning rate
    warmup_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train!
trainer.train()

# Save
model.save_pretrained("./finetuned-maths-tutor")
tokenizer.save_pretrained("./finetuned-maths-tutor")
