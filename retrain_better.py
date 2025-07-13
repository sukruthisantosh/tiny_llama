from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json

# Load combined dataset
with open("data_combined.json") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} training examples")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,  # More parameters to learn with (was 8)
    lora_alpha=64,  # Scaling factor (was 32)
    lora_dropout=0.05,  # Less dropout for more stable training (was 0.1)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Focus on attention layers
)
model = get_peft_model(model, peft_config)

# Convert our data to the chat format that TinyLlama expects
def format_chat_data(examples):
    formatted_texts = []
    for ex in examples:
        messages = [{"role": "user", "content": ex['instruction']}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = chat_text + ex['output'] + tokenizer.eos_token
        formatted_texts.append({"text": full_text})
    return formatted_texts

# Format all data
formatted_data = format_chat_data(raw_data)
data = Dataset.from_list(formatted_data)

# Convert text to numbers (tokens) that the model can understand
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized = data.map(tokenize, batched=True)

# Training settings
training_args = TrainingArguments(
    output_dir="./finetuned-maths-tutor-v2",
    per_device_train_batch_size=1,  # Small batches to avoid memory issues
    num_train_epochs=20,  # More time to learn the concept properly
    save_steps=10,  # Save checkpoints every 10 steps
    logging_steps=10,  # Print progress every 10 steps (less spam)
    save_total_limit=3,  # Keep only 3 saved versions to save space
    learning_rate=1e-5,  # Much slower learning rate - the last one was too fast
    warmup_steps=10,  # Start learning slowly
    weight_decay=0.01,  # Prevent overfitting
    gradient_accumulation_steps=4,  # Effective batch size = 1 * 4 = 4
    fp16=False,  # Disabled for MPS (Apple Silicon) compatibility
    dataloader_pin_memory=False,  # Avoid issues with MPS (Apple Silicon)
)

# Set up the trainer with our model and data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("Starting the improved training run...")
print(f"Training for {training_args.num_train_epochs} epochs with {len(raw_data)} examples")
trainer.train()

# Save our trained model
model.save_pretrained("./finetuned-maths-tutor-v2")
tokenizer.save_pretrained("./finetuned-maths-tutor-v2")
print("Training complete! Model saved to ./finetuned-maths-tutor-v2") 