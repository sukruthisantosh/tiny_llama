import json

# Load original data
with open('data.json') as f:
    original = json.load(f)

# Load additional data
with open('additional_data.json') as f:
    additional = json.load(f)

# Combine datasets
combined = original + additional

# Save combined dataset
with open('data_combined.json', 'w') as f:
    json.dump(combined, f, indent=4)

print(f"✅ Combined {len(original)} + {len(additional)} = {len(combined)} examples")
print(f"📁 Saved to: data_combined.json") 