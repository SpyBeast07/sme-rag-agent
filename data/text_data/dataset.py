from datasets import load_dataset

dataset = load_dataset("yasserrmd/food-safety")

print(dataset)
print(dataset['train'][0])

with open("data/food_safety_texts.txt", "w") as f:
    for item in dataset["train"]:
        text_block = f"""
Question: {item['prompt'].strip()}

Answer: {item['completion'].strip()}
        """
        f.write(text_block + "\n\n")





from datasets import load_dataset
import os

squad = load_dataset("rajpurkar/squad")

print(squad)
print(squad['train'][0])

# Use a set to automatically remove duplicates
unique_contexts = set(item["context"].strip() for item in squad["train"])

print(f"Number of unique contexts: {len(unique_contexts)}")

with open("data/squad_contexts.txt", "w") as f:
    for context in unique_contexts:
        f.write(context + "\n\n")  # add double newline between contexts