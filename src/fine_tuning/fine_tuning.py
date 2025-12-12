from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json

initial_dataset = load_dataset("samdotme/vader-speak")

# this function is used to output the right format for each row in the dataset
def create_text_row(instruction, output):
    text_row = f"""Question: {instruction}. Answer: {output}"""
    return text_row

# iterate over all the rows and store the final format as a giant text file
def save_file(output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for item in initial_dataset["train"]:
            output_file.write(create_text_row(item["prompt"], item["response"]) + "\n")

# Provide the path where we want to save the formatted dataset
save_file("./training_dataset.txt")

# We now load the formatted dataset from the text file
actual_dataset = load_dataset('text', data_files={'train': 'training_dataset.txt'}, encoding='utf-8')

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Important for GPT-2 to get correct results.
tokenizer.pad_token = tokenizer.eos_token

# Worth uncommenting for understanding of the dataset structure.
# print(dataset['train']['text'])

# Tokenize the data
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_datasets = actual_dataset.map(tokenize_function, batched=True)

model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

trainer.save_model("./vader_gpt2")
print()

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./vader_gpt2')

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# We house the LLM query logic in a function so we can call it easily later.
def take_prompt_output_response(prompt):
 # Tokenize the input prompt
  input_ids = tokenizer.encode(prompt, return_tensors='pt')

  # Create attention mask (1 for real tokens, 0 for padding tokens)
  attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

  # Generate text
  output = model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_length=100,  # Adjust the max length to control the output length
      num_return_sequences=1,
      no_repeat_ngram_size=2,
      top_k=50,
      top_p=0.95,
      temperature=0.7,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id to eos_token_id
  )

  # Decode and print the generated text
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  print(generated_text)

# Call the model with the prompt
take_prompt_output_response("Did you remember to buy your ventilator filters?")

take_prompt_output_response("The earth is filled with wonder")

