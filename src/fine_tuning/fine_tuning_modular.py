from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json


def load_initial_dataset(dataset_name="samdotme/vader-speak"):
    """Load the initial dataset from Hugging Face."""
    return load_dataset(dataset_name)


def create_text_row(instruction, output, format_template=None):
    """
    Format a single row of data.
    
    Args:
        instruction: The input prompt/question
        output: The expected response/answer
        format_template: Optional custom format string with {instruction} and {output} placeholders
    """
    if format_template:
        return format_template.format(instruction=instruction, output=output)
    return f"""Question: {instruction}. Answer: {output}"""


def create_training_file(dataset, output_file_path, prompt_key="prompt", response_key="response", format_template=None):
    """
    Save the formatted dataset to a text file.
    
    Args:
        dataset: The dataset object to process
        output_file_path: Path where the training file will be saved
        prompt_key: Key name for the prompt in the dataset
        response_key: Key name for the response in the dataset
        format_template: Optional custom format string
    """
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for item in dataset["train"]:
            output_file.write(create_text_row(item[prompt_key], item[response_key], format_template) + "\n")
    print(f"Training dataset saved to {output_file_path}")


def load_training_dataset(training_file_path):
    """Load the formatted training dataset from a text file."""
    return load_dataset('text', data_files={'train': training_file_path}, encoding='utf-8')


def setup_tokenizer(model_name='gpt2'):
    """
    Initialize and configure the tokenizer.
    
    Args:
        model_name: Name of the pretrained model to load tokenizer from
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset for training.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
    """
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs
    
    return dataset.map(tokenize_function, batched=True)


def train_model(
    tokenized_dataset,
    model_name='gpt2',
    output_dir="./results",
    model_save_path="./vader_gpt2",
    num_epochs=3,
    batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200
):
    """
    Train the language model.
    
    Args:
        tokenized_dataset: The tokenized dataset
        model_name: Name of the base model to fine-tune
        output_dir: Directory for training outputs
        model_save_path: Path to save the final model
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        save_steps: Save checkpoint every X steps
        save_total_limit: Maximum number of checkpoints to keep
        logging_dir: Directory for training logs
        logging_steps: Log every X steps
    
    Returns:
        Trained model
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
    )
    
    print("Starting training...")
    trainer.train()
    
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model


def load_trained_model(model_path, tokenizer_name='gpt2'):
    """
    Load a fine-tuned model and its tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model
        tokenizer_name: Name of the tokenizer to load
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def generate_response(
    prompt,
    model,
    tokenizer,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True
):
    """
    Generate text response from a prompt.
    
    Args:
        prompt: Input text prompt
        model: The language model
        tokenizer: The tokenizer
        max_length: Maximum length of generated text
        num_return_sequences: Number of sequences to generate
        no_repeat_ngram_size: Size of n-grams that can only occur once
        top_k: Number of highest probability tokens to keep for filtering
        top_p: Cumulative probability for nucleus sampling
        temperature: Sampling temperature (higher = more random)
        do_sample: Whether to use sampling or greedy decoding
    
    Returns:
        Generated text string
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main pipeline for training and testing the model."""
    # Configuration
    DATASET_NAME = "databricks/databricks-dolly-15k"
    TRAINING_FILE = "./training_dataset.txt"
    MODEL_SAVE_PATH = "./databricks_dolly_gpt2"
    
    # Step 1: Load and prepare dataset
    print("Loading initial dataset...")
    initial_dataset = load_initial_dataset(DATASET_NAME)
    
    # Step 2: Create training file
    print("Creating training file...")
    create_training_file(initial_dataset, TRAINING_FILE)
    
    # Step 3: Load training dataset
    print("Loading training dataset...")
    actual_dataset = load_training_dataset(TRAINING_FILE)
    
    # Step 4: Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer()
    
    # Step 5: Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_datasets = tokenize_dataset(actual_dataset, tokenizer)
    
    # Step 6: Train model
    trained_model = train_model(
        tokenized_datasets,
        model_save_path=MODEL_SAVE_PATH,
        num_epochs=3,
        batch_size=4
    )
    
    # Step 7: Test the model
    print("\n" + "="*50)
    print("Testing the fine-tuned model:")
    print("="*50 + "\n")
    
    model, tokenizer = load_trained_model(MODEL_SAVE_PATH)
    
    test_prompts = [
        "Did you remember to buy your ventilator filters?",
        "The earth is filled with wonder",
        "How many legs does a spider have?"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {generate_response(prompt, model, tokenizer)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
