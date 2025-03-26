import re
import json
import random
import logging
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set random seed for reproducibility
random.seed(310)

def sanitize_filename(text: str) -> str:
    """
    Replace all non-alphanumeric characters in 'text' with underscores,
    keeping letters, numbers, and underscores.
    """
    sanitized = re.sub(r"\W", "_", text)
    logging.debug("Sanitized filename: '%s' -> '%s'", text, sanitized)
    return sanitized

LABEL_PATTERN = re.compile(r"^<label>(negative|neutral|positive)</label>$")

def verify_and_extract(text: str, label: str) -> list[bool]:
    """
    Verify if 'text' matches the format:
        <label>(negative|neutral|positive)</label>

    Returns a list of two booleans:
      - The first is True if the format is valid, False otherwise.
      - The second is True if, when the format is valid, the extracted label
        exactly matches the provided 'label'. If the format is invalid, the first
        token of the string is checked to see if it contains 'label'.

    Examples:
        verify_and_extract("<label>negative</label>", "negative")  --> [True, True]
        verify_and_extract("negative", "negative")                  --> [False, True]
    """
    expected_label = LABEL_PATTERN.fullmatch(label).group(1)
    match = LABEL_PATTERN.fullmatch(text)
    if match:
        result = [True, match.group(1) == expected_label]
        logging.debug("Successful verification: '%s' with expected '%s' -> %s", text, expected_label, result)
        return result
    else:
        first_token = text.strip().split(" ")[0]
        result = [False, expected_label in first_token]
        logging.debug("Fallback verification: '%s' with expected '%s' -> %s", text, expected_label, result)
        return result

def main():
    logging.info("Starting main process.")

    # Load data
    data_path = Path("./data")
    data_file = data_path / "test.json"
    logging.info("Loading data from %s", data_file)
    with data_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    logging.info("Data loaded: %d samples found.", len(data))

    # Model configuration
    MODEL_NAME = "gemma3:1b"
    TRAIN_SAMPLES = 5000
    model_checkpoint = f"./lora_output/model_{TRAIN_SAMPLES}/checkpoint"
    model_class = AutoModelForCausalLM
    logging.info("Configuring model: %s", MODEL_NAME)
    logging.info("Using %d training samples.", TRAIN_SAMPLES)

    # Determine torch dtype based on GPU capabilities
    device_capability = torch.cuda.get_device_capability()[0]
    torch_dtype = torch.bfloat16 if device_capability >= 8 else torch.float16
    logging.info("Detected device capability: %d. Using torch_dtype: %s", device_capability, torch_dtype)

    # Load model and tokenizer
    logging.info("Loading model from checkpoint: %s", model_checkpoint)
    model = model_class.from_pretrained(
        model_checkpoint,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    logging.info("Model loaded successfully.")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    logging.info("Tokenizer loaded successfully.")

    # Set up text generation pipeline
    logging.info("Creating text generation pipeline.")
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
        truncation=True,
    )

    stop_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    logging.info("Stop tokens configured: %s", stop_token_ids)

    results = []
    NUM_TEST_SAMPLES = 1000
    logging.info("Starting text generation for %d samples.", NUM_TEST_SAMPLES)
    for idx, sample in enumerate(tqdm(data[:NUM_TEST_SAMPLES], desc="Processing samples")):
        # Log every 100 samples to avoid excessive logging
        if idx % 100 == 0:
            logging.info("Processing sample number: %d", idx)

        expected_label = sample["messages"][-1]["content"]
        logging.debug("Expected label: %s", expected_label)
        prompt = text_gen_pipeline.tokenizer.apply_chat_template(
            sample["messages"][:2],
            tokenize=False,
            add_generation_prompt=True
        )
        logging.debug("Generated prompt: %s", prompt)
        outputs = text_gen_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            eos_token_id=stop_token_ids,
            disable_compile=True,
        )
        generated_text = outputs[0]['generated_text'][len(prompt):].strip()
        logging.debug("Generated text: %s", generated_text)
        verification_result = verify_and_extract(generated_text, expected_label)
        logging.debug("Verification result: %s", verification_result)
        results.append(verification_result)

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["structure", "accuracy"])
    output_filename = f"./test/{sanitize_filename(MODEL_NAME)}_{TRAIN_SAMPLES}.csv"
    results_df.to_csv(output_filename, index=False)
    logging.info("Results saved to: %s", output_filename)
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()
