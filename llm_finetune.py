import argparse
import gc
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(data_path: Path, num_samples: int = None, seed: int = 315):
    """
    Load and shuffle the dataset from a JSON file.
    """
    logging.info("Loading dataset from %s", data_path)
    dataset_file = data_path / "train.json"
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")
    total_samples = len(dataset)
    logging.info("Dataset loaded with %d total samples", total_samples)
    
    if num_samples is None or num_samples <= 0:
        num_samples = total_samples
        logging.info("Using all samples: %d", num_samples)
    else:
        logging.info("Selecting %d samples from dataset", num_samples)
    
    shuffled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    logging.info("Dataset shuffled and subsetted")
    return shuffled_dataset


def get_torch_dtype():
    """
    Determine the appropriate torch data type based on GPU capability.
    """
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        logging.info("CUDA is available. Device capability: %s", device_capability)
        dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16
    else:
        logging.info("CUDA is not available. Using float32")
        dtype = torch.float32  # Default to float32 if CUDA is not available
    logging.info("Selected torch data type: %s", dtype)
    return dtype


def load_model_and_tokenizer(model_id: str, torch_dtype: torch.dtype):
    """
    Load the model and tokenizer with quantization configuration.
    """
    logging.info("Loading model and tokenizer for model ID: %s", model_id)
    model_kwargs = {
        "attn_implementation": "eager",  # Use "flash_attention_2" for newer GPUs if desired
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }

    # Enable 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    logging.info("Initializing model with quantization configuration")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def create_trainer(model, tokenizer, dataset, torch_dtype: torch.dtype, num_samples: int, output_dir: Path):
    """
    Create and return a SFTTrainer for model fine-tuning.
    """
    logging.info("Creating trainer for fine-tuning")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir / f"model_{num_samples}"),
        max_seq_length=512,
        packing=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True if torch_dtype == torch.float16 else False,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    logging.info("Trainer configuration complete. Output directory: %s", training_args.output_dir)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )
    logging.info("Trainer created successfully")
    return trainer


def clear_memory():
    """
    Clear memory by collecting garbage and emptying CUDA cache.
    """
    logging.info("Clearing memory")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleared")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="CLI for fine-tuning a model using SFTTrainer"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to use for training (default: None, use all samples)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the data directory (default: ./data)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model identifier (required)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the training output (default: ./output)"
    )
    args = parser.parse_args()
    logging.info("Arguments parsed: %s", args)
    return args


def main():
    logging.info("Starting main process")
    args = parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    n_samples = args.n_samples

    # Load and prepare the dataset
    logging.info("Loading and preparing the dataset")
    dataset = load_data(data_path, n_samples)

    # Determine the appropriate torch data type based on GPU capability
    logging.info("Determining torch data type")
    torch_dtype = get_torch_dtype()

    # Load the model and tokenizer with quantization settings
    logging.info("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(args.model_id, torch_dtype)

    # Create the trainer for fine-tuning the model
    logging.info("Creating trainer")
    trainer = create_trainer(model, tokenizer, dataset, torch_dtype, len(dataset), output_dir)

    # Start training
    logging.info("Starting training")
    trainer.train()
    logging.info("Training completed")

    # Clean up and free memory
    logging.info("Cleaning up memory")
    del model, trainer
    clear_memory()
    logging.info("Main process finished")


if __name__ == "__main__":
    main()
