"""LoRA trainer for fine-tuning language models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    logging.error(f"Required dependencies not installed: {e}")
    raise ImportError(
        "Please install required dependencies: torch, transformers, peft, datasets"
    )

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA trainer for fine-tuning language models on document data."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Initialize the LoRA trainer.

        Args:
            model_name: Name of the base model to fine-tune
            max_length: Maximum sequence length for tokenization
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA adaptation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules  # Will be auto-detected if None

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None

        self._load_model()

    def _load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Auto-detect target modules if not provided
        if self.target_modules is None:
            self.target_modules = self._find_target_modules()
            logger.info(f"Auto-detected target modules: {self.target_modules}")

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def _prepare_dataset(self, documents: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare the dataset for training.

        Args:
            documents: List of parsed documents

        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Preparing dataset for training")

        # Extract text content from documents
        texts = []
        for doc in documents:
            content = doc.get("content", "")
            if content.strip():
                texts.append(content)

        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return tokenized_dataset

    def train(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 5e-4,
        output_dir: str = "./lora_training_output",
    ):
        """
        Train the LoRA model on the documents.

        Args:
            documents: List of parsed documents
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            output_dir: Directory to save training outputs
        """
        logger.info("Starting LoRA training")

        # Prepare dataset
        dataset = self._prepare_dataset(documents)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
            dataloader_drop_last=True,
            report_to=None,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        logger.info("Training completed")

    def save_adapter(self, output_path: str) -> str:
        """
        Save the LoRA adapter to a file.

        Args:
            output_path: Path to save the adapter

        Returns:
            Path to the saved adapter file
        """
        logger.info(f"Saving LoRA adapter to: {output_path}")

        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the adapter
        if output_path.suffix == ".json":
            # Save as JSON for compatibility
            adapter_dir = output_path.parent / f"{output_path.stem}_adapter"
            self.peft_model.save_pretrained(adapter_dir)

            # Create a JSON file with metadata
            metadata = {
                "adapter_path": str(adapter_dir),
                "base_model": self.model_name,
                "lora_config": {
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                    "target_modules": self.target_modules,
                },
                "max_length": self.max_length,
            }

            with open(output_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return str(output_path)
        else:
            # Save directly as adapter directory
            self.peft_model.save_pretrained(output_path)
            return str(output_path)

    def load_adapter(self, adapter_path: str):
        """
        Load a LoRA adapter from a file.

        Args:
            adapter_path: Path to the adapter file or directory
        """
        logger.info(f"Loading LoRA adapter from: {adapter_path}")

        adapter_path = Path(adapter_path)

        if adapter_path.suffix == ".json":
            # Load from JSON metadata
            with open(adapter_path, "r") as f:
                metadata = json.load(f)

            actual_adapter_path = metadata["adapter_path"]
            self.peft_model.load_adapter(actual_adapter_path)
        else:
            # Load directly from adapter directory
            self.peft_model.load_adapter(adapter_path)

    def _find_target_modules(self):
        """
        Auto-detect suitable target modules for LoRA based on the model architecture.

        Returns:
            List of target module names suitable for LoRA adaptation
        """
        # Common target modules for different model types
        target_modules_by_model = {
            # GPT-2 and DialoGPT models
            "gpt2": ["c_attn"],
            "dialogpt": ["c_attn"],
            "microsoft/dialogpt": ["c_attn"],
            # GPT-J models
            "gptj": ["q_proj", "v_proj"],
            "gpt-j": ["q_proj", "v_proj"],
            # LLaMA models
            "llama": ["q_proj", "v_proj"],
            # Mistral models
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral-7b": ["q_proj", "k_proj", "v_proj", "o_proj"],
            # T5 models
            "t5": ["q", "v"],
            # BERT models
            "bert": ["query", "value"],
            # Default fallback
            "default": ["c_attn"],
        }

        model_name_lower = self.model_name.lower()

        # Try to match model type
        for model_type, modules in target_modules_by_model.items():
            if model_type in model_name_lower:
                logger.info(f"Detected model type: {model_type}")
                return modules

        # If no match, try to introspect the model
        try:
            target_modules = []
            for name, module in self.model.named_modules():
                # Look for common attention module patterns
                if any(pattern in name for pattern in ["attn", "attention"]):
                    # Check if it's a linear layer
                    if hasattr(module, "weight") and len(module.weight.shape) == 2:
                        module_name = name.split(".")[-1]
                        if module_name not in target_modules:
                            target_modules.append(module_name)

            if target_modules:
                logger.info(
                    f"Found attention modules by introspection: {target_modules}"
                )
                return target_modules[:2]  # Return first 2 to avoid too many modules

        except Exception as e:
            logger.warning(f"Could not introspect model modules: {e}")

        # Final fallback - use the most common pattern
        logger.info("Using default target modules")
        return ["c_attn"]  # Most common for GPT-style models
