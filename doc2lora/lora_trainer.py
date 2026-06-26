"""LoRA trainer for fine-tuning language models."""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
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

# optional 4-bit QLoRA path (cuda only)
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    import bitsandbytes as _bnb  # noqa: F401

    _HAS_BITSANDBYTES = True
except ImportError:
    _HAS_BITSANDBYTES = False

# Cloudflare Workers AI accepts BYO LoRA adapters up to rank 32 (300MB safetensors)
CLOUDFLARE_MAX_RANK = 32

logger = logging.getLogger(__name__)


def get_device():
    """
    Detect and return the best available device for training.

    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 GPU detected: {gpu_name} ({total_memory:.1f} GB)")
        logger.info(f"📊 Available GPUs: {gpu_count}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
        return device
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU (GPU not available)")
        return device


class LoRATrainer:
    """LoRA trainer for fine-tuning language models on document data."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        max_length: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: Optional[str] = None,
        gradient_checkpointing: bool = True,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the LoRA trainer.

        Args:
            model_name: Name of the base model to fine-tune
            max_length: Maximum sequence length for tokenization
            lora_r: LoRA rank parameter (Cloudflare Workers AI supports up to 32)
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA adaptation
            device: Device to use for training ('cuda', 'mps', 'cpu', or None for auto-detection)
            gradient_checkpointing: Trade compute for memory (helps on low-RAM machines)
            load_in_4bit: Use 4-bit QLoRA (requires bitsandbytes + CUDA; ignored otherwise)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules  # Will be auto-detected if None
        self.gradient_checkpointing = gradient_checkpointing
        self.load_in_4bit = load_in_4bit
        self.use_bf16 = False  # resolved in _load_model

        # Warn about LoRA rank limit for Cloudflare Workers AI
        if lora_r > CLOUDFLARE_MAX_RANK:
            logger.warning(
                f"⚠️  LoRA rank {lora_r} exceeds Cloudflare Workers AI limit of "
                f"{CLOUDFLARE_MAX_RANK}. Consider --lora-r {CLOUDFLARE_MAX_RANK} or "
                f"lower for compatibility."
            )
        elif lora_r > 8:
            logger.info(
                f"ℹ️  LoRA rank {lora_r} (>8) needs Cloudflare's higher-rank runtime "
                f"and keeps the adapter under the 300MB file limit."
            )

        # Detect and set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
            logger.info(f"🔧 Using specified device: {self.device}")

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None

        self._load_model()

    def _load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Get HuggingFace token from environment
        import os

        hf_token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_TOKEN")
        auth_kwargs = {"token": hf_token} if hf_token else {}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **auth_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_cuda = self.device.type == "cuda"
        is_gpu_available = self.device.type in ["cuda", "mps"]

        # resolve 4-bit QLoRA availability (cuda + bitsandbytes only)
        if self.load_in_4bit and not (
            is_cuda and _HAS_BITSANDBYTES and BitsAndBytesConfig is not None
        ):
            logger.warning(
                "⚠️  4-bit QLoRA requires CUDA + bitsandbytes; falling back to "
                "standard LoRA"
            )
            self.load_in_4bit = False

        # prefer bf16 on capable CUDA hardware, else fp16 on gpu, else fp32
        self.use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
        if is_gpu_available:
            compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            logger.info(f"💾 Using {compute_dtype} precision for GPU training")
        else:
            compute_dtype = torch.float32
            logger.info("💾 Using float32 precision for CPU training")

        load_kwargs = {"low_cpu_mem_usage": True, **auth_kwargs}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["device_map"] = {"": 0}
            logger.info("🧮 Using 4-bit QLoRA (nf4, double quantization)")
        else:
            load_kwargs["torch_dtype"] = compute_dtype
            load_kwargs["device_map"] = None

        def _load(**kwargs):
            return AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        try:
            self.model = _load(**load_kwargs)
            if not self.load_in_4bit:
                self.model = self.model.to(self.device)
            logger.info(f"📱 Model loaded successfully on {self.device}")
        except Exception as e:  # includes torch.cuda.OutOfMemoryError
            if self.device.type == "cpu":
                logger.error(f"❌ Failed to load model on CPU: {e}")
                raise
            logger.warning(
                f"⚠️  Could not load on {self.device}, falling back to CPU: {e}"
            )
            self.device = torch.device("cpu")
            self.load_in_4bit = False
            self.use_bf16 = False
            self.model = _load(
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                **auth_kwargs,
            )
            self.model = self.model.to(self.device)
            logger.info("💻 Successfully loaded model on CPU")

        # prepare for k-bit training (gradient checkpointing handled via Trainer)
        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=False
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

        # gradient checkpointing needs input grads enabled on the wrapped model
        if self.gradient_checkpointing:
            try:
                self.peft_model.enable_input_require_grads()
            except Exception as e:
                logger.warning(f"Could not enable input require grads: {e}")

        self.peft_model.print_trainable_parameters()

        logger.info(f"✅ Model successfully loaded on {self.device}")

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
        num_epochs: Optional[int] = 3,
        max_steps: Optional[int] = None,
        learning_rate: float = 5e-4,
        gradient_accumulation_steps: int = 1,
        output_dir: str = "./lora_training_output",
    ):
        """
        Train the LoRA model on the documents.

        Args:
            documents: List of parsed documents
            batch_size: Training batch size
            num_epochs: Number of training epochs (ignored if max_steps is set)
            max_steps: Maximum number of training steps (overrides num_epochs if set)
            learning_rate: Learning rate for training
            gradient_accumulation_steps: Accumulate grads to emulate a larger batch
                on low-memory machines (effective batch = batch_size * this)
            output_dir: Directory to save training outputs
        """
        logger.info("Starting LoRA training")

        # Prepare dataset
        dataset = self._prepare_dataset(documents)

        # Prepare training arguments
        is_cuda = self.device.type == "cuda"
        is_gpu_available = self.device.type in ["cuda", "mps"]

        # scale warmup to the actual run length (fixed 100 over-warms tiny corpora)
        effective_batch = max(1, batch_size * gradient_accumulation_steps)
        steps_per_epoch = max(1, len(dataset) // effective_batch)
        total_steps = max_steps or steps_per_epoch * (num_epochs or 3)
        warmup_steps = min(100, max(0, total_steps // 10))

        training_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "bf16": self.use_bf16,
            "fp16": is_cuda and not self.use_bf16,
            "logging_steps": 10,
            "save_strategy": "epoch" if max_steps is None else "steps",
            "remove_unused_columns": False,
            "dataloader_drop_last": True,
            "report_to": [],
        }

        # gradient checkpointing for 4-bit is handled in _load_model; let the
        # Trainer own it for the standard path
        if self.gradient_checkpointing and not self.load_in_4bit:
            training_kwargs["gradient_checkpointing"] = True
            training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        # Log training configuration
        logger.info("🏋️  Training configuration:")
        logger.info(f"   Device: {self.device}")
        logger.info(
            f"   Batch size: {batch_size} (accum x{gradient_accumulation_steps})"
        )
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(
            f"   Precision: {'bf16' if self.use_bf16 else ('fp16' if is_cuda else 'fp32')}"
        )
        logger.info(f"   Gradient checkpointing: {self.gradient_checkpointing}")
        logger.info(f"   QLoRA (4-bit): {self.load_in_4bit}")
        if not is_gpu_available:
            logger.info(
                "   ⚡ Tip: Training on CPU is slower. A GPU (or 4-bit QLoRA on "
                "CUDA) speeds this up substantially."
            )

        # Set either epochs or max_steps
        if max_steps is not None:
            training_kwargs["max_steps"] = max_steps
            training_kwargs["save_steps"] = max(
                1, max_steps // 10
            )  # Save every 10% of training
        else:
            training_kwargs["num_train_epochs"] = num_epochs or 3

        training_args = TrainingArguments(**training_kwargs)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Initialize trainer; newer transformers renamed `tokenizer` ->
        # `processing_class`, so pass whichever the installed version accepts
        trainer_kwargs = {
            "model": self.peft_model,
            "args": training_args,
            "train_dataset": dataset,
            "data_collator": data_collator,
        }
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = Trainer(**trainer_kwargs)

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

            # Ensure Cloudflare Workers AI compatibility by updating adapter_config.json
            config_path = adapter_dir / "adapter_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Add model_type required by Cloudflare Workers AI
                config["model_type"] = self._get_cloudflare_model_type()

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                logger.info(
                    f"✅ Updated adapter_config.json with model_type: {config['model_type']}"
                )

            # Create a JSON file with metadata
            metadata = {
                "adapter_path": str(adapter_dir),
                "base_model": self.model_name,
                "model_type": self._get_cloudflare_model_type(),
                "lora_config": {
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                    "target_modules": self.target_modules,
                },
                "max_length": self.max_length,
                "cloudflare_compatible": True,
            }

            with open(output_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"🚀 LoRA adapter saved for Cloudflare Workers AI compatibility"
            )
            logger.info(f"📁 Adapter directory: {adapter_dir}")
            logger.info(
                f"📄 Required files: adapter_config.json, adapter_model.safetensors"
            )

            return str(output_path)
        else:
            # Save directly as adapter directory
            self.peft_model.save_pretrained(output_path)

            # Ensure Cloudflare Workers AI compatibility by updating adapter_config.json
            config_path = Path(output_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Add model_type required by Cloudflare Workers AI
                config["model_type"] = self._get_cloudflare_model_type()

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                logger.info(
                    f"✅ Updated adapter_config.json with model_type: {config['model_type']}"
                )
                logger.info(
                    f"🚀 LoRA adapter saved for Cloudflare Workers AI compatibility"
                )

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

    def _get_cloudflare_model_type(self) -> str:
        """
        Get the model_type required by Cloudflare Workers AI.

        Returns:
            Model type string ("mistral", "gemma", "llama", or "qwen")
        """
        model_name_lower = self.model_name.lower()

        if "mistral" in model_name_lower:
            return "mistral"
        elif "gemma" in model_name_lower:
            return "gemma"
        elif "llama" in model_name_lower:
            return "llama"
        elif "qwen" in model_name_lower or "qwq" in model_name_lower:
            # QwQ / Qwen base models (e.g. @cf/qwen/qwq-32b)
            return "qwen"
        else:
            # Default to mistral as it's the most common
            logger.warning(
                f"Unknown model type for {self.model_name}, defaulting to 'mistral'"
            )
            return "mistral"

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
            # Qwen / QwQ models (Qwen2 attention projections)
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "qwq": ["q_proj", "k_proj", "v_proj", "o_proj"],
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
