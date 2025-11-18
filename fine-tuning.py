#!/usr/bin/env python
"""LoRA fine-tuning entry point for Llama-3 8B."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def _comma_separated_list(raw: str) -> Iterable[str]:
  return [token.strip() for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Fine-tune Llama-3 (8B) with LoRA/PEFT on a Hugging Face dataset."
  )
  parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "meta-llama/Llama-3-8b"))
  parser.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME", "costadev00/pricer-data"))
  parser.add_argument("--train-split", default="train")
  parser.add_argument("--dataset-text-field", default="text")
  parser.add_argument("--response-template", default="Price is $")
  parser.add_argument("--max-seq-length", type=int, default=182)
  parser.add_argument("--per-device-train-batch-size", type=int, default=4)
  parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--epochs", type=float, default=1.0)
  parser.add_argument("--warmup-ratio", type=float, default=0.03)
  parser.add_argument("--lr-scheduler-type", default="cosine")
  parser.add_argument("--save-steps", type=int, default=200)
  parser.add_argument("--logging-steps", type=int, default=25)
  parser.add_argument("--save-total-limit", type=int, default=5)
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--seed", type=int, default=42)

  parser.add_argument("--lora-r", type=int, default=32)
  parser.add_argument("--lora-alpha", type=int, default=64)
  parser.add_argument("--lora-dropout", type=float, default=0.1)
  parser.add_argument(
    "--target-modules",
    default="q_proj,k_proj,v_proj,o_proj",
    help="Comma separated list of attention projection modules to adapt",
  )

  parser.add_argument("--load-in-4bit", action="store_true", default=True)
  parser.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_TOKEN"))
  parser.add_argument("--hf-user", default=os.environ.get("HF_USERNAME", "costadev00"))
  parser.add_argument("--push-to-hub", action="store_true", default=False)
  parser.add_argument("--hub-model-id", default=None)
  parser.add_argument("--hub-private-repo", action="store_true", default=True)

  parser.add_argument("--log-to-wandb", action="store_true", default=False)
  parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "pricer"))
  parser.add_argument("--max-train-samples", type=int, default=None)

  parser.add_argument("--run-name", default=None)

  return parser.parse_args()


def maybe_login_hf(token: Optional[str], enable_push: bool) -> None:
  if enable_push:
    if not token:
      raise ValueError("HUGGINGFACE_TOKEN is required when --push-to-hub is enabled.")
    login(token=token, add_to_git_credential=True)
  elif token:
    login(token=token, add_to_git_credential=False)


def maybe_init_wandb(enable: bool, project: str, run_name: str):
  if not enable:
    return None
  api_key = os.environ.get("WANDB_API_KEY")
  if not api_key:
    raise ValueError("WANDB_API_KEY environment variable is required when --log-to-wandb is set.")
  import wandb  # pylint: disable=import-outside-toplevel

  wandb.login(key=api_key)
  return wandb.init(project=project, name=run_name, reinit=True)


def build_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig:
  if load_in_4bit:
    return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type="nf4",
    )
  return BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
  )


def main() -> None:
  args = parse_args()
  run_name = args.run_name or f"{args.hf_user}-{datetime.utcnow():%Y-%m-%d_%H.%M.%S}"
  output_dir = args.output_dir or os.path.join("outputs", run_name)

  set_seed(args.seed)
  maybe_login_hf(args.hf_token, args.push_to_hub)
  wandb_run = maybe_init_wandb(args.log_to_wandb, args.wandb_project, run_name)

  dataset = load_dataset(args.dataset_name)
  train_dataset = dataset[args.train_split]
  if args.max_train_samples:
    train_dataset = train_dataset.select(range(args.max_train_samples))

  tokenizer = AutoTokenizer.from_pretrained(args.base_model)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  quant_config = build_quant_config(args.load_in_4bit)
  base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    device_map="auto",
    quantization_config=quant_config,
  )
  base_model.generation_config.pad_token_id = tokenizer.pad_token_id

  collator = DataCollatorForCompletionOnlyLM(
    response_template=args.response_template,
    tokenizer=tokenizer,
  )

  lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=list(_comma_separated_list(args.target_modules)),
  )

  hub_kwargs = {}
  if args.push_to_hub:
    hub_kwargs = {
      "hub_strategy": "every_save",
      "hub_model_id": args.hub_model_id or f"{args.hf_user}/{run_name}",
      "hub_private_repo": args.hub_private_repo,
    }

  training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    optim="paged_adamw_32bit",
    bf16=torch.cuda.is_available(),
    max_seq_length=args.max_seq_length,
    dataset_text_field=args.dataset_text_field,
    report_to="wandb" if args.log_to_wandb else None,
    run_name=run_name,
    push_to_hub=args.push_to_hub,
    **hub_kwargs,
  )

  trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    args=training_args,
    data_collator=collator,
  )

  trainer.train()

  if args.push_to_hub:
    model_id = hub_kwargs.get("hub_model_id")
    trainer.model.push_to_hub(model_id, private=args.hub_private_repo)

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == "__main__":
  main()