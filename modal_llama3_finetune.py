import os
import modal


def _load_env():
    # Loads .env locally (for HF_TOKEN / WANDB_API_KEY), without requiring python-dotenv.
    try:
        from dotenv import load_dotenv  # optional dependency
        load_dotenv()
        return
    except Exception:
        pass

    env_path = ".env"
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


_load_env()

# Create Modal secrets from your local environment (.env values get loaded above).
modal_secrets = [modal.Secret.from_local_environ(["HF_TOKEN"])]
if os.getenv("WANDB_API_KEY"):
    modal_secrets.append(modal.Secret.from_local_environ(["WANDB_API_KEY"]))

# Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")  # needed if you ever use hf_login(add_to_git_credential=True)
    .uv_pip_install(
        "torch<3",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "bitsandbytes==0.46.0",
        "peft==0.14.0",
        "trl==0.14.0",
        "wandb",
        "huggingface_hub",
        "tqdm",
        "matplotlib",
    )
)

app = modal.App("llama3-pricer-finetune")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    secrets=modal_secrets,
)

def train():
    import os
    import json
    from datetime import datetime
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from huggingface_hub import login as hf_login, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from peft import LoraConfig

    # ======================
    # Resume settings
    # ======================
    RESUME_REPO_ID = "costadev00/pricer-2025-12-01_15.09.36"
    RESUME_REVISION = "main"  # <--- PULL LATEST (use a commit sha here if you want to pin)

    # Optional wandb
    LOG_TO_WANDB = True
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        LOG_TO_WANDB = False

    if LOG_TO_WANDB:
        import wandb
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_PROJECT"] = "pricer"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_WATCH"] = "gradients"

    # ===== Constants =====
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    PROJECT_NAME = "pricer"
    HF_USER = "costadev00"
    DATASET_NAME = f"{HF_USER}/pricer-data"
    MAX_SEQUENCE_LENGTH = 1024

    run_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    # QLoRA hyperparams
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1
    QUANT_4_BIT = True
    
    # Training hyperparams
    EPOCHS = 1
    BATCH_SIZE = 128
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    SAVE_STEPS = 2000

    # ===== Hugging Face login =====
    hf_login(os.environ["HF_TOKEN"], add_to_git_credential=True)

    if LOG_TO_WANDB:
        import wandb
        wandb.login(key=wandb_api_key)
        wandb.init(project=PROJECT_NAME, name=f"resume-{run_name}")

    # ======================
    # Download checkpoints from the Hub and pick the latest one
    # ======================
    resume_root = Path("/root/resume_from_hub")
    resume_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=RESUME_REPO_ID,
        revision=RESUME_REVISION,
        local_dir=str(resume_root),
        local_dir_use_symlinks=False,
        token=os.environ["HF_TOKEN"],
        allow_patterns=[
            # checkpoints
            "last-checkpoint/**",
            "checkpoint-*/**",

            # trainer metadata
            "**/trainer_state.json",
            "**/training_args.bin",

            # peft adapter
            "**/adapter_config.json",
            "**/adapter_model.safetensors",
            "**/adapter_model.bin",

            # resume state (very important for true continuation)
            "**/optimizer.pt",
            "**/scheduler.pt",
            "**/rng_state.pth",
            "**/scaler.pt",
        ],
    )

    resume_ckpt = None
    last_ckpt = resume_root / "last-checkpoint"
    if last_ckpt.exists():
        resume_ckpt = str(last_ckpt)
    else:
        ckpts = [p for p in resume_root.glob("checkpoint-*") if p.is_dir()]
        if ckpts:
            ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
            resume_ckpt = str(ckpts[-1])

    if not resume_ckpt:
        raise RuntimeError("No checkpoint found. Expected last-checkpoint/ or checkpoint-*/ folders.")

    state_path = Path(resume_ckpt) / "trainer_state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"Resuming from {resume_ckpt} (global_step={state.get('global_step')}, max_steps={state.get('max_steps')})")
    else:
        print(f"Resuming from {resume_ckpt} (trainer_state.json not found)")

    # Avoid instantly stopping if your new config would compute fewer total steps
    resume_max_steps = state.get("max_steps")
    MAX_STEPS = resume_max_steps if isinstance(resume_max_steps, int) and resume_max_steps > 0 else None

    # ===== Load dataset =====
    dataset = load_dataset(DATASET_NAME)
    train_ds = dataset["train"]

    # ===== Quantization config =====
    if QUANT_4_BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        use_bf16 = True
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        use_bf16 = True

    # ===== Tokenizer and base model =====
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

    # ===== Data collator =====
    collator = DataCollatorForCompletionOnlyLM(
        response_template="Price is $",
        tokenizer=tokenizer,
    )

    # ===== LoRA + SFT config =====
    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    cfg_kwargs = dict(
        output_dir=str(resume_root),
        max_seq_length=MAX_SEQUENCE_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIMIZER,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=use_bf16,
        report_to="wandb" if LOG_TO_WANDB else "none",
        push_to_hub=True,
        hub_model_id=RESUME_REPO_ID,      # keep pushing to same repo
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )

    if MAX_STEPS is not None:
        train_parameters = SFTConfig(max_steps=MAX_STEPS, **cfg_kwargs)
    else:
        train_parameters = SFTConfig(num_train_epochs=EPOCHS, **cfg_kwargs)

    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train_ds,
        peft_config=lora_parameters,
        args=train_parameters,
        data_collator=collator,
    )

    fine_tuning.train(resume_from_checkpoint=resume_ckpt)

    fine_tuning.push_to_hub()
    print(f"Resumed and pushed checkpoints to: {RESUME_REPO_ID}")

    if LOG_TO_WANDB:
        import wandb
        wandb.finish()