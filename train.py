from datasets import load_dataset
from colorama import Fore
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch


def format_chat_template(batch, tokenizer):
    system_prompt = (
        "You are an expert data science coding assistant. "
        "Given a problem description and setup code, write a concise Python solution. "
        "Output only the solution code with no explanation or markdown."
    )
    samples = []
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples
    }


if __name__ == "__main__":

    base_model = "Qwen/Qwen2.5-1.5B"  # Instruct variant — already has a chat template

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load all your generated + validated jsonl files
    dataset = load_dataset(
        "json",
        data_files={
            "train": [
                "validated/generated_samples_pandas_valid.jsonl",
                "validated/generated_samples_numpy_valid.jsonl",
                "validated/generated_samples_matplotlib_valid.jsonl",
                "validated/generated_samples_sklearn_valid.jsonl",
                "validated/generated_samples_scipy_valid.jsonl",
                "validated/generated_samples_pytorch_valid.jsonl",
                "validated/generated_samples_tensorflow_valid.jsonl",
            ]
        },
        split="train"
    )

    print(Fore.YELLOW + f"Total training samples: {len(dataset)}" + Fore.RESET)
    print(Fore.YELLOW + str(dataset[0]) + Fore.RESET)

    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        num_proc=1,
        batched=True,
        batch_size=10
    )
    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]["text"]) + Fore.RESET)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cuda:0",
        quantization_config=quant_config,
        trust_remote_code=True,
        cache_dir="./workspace",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir="Qwen2.5-1.5B-DS1000-2",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            save_steps=100,
            logging_steps=10,
            bf16=True,
            dataset_text_field="text",
        ),
        peft_config=peft_config,
    )

    print(Fore.YELLOW + "Starting training..." + Fore.RESET)
    trainer.train()

    trainer.save_model("complete_checkpoint3")
    trainer.model.save_pretrained("final_model3")
    tokenizer.save_pretrained("final_model3")
    print(Fore.GREEN + "Training complete. Model saved to final_model3/" + Fore.RESET)