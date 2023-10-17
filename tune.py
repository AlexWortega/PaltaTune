import os
os.environ["WANDB_PROJECT"]="test-task"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import tensor_parallel as tp
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset, block_size):
        self.examples = []
        self.block_size = block_size

        for chunk in tqdm.tqdm(zip(dataset['system_prompt'],dataset["question"], dataset["response"])):
            chunk = chunk[0] + chunk[1] + chunk[2] + eos_token
            if len(chunk) > 0:
                self.examples.append(chunk)
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        chunk = self.examples[idx]

        sample = tokenizer(
            chunk,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            )
        return sample.copy()
       

def run():
    config = load_config('config.json')

    model_name = config['model_name']
    tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length=config['model_max_length'])
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(config['dataset_name'],cache_dir=config['cache_dir'])
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config['torch_dtype']).to(config['cpu'])
    
    model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])# harcoooded
    
    train_samples, test_samples = dataset["train"], dataset["train"]

    train_samples = DialogueDataset(tokenizer=tokenizer,  dataset=train_samples, block_size=config['block_size'])
    test_samples = DialogueDataset(tokenizer=tokenizer,  dataset=test_samples, block_size=config['block_size'])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    peft_config = LoraConfig(
        r=config['r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias=config['bias'],
        task_type=config['task_type'],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        output_dir=config['output_model_dir'],
        overwrite_output_dir=False,
        save_total_limit=config['save_total_limit'],
        learning_rate=config['lr'],
        lr_scheduler_type=config['lr_scheduler_type'],
        num_train_epochs=config['epochs'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        warmup_steps=config['warmup_steps'],
        logging_first_step=config['logging_first_step'],
        fp16=config['fp16'],
        report_to=config['report_to'],
        optim=config['optim'],
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        dataloader_num_workers=config['dataloader_num_workers'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_samples,
        eval_dataset=test_samples,
    )
    train_result = trainer.train()

if __name__ == "__main__":
    run()