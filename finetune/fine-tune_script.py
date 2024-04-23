import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    IA3Config,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")

        ### modified by Junjie
        # file_name = r"./alpaca.json"
        file_name = r"/content/drive/My Drive/Vul_Gen/test/alpaca.json"
        print(osp.exists(file_name))
        print(os.getcwd())
        ### finished
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

target_modeules = {
    "codegen2-7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "CodeLlama-7b-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["fc_in", "fc_out"]
    }
}


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "",
    output_dir: str = "",

    finetune_type: str = 'lora',
    data_type: str = 'line',
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "all",  # options: false | gradients | all
    wandb_log_model: str = "checkpoint",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    if 'codegen2' in  base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True)
        ### add layers to learn
        if finetune_type == 'lora': 
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules = target_modeules['codegen2-7B']['target_modules_lora'],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif finetune_type == 'ia3':
            config = IA3Config(task_type="CAUSAL_LM",
                        target_modules=target_modeules['codegen2-7B']['target_modules_ia3'],
                        feedforward_modules= target_modeules['codegen2-7B']["ff_modules"])
    elif 'CodeLlama' in base_model:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        ### add layers to learn
        if finetune_type == 'lora': 
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules = target_modeules['CodeLlama-7b-hf']['target_modules_lora'],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif finetune_type == 'ia3':
            config = IA3Config(task_type="CAUSAL_LM",
                        target_modules=target_modeules['CodeLlama-7b-hf']['target_modules_ia3'],
                        feedforward_modules= target_modeules['CodeLlama-7b-hf']["ff_modules"])
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_function(examples):
        full_prompt = "### Instruction:\nPlease complete the code\n\n{}".format(examples["text"])
        return tokenize(full_prompt)
    
    model = prepare_model_for_int8_training(model)

    
    model = get_peft_model(model, config)

    ### load data
    train_data_path = os.path.join(data_path, 'train-{}.json'.format(data_type))
    val_data_path = os.path.join(data_path,"validation-{}.json".format(data_type))
    train_data = load_dataset("json", data_files=train_data_path)
    val_data = load_dataset("json", data_files=val_data_path)

    train_data = train_data['train'].shuffle().map(tokenize_function)
    val_data = val_data['train'].shuffle().map(tokenize_function)
    val_set_size = len(val_data)
    print(train_data)
    print(f"len_train_data:{len(train_data)}")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    if data_path.endswith("cross/train.json"):
        save_steps=(len(train_data)/5)*10
        logging_steps=600
    else:
        save_steps=(len(train_data)/5)*5
        logging_steps=10
    print(f"save_steps:{save_steps}")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1000 if val_set_size > 0 else None,
            save_steps=1000,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

if __name__ == "__main__":
    fire.Fire(train)