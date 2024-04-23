import argparse
import os
from os import PathLike
import jsonlines
# from model import DecoderBase, make_model
import transformers
import torch
import sys
from peft import PeftModel

def list_of_strings(arg):
    return arg.split(',')

def propose_prompt(raw_prompt):
    full_prompt = "### Instruction:\nPlease complete the code\n\n{}".format(raw_prompt).strip()
    return full_prompt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--adapter_path", default="" , type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    # parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=list_of_strings)
    # parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_path", type=list_of_strings, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--is_local_model", default=True, type=bool)
    # parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--id_list",)
    # parser.add_argument('--id_list', nargs='+', type=int, default=[],)
    args = parser.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"Debug_mode: {args.debug}")
    if 'codegen2' in  args.model_path:

     # Allow batched inference
    # tokenizer = AutoTokenizer.from_pretrained("/home/l_unjie/HF_models/CodeLlama-7b-hf",local_files_only=True)
                # model = transformers.AutoModelForCausalLM.from_pretrained("/home/l_unjie/HF_models/CodeLlama-7b-hf").to(device)
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model="/home/l_unjie/HF_models/CodeLlama-7b-hf",
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,local_files_only=args.is_local_model)
        if args.adapter_path != "":
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16,local_files_only=args.is_local_model)
            model = PeftModel.from_pretrained(model, args.adapter_path).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16,local_files_only=args.is_local_model).to(device)


    elif 'CodeLlama' in args.model_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"
        if args.adapter_path != "":
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(model, args.adapter_path).to(device)

        else:
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16,
            ).to(device)
    
    length_dataset = 1 if type(args.dataset_path) != list else len(args.dataset_path) 
    print(f"The dataset path are:{args.dataset_path}")
    print(f"The save path are :{args.save_path}")

    if type(args.save_path) == str: 
        if os.path.exists(args.save_path):
            print(f"{args.save_path} exists!!" )
            exit(0)
    elif type(args.save_path) == list:
        if os.path.exists(args.save_path[0]):
            print(f"{args.save_path} exists!!" )
            exit(0)
    for iter_num in range(length_dataset):
        reader_list = []
        with jsonlines.open(args.dataset_path[iter_num]) as reader:
            for obj in reader:
                reader_list.append(obj)

        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        # print(f"Debug_mode: {args.debug}")

                


        temp_count = 0

        req_iter = args.n_samples // args.batch
        print(f"req_iter is :{req_iter}")
        for line in reader_list:

            print(f"start to generate: {iter_num}/{length_dataset}:{temp_count}")
            result_list= []
            result_dic = {}
            result_dic = line
            
            

            inputs = tokenizer(propose_prompt(line['input']), return_tensors="pt", padding=True).to(device)
            input_ids = inputs["input_ids"].to(device)

            for r_i in range(req_iter): 
                print(f"start the batch:{r_i} for the temp_count:{temp_count}")
                # sequences = pipeline(
                #     propose_prompt(line['input']),
                #     do_sample=True,
                #     temperature=0.8,
                #     top_p=0.95,
                #     num_return_sequences=args.batch,
                #     eos_token_id=tokenizer.eos_token_id,
                #     pad_token_id=tokenizer.eos_token_id,
                #     max_length=512,
                # )
                generated_ids = model.generate(input_ids=input_ids, temperature = args.temperature,do_sample=True,num_return_sequences = args.batch,pad_token_id=tokenizer.eos_token_id, max_new_tokens=128)
            
                # gen_seqs = generated_ids.sequences[:, len(input_ids[0]) :]
                gen_seqs = generated_ids[:, :]
                gen_strs = tokenizer.batch_decode(
                    gen_seqs
                )
                outputs = []
                # removes eos tokens.
                for output in gen_strs:
                    outputs.append(output)
                

                
                for i in range(args.batch):
                    result_dic[f'output_{i+r_i*args.batch+1}'] = outputs[i]
            result_list.append(result_dic)
            temp_count+=1
            
            
            with jsonlines.open(args.save_path[iter_num],mode='a') as writer:
                for line in result_list:
                    jsonlines.Writer.write(writer,line)


if __name__ == "__main__":
    print(transformers.__version__)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    main()
