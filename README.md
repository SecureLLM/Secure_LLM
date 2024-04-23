# Secure_LLM

This is the replication


## Dependencies
Download the dependencies. Please using Python 3.10+

	> pip install -r requirements.txt

## Fine-tuning
In the folder 'finetune', run the command to fine-tune the specific model. An example is shown below.

	> python3 ./fine-tune_script.py \
    --base_model='Salesforce/codegen2-7B' \
    --output_dir='./codegen2-7b_block_lora' \
    --finetune_type='lora' --data_type='block' \
    --wandb_run_name='codegen2-7b-block_lora' \
    --batch_size=5 --micro_batch_size=5

## Code Generation  
In the folder 'code_generation', run the command to generate code with the adapter. An example is shown below.

	> python3 ./generate.py \
    --model_path='Salesforce/codegen2-7B' \
    --adapter_path="../codegen2-7b_block_lora" \
    --dataset_path=./model_inputs/CWE_inputs_rq2.jsonl \
    --save_path=./finetuned_block_codegen2-7b_lora.jsonl  \
    --temperature=0.8 --n_samples=30 --batch=2

## Evaluation
The scripts are put in the folder 'evaluation'.
Before evaluation, there is one post-processing for the generated code

Run the command to process the generated code. An example is shown below.

	> python3 ./post_process_gen_results.py \
    --raw_path='../../datasets/evaluate_dataset/gen_raw_results/finetuned_block_codegen2-7b.jsonl' \
    --c_cleaned_path='../../datasets/evaluate_dataset/codeql_results/lora/codegen/rq2/c/block-level' \
    --cpp_cleaned_path='../../datasets/evaluate_dataset/codeql_results/lora/codegen/rq2/c/block-level'


Then, run the command to evaluate using CodeQL. Please be sure that the CodeQL is installed and configured in your machine environment. An example is shown below.

	> python3 ./evaluate_cwe.py \
    --gen_code_path='../../datasets/evaluate_dataset/codeql_results/lora/codegen/rq2/c/block-level' \
    --makefile_path='../config_files/c_makefile/Makefile' \
    --db_path='../../datasets/evaluate_dataset/codeql_results/lora/codegen/rq2/c/block-level_db' \
    --results_path='../../datasets/evaluate_dataset/codeql_results/lora/codegen/rq2/c/block-level_results' \
    --statistic_path='../../datasets/evaluate_dataset/codeql_results/lora/statistics_results/codegen-block-c.csv' \
    --ql_rule_path='../config_files/ql.jsonl' \
    --language='c'
## Results
The folder 'results' saved the results. For the details of the raw data and the adapter weights, please follow the link **todo**




