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
**todo  **

The folder 'results' saved the results. For the details of the raw data, please follow the link **todo**
