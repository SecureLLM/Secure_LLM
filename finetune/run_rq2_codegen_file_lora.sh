python3 ./fine-tune_script.py --base_model='Salesforce/codegen2-7B' --output_dir='./codegen2-7b_file_lora' --finetune_type='lora' --data_type='file' --wandb_run_name='codegen2-7b-file_lora' --batch_size=5 --micro_batch_size=5
