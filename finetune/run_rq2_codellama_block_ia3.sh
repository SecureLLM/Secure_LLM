python3 ./fine-tune_script.py --base_model='codellama/CodeLlama-7b-hf' --output_dir='./codellama-7b_block_ia3' --finetune_type='ia3' --data_type='block' --wandb_run_name='codellama-7b-block_ia3' --batch_size=5 --micro_batch_size=5