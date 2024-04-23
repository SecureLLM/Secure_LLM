
python3 ./generate.py \
--model_path='Salesforce/codegen2-7B' \
--adapter_path="../codegen2-7b_block_ia3" \
--dataset_path=./model_inputs/CWE_inputs_rq2.jsonl \
--save_path=./finetuned_block_codegen2-7b_ia3.jsonl  --temperature=0.8 --n_samples=30 --batch=2
