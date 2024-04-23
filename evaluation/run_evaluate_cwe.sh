python3 ./evaluate_cwe.py \
--gen_code_path='../../datasets/evaluate_dataset/codeql_results/codegen/rq1/c/pre-trained' \
--makefile_path='../config_files/c_makefile/Makefile' \
--db_path='../../datasets/evaluate_dataset/codeql_results/codegen/rq1/c/pre-trained_db' \
--results_path='../../datasets/evaluate_dataset/codeql_results/codegen/rq1/c/pre-trained_results' \
--statistic_path='../../datasets/evaluate_dataset/codeql_results/statistics_results/codegen-pretrained-c.csv' \ 
--ql_rule_path='../config_files/ql.jsonl' \
--language='c'

