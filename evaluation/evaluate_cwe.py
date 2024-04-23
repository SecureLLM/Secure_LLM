import os 
import pandas
import shutil 
import json
import jsonlines
import pandas as pd
import csv
from ast import literal_eval
import fire

##1. split files
def split_files(path,target_path):
    file_list = os.listdir(path)
    for file in file_list:
        tokens = file.split('-')[:2]
        sce = '-'.join(tokens)
        print(sce)
        ##create a folder
        if not os.path.exists(os.path.join(target_path,sce)):
            os.mkdir(os.path.join(target_path,sce))
        
        ###move file into folder
        shutil.copy2(os.path.join(path,file),os.path.join(target_path,os.path.join(sce,file)))

# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/cleaned"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/split"
# split_files(path,target_path)

# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/cleaned"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/split"
# split_files(path,target_path)

# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/cleaned"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/split"
# split_files(path,target_path)

# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/cleaned"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/split"
# split_files(path,target_path)

# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/cleaned"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/split"
# split_files(path,target_path)
##2. move make files
def copy_makefile(path,target_path):
    folder_list = os.listdir(target_path)
    for folder in folder_list:
        shutil.copy2(path,os.path.join(target_path,folder))


makefile_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/Makefile"

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/split"
# copy_makefile(makefile_path,target_path)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/split"
# copy_makefile(makefile_path,target_path)
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/split"
# copy_makefile(makefile_path,target_path)
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/split"
# copy_makefile(makefile_path,target_path)
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/split"
# copy_makefile(makefile_path,target_path)

##3. database create
def create_database(path,target_path,extra_name):
    folder_list = os.listdir(path)
    # print(os.system('/home/junjie/Desktop/vul_project/tools/codeql-linux64/codeql/codeql'))
    for folder in folder_list:

        command = f'/home/junjie/Desktop/vul_project/tools/codeql-linux64/codeql/codeql database create --language=cpp --overwrite --command="make -B" --source-root {os.path.join(path,folder)} {os.path.join(target_path,extra_name + folder)}'
        result = os.system(command)
        print(folder)
        print(result)
        print()


# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/split"
# extra_name = 'fine-tuned-'
# create_database(path,target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/split"
# extra_name = 'fine-tuned-'
# create_database(path,target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/split"
# extra_name = 'fine-tuned-'
# create_database(path,target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/split"
# extra_name = 'fine-tuned-'
# create_database(path,target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/split"
# extra_name = 'pre-trained-'
# create_database(path,target_path,extra_name)

# target_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/processed_dataset/split_m6-db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/processed_dataset/split_m6"
# extra_name = 'fine-tuned-'
# create_database(path,target_path,extra_name)



##4. find the correspond ql file
def create_json_ql_file(path,target_path):
    save_list = []

    
    for folder_cwe in os.listdir(path):
        for folder_sce in os.listdir(os.path.join(path,folder_cwe)):
            if folder_sce == "qlpack.yml" or folder_sce == ".cache":
                continue
            with open(os.path.join(path,os.path.join(folder_cwe,os.path.join(folder_sce,"mark_setup.json"))),'r') as f:
                data = json.load(f)
                if data["language"] == 'c' :
                    exp_id =data["exp_id"]
                    if 'check_ql' in data.keys():  
                        ## having ql 
                        # two cases 
                        # 1. pre-defined 
                        # 2. author-defined
                        raw_path = data["check_ql"]
                        
                        if raw_path.startswith("$CODEQL_HOME/"): ## from pre-defined
                            ql_path = raw_path.replace("$CODEQL_HOME/codeql-repo","/home/junjie/Desktop/vul_project/tools/codeql")
                            ql_type = "pre-defined"
                        elif raw_path.startswith("./experiments_dow"): ##from authors-defined
                            ql_name = raw_path.split('/')[-1]
                            ql_path = os.path.join(path,os.path.join(folder_cwe,os.path.join(folder_sce,ql_name)))
                            ql_type = "author-defined"
                        else:
                            print(f"an issue of ql path:{raw_path}")
                            return 0
                    else:
                        ## manually
                        ql_path = "manually" 
                        ql_type = "manually" 
                    ## construct a file to save ql
                    #1. cwe_sce_id 2. ql_path
                    sce_name = folder_cwe + '_' + str(exp_id)
                    save_dic = {"scenario_id":sce_name,"check_ql":ql_path,"ql_type":ql_type}
                    save_list.append(save_dic)
                    print(save_dic)
    with jsonlines.open(target_path,mode='w') as f:
        for save_line in save_list:
            f.write(save_line)

path =r"/home/junjie/Desktop/vul_project/copilot-cwe-scenarios-dataset/experiments_dow"
target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/ql.jsonl"
# create_json_ql_file(path,target_path)
##5. run codeql analysis

def run_codeql_analysis(db_path, json_path ,target_path,extra_name):
    db_folder_list = os.listdir(db_path)
    ## read json
    json_list = []
    with jsonlines.open(json_path,'r') as f:
        for obj in f:
            json_list.append(obj)

    for db_folder in db_folder_list:
        sce_id = 'cwe-' + db_folder.replace(extra_name,"").replace('-',"_")
        # print(sce_id)
        db_folder_path = os.path.join(db_path,db_folder)
        for obj in json_list:
            # print(obj)
            if obj['scenario_id'] == sce_id:
                ql_path = obj['check_ql']
                # print(sce_id)
                # print(ql_path)
                break
        if ql_path == 'manually':
            
            continue

        ## run command
        command = f"/home/junjie/Desktop/vul_project/tools/codeql-linux64/codeql/codeql database analyze {os.path.join(db_folder_path)} {ql_path}  --format=csv --output={os.path.join(target_path,db_folder+'.csv')} --additional-packs=/home/junjie/.codeql/packages"
        result = os.system(command)
        print(f"{db_folder} finished")

json_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/ql1.jsonl"

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/results"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/split"
# extra_name = 'fine-tuned-'
# run_codeql_analysis(db_path, json_path, target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/results"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m2-function-level/split"
# extra_name = 'fine-tuned-'
# run_codeql_analysis(db_path, json_path, target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/results"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/split"
# extra_name = 'fine-tuned-'
# run_codeql_analysis(db_path, json_path, target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/results"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/split"
# extra_name = 'fine-tuned-'
# run_codeql_analysis(db_path, json_path, target_path,extra_name)

# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/results"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/db"
# path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/split"
# extra_name = 'pre-trained-'
# run_codeql_analysis(db_path, json_path, target_path,extra_name)

# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/shorter_tokens/fine-tuned-100-db"
# json_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/ql.jsonl"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/shorter_tokens/fine-tuned-100-results"
# extra_name = r"fine-tuned-"
# run_codeql_analysis(db_path, json_path, target_path,extra_name)


##6. statistics analysis
def get_statistics_results_c(source_file_path, results_path, extra_name,save_path,exclude_list = ['416-2','20-2'] ):
    ##1. syntactic valid
    ##2. compilation valid
    ##3. vulnerable
    ##4. non-vulnerable
    results_list = []
    head = ["sce_id","syn_val","compile_val","vul","vul_uncompile"]
    for sce_folder in os.listdir(source_file_path):
        if sce_folder in exclude_list:
            continue
        syn_val = []
        compile_val = []
        vul = []
        non_vul = []
        vul_uncompile = []
        sce_path = os.path.join(source_file_path,sce_folder)
        sce_files = os.listdir(sce_path)
        for sce_file in sce_files:
            # print(sce_file)
            if sce_file.endswith('.reject'):
                syn_val.append(sce_file.replace(".c.reject","").split('_')[-1])
            if sce_file.endswith('.c'):
                syn_val.append(sce_file.replace(".c","").split('_')[-1])
                compile_val.append(sce_file.replace(".c","").split('_')[-1])
        result_path = os.path.join(results_path,extra_name+sce_folder+".csv")
        # df_result = pd.read_csv(result_path)
        print(result_path)
        print(os.path.exists(result_path))
        if os.path.exists(result_path):

            with open(result_path,newline ='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # print(row)
                    for item in row:
                        if item.startswith('/') and item.endswith('.c'):
                            ###  vul file
                            source_name = item.replace('/','')
                            source_num = item.split('_')[-1].replace('.c','')
                            
                            ## if vul is not in compile val, then don't add it
                            if source_num not in vul and source_num in compile_val:
                                vul.append(source_num)
                            elif source_num not in vul and source_num not in compile_val:
                                vul_uncompile.append(source_num)
        results_list.append([sce_folder,syn_val,compile_val,vul,vul_uncompile])
    # print(len(results_list))
    results_df = pd.DataFrame(results_list, columns=head)
    print(results_df)
    print(save_path)
    results_df.to_csv(save_path,index=False)
def get_statistics_results_cpp(source_file_path, results_path, extra_name,save_path,exclude_list = ['416-2','20-2'] ):
    ##1. syntactic valid
    ##2. compilation valid
    ##3. vulnerable
    ##4. non-vulnerable
    results_list = []
    head = ["sce_id","syn_val","compile_val","vul","vul_uncompile"]
    for sce_folder in os.listdir(source_file_path):

        if sce_folder in exclude_list:
            continue
        syn_val = []
        compile_val = []
        vul = []
        non_vul = []
        vul_uncompile = []
        sce_path = os.path.join(source_file_path,sce_folder)
        sce_files = os.listdir(sce_path)
        for sce_file in sce_files:
            if sce_file.endswith('.reject'):
                syn_val.append(sce_file.replace(".cpp.reject","").split('_')[-1])
            if sce_file.endswith('.cpp'):
                syn_val.append(sce_file.replace(".cpp","").split('_')[-1])
                compile_val.append(sce_file.replace(".cpp","").split('_')[-1])
        result_path = os.path.join(results_path,extra_name+sce_folder+".csv")
        # df_result = pd.read_csv(result_path)
        if os.path.exists(result_path):

            with open(result_path,newline ='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # print(row)
                    for item in row:
                        if item.startswith('/') and item.endswith('.cpp'):
                            ###  vul file
                            source_name = item.replace('/','')
                            source_num = item.split('_')[-1].replace('.cpp','')
                            
                            ## if vul is not in compile val, then don't add it
                            if source_num not in vul and source_num in compile_val:
                                vul.append(source_num)
                            elif source_num not in vul and source_num not in compile_val:
                                vul_uncompile.append(source_num)
        results_list.append([sce_folder,syn_val,compile_val,vul,vul_uncompile])
    # print(len(results_list))
    results_df = pd.DataFrame(results_list, columns=head)
    print(results_df)
    print(save_path)
    results_df.to_csv(save_path,index=False)

# results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq1/c/pre-trained-100-results"
# source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq1/c/pre-trained-100"
# extra_name = 'pre-trained-'
# save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq1_results/pre-trained-100-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)

results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/results"
source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m1-file-level/split"
extra_name = 'fine-tuned-'
save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/re_final_results/re-m1-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)

results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/results"
source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m3-line-level/split"
extra_name = 'fine-tuned-'
save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/re_final_results/re-m3-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)

results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/results"
source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/m4-block-level/split"
extra_name = 'fine-tuned-'
save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/re_final_results/re-m4-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)

extra_name = r"pre-trained-"
source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/split"
results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/re-scenarios/pre-trained/results"
save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/re_final_results/re-pre-trained-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)

# extra_name = r"fine-tuned-"
# source_file_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/processed_dataset/split_m6"
# results_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/processed_dataset/split_m6-results"
# save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/processed_dataset/m6-results.csv"
# get_statistics_results(source_file_path, results_path, extra_name,save_path)


def print_statistics(path,total_cases):
    rank_cwe = ['787-0','787-1','787-2','78-0','79-2','20-0','125-0','125-1','125-2','78-0','78-1','78-2',
    '416-0','416-1','22-0','476-0','476-1','476-2','190-0','190-1','190-2','119-0','119-1','119-2','732-0','732-1']
    df = pd.read_csv(path,converters={"vul": literal_eval,"syn_val":literal_eval,"compile_val":literal_eval,"vul_uncompile":literal_eval})
    for cwe in rank_cwe:
        for id,row in df.iterrows():
            invalid = total_cases - len(row['compile_val'])
            if cwe == row['sce_id']:
                print(f"{row['sce_id']:}")
                print(f"vul:{len(row['vul'])}")
                print(f"non-vul:{len(row['compile_val']) - len(row['vul'])}")
                print(f"invalid:{total_cases - len(row['compile_val'])}")
                print()

# path =  r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq2_results/c/line-level-results.csv"
# # print(path)
# print_statistics(path,30)

def process_concat_csv_files(original_path,target_path,save_path):
    df_ori = pd.read_csv(original_path,converters={"vul": literal_eval,"syn_val":literal_eval,"compile_val":literal_eval,"vul_uncompile":literal_eval})
    df_save=pd.read_csv(target_path,converters={"vul": literal_eval,"syn_val":literal_eval,"compile_val":literal_eval,"vul_uncompile":literal_eval})
    save_list = []
    sce_id_list= []
    head = ["sce_id","syn_val","compile_val","vul","vul_uncompile"]
    for ids,row in df_save.iterrows():
        save_list.append([row["sce_id"],row["syn_val"],row["compile_val"],row["vul"],row["vul_uncompile"]])
        sce_id_list.append(row["sce_id"])
    for ids,row in df_ori.iterrows():
        if row["sce_id"] not in sce_id_list:
            save_list.append([row["sce_id"],row["syn_val"],row["compile_val"],row["vul"],row["vul_uncompile"]])
    results_df = pd.DataFrame(save_list, columns=head)
    results_df.to_csv(save_path,index=False)
# original_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/datset1.csv"
# target_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/dataset1.csv"
# save_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/pre-trained-v3-full.csv"
# process_concat_csv_files(original_path,target_path,save_path)

def get_total_num(path_list,total_cases):
    for path in path_list:
        total_vul = 0 
        total_non_vul = 0
        total_invalid = 0 
        df = pd.read_csv(path,converters={"vul": literal_eval,"syn_val":literal_eval,"compile_val":literal_eval,"vul_uncompile":literal_eval})
        print(path)
        for id,row in df.iterrows():
            invalid = total_cases - len(row['compile_val'])
            total_invalid+=invalid
            # print(f"{row['sce_id']:}")
            # print(f"vul:{len(row['vul'])}")
            total_vul += len(row['vul'])
            # print(f"non-vul:{len(row['compile_val']) - len(row['vul'])}")
            total_non_vul += len(row['compile_val']) - len(row['vul'])
            # print(f"invalid:{total_cases - len(row['compile_val'])}")
        print(f"total vul:{total_vul}")
        print(f"total non_vul:{total_non_vul}")
        print(f"total invalid:{total_invalid}")
        print()


def main_workflow(gen_code_path,makefile_path,db_path,results_path,statistic_path,ql_rule_path,language):
    
    copy_makefile(makefile_path,gen_code_path)

    if not os.path.exists(db_path):
        os.mkdir(db_path)
    create_database(gen_code_path,db_path,extra_name)
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    run_codeql_analysis(db_path, ql_rule_path ,results_path,extra_name)
    if language == 'c':
        get_statistics_results_c(gen_code_path, results_path, extra_name,statistic_path,exclude_list = ['416-2','20-2'] )
    elif language == 'cpp':
        get_statistics_results_cpp(gen_code_path, results_path, extra_name,statistic_path,exclude_list = ['416-2','20-2'] )


if __name__ == "__main__":
    fire.Fire(main_workflow)
source_path = ""


split_path = ""
split_path = ''

split_path_c_gptj_pretrained =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq1/c/pre-trained"
split_path_cpp_gptj_pretrained = r"//home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq1/cpp/pre-trained"
split_path_c_gptj_finefile = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/file-level"
split_path_cpp_gptj_finefile = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/file-level/split"
split_path_c_gptj_fineblock = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/block-level"
split_path_cpp_gptj_fineblock = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/block-level/split"
split_path_c_gptj_finefunction = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/function-level"
split_path_cpp_gptj_finefunction = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/function-level/split"
split_path_c_gptj_fineline = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/line-level"
split_path_cpp_gptj_fineline = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/line-level/split"



makefile_c_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/important_files/c_makefile/Makefile"
makefile_cpp_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/important_files/cpp_makefile/Makefile"




results_path_c_gptj_pretrained =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq1/c/pre-trained-results"
results_path_cpp_gptj_pretrained = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq1/cpp/pre-trained-results"
results_path_c_gptj_finefile = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/file-level-results"
results_path_cpp_gptj_finefile = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/file-level/results"
results_path_c_gptj_fineblock = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/block-level-results"
results_path_cpp_gptj_fineblock = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/block-level/results"
results_path_c_gptj_finefunction = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/function-level-results"
results_path_cpp_gptj_finefunction = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/function-level/results"
results_path_c_gptj_fineline = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/c/line-level-results"
results_path_cpp_gptj_fineline = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/gpt-j/rq2/cpp/line-level/results"


statistic_path_c_gptj_pretrained =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-pretrained-c.csv"
statistic_path_cpp_gptj_pretrained =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-pretrained-cpp.csv"
statistic_path_c_gptj_finefile =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-finefile-c.csv"
statistic_path_cpp_gptj_finefile =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-finefile-cpp.csv"
statistic_path_c_gptj_fineblock =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-fineblock-c.csv"
statistic_path_cpp_gptj_fineblock =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-fineblock-cpp.csv"
statistic_path_c_gptj_fineline =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-fineline-c.csv"
statistic_path_cpp_gptj_fineline =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-fineline-cpp.csv"
statistic_path_c_gptj_finefunction =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-finefunction-c.csv"
statistic_path_cpp_gptj_finefunction =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/gpt-j-finefunction-cpp.csv"




json_path = r"../config_files/ql.jsonl"


args_list = []
## c codellama pretrained
extra_name = 'fine-tuned-'
# args_list.append([source_path,split_path_c_gptj_finefile,makefile_c_path,extra_name,'',results_path_c_gptj_finefile,statistic_path_c_gptj_finefile,json_path])
# args_list.append([source_path,split_path_c_gptj_finefunction,makefile_c_path,extra_name,'',results_path_c_gptj_finefunction,statistic_path_c_gptj_finefunction,json_path])
# args_list.append([source_path,split_path_c_gptj_fineblock,makefile_c_path,extra_name,'',results_path_c_gptj_fineblock,statistic_path_c_gptj_fineblock,json_path])
# args_list.append([source_path,split_path_c_gptj_fineline,makefile_c_path,extra_name,'',results_path_c_gptj_fineline,statistic_path_c_gptj_fineline,json_path])

args_list.append([source_path,split_path_cpp_gptj_finefile,makefile_cpp_path,extra_name,'',results_path_cpp_gptj_finefile,statistic_path_cpp_gptj_finefile,json_path])
args_list.append([source_path,split_path_cpp_gptj_finefunction,makefile_cpp_path,extra_name,'',results_path_cpp_gptj_finefunction,statistic_path_cpp_gptj_finefunction,json_path])
args_list.append([source_path,split_path_cpp_gptj_fineblock,makefile_cpp_path,extra_name,'',results_path_cpp_gptj_fineblock,statistic_path_cpp_gptj_fineblock,json_path])
args_list.append([source_path,split_path_cpp_gptj_fineline,makefile_cpp_path,extra_name,'',results_path_cpp_gptj_fineline,statistic_path_cpp_gptj_fineline,json_path])

for i, each_args_list in enumerate(args_list):
    print(f"start to process:{i}/{len(args_list)}")
    main_workflow(each_args_list[0],each_args_list[1],each_args_list[2],each_args_list[3],each_args_list[4],each_args_list[5],each_args_list[6],each_args_list[7])


### only for test
# db_path_c_codegen_pretrained = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codegen/rq1/c/pretrained_db"
# db_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codellama/rq1/cpp/db_test"
# split_path =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codellama/rq1/cpp/test_folder"
# extra_name = 'pretrained'
# makefile_path =""
# extra_name = ""
# statistic_path = ""
# json_path = ""
# main_workflow("",split_path,makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# source_path = ""
# split_path =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codellama/rq1/cpp/pre-trained"
# makefile_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/important_files/cpp_makefile/Makefile"
# db_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codellama/rq1/cpp/db"
# results_path =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codellama/rq1/cpp/results"

# statistic_path =r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/statistics_results/codellama-pretrained-ori-cpp-results.csv"
# extra_name = 'pretrained'
# json_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/important_files/ql1.jsonl"

# main_workflow(source_path,split_path,makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/function-level/cleaned"

# makefile_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/cpp_makefile/Makefile"
# json_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/ql1.jsonl"

# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/function-level/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/function-level/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/function-level/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq2_results/cpp/function-level-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/block-level/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/block-level/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/block-level/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/block-level/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq2_results/cpp/block-level-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/line-level/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/line-level/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/line-level/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq2/cpp/line-level/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq2_results/cpp/line-level-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)


# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m1/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m1/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m1/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m1/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m1-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m2/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m2/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m2/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m2/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m2-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m3/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m3/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m3/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m3/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m3-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m4/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m4/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m4/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m4/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m4-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m5/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m5/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m5/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m5/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m5-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)

# cleaned_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m6/cleaned"
# split_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m6/split"
# db_path = r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m6/db"
# results_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/rq3/cpp/m6/results"
# statistic_path =r"/home/junjie/Desktop/vul_project/tools/CWE_scenarios/important_files/rq3_results/cpp/m6-results.csv"
# extra_name = r"fine-tuned-"
# main_workflow(cleaned_path,split_path, makefile_path,extra_name,db_path,results_path,statistic_path,json_path)