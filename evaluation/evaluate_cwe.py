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

##2. move make files
def copy_makefile(path,target_path):
    folder_list = os.listdir(target_path)
    for folder in folder_list:
        shutil.copy2(path,os.path.join(target_path,folder))


def create_database(path,target_path):
    folder_list = os.listdir(path)
    for folder in folder_list:

        command = f'codeql database create --language=cpp --overwrite --command="make -B" --source-root {os.path.join(path,folder)} {os.path.join(target_path,folder)}'
        result = os.system(command)
        print(folder)
        print(result)
        print()


##5. run codeql analysis

def run_codeql_analysis(db_path, json_path ,target_path):
    db_folder_list = os.listdir(db_path)
    ## read json
    json_list = []
    with jsonlines.open(json_path,'r') as f:
        for obj in f:
            json_list.append(obj)

    for db_folder in db_folder_list:
        sce_id = 'cwe-' + db_folder.replace('-',"_")
        db_folder_path = os.path.join(db_path,db_folder)
        for obj in json_list:
            if obj['scenario_id'] == sce_id:
                ql_path = obj['check_ql']
                break
        if ql_path == 'manually':
            
            continue

        ## run command
        command = f"codeql database analyze {os.path.join(db_folder_path)} {ql_path}  --format=csv --output={os.path.join(target_path,db_folder+'.csv')} --additional-packs=~/.codeql/packages"
        result = os.system(command)
        print(f"{db_folder} finished")



##6. statistics analysis
def get_statistics_results_c(source_file_path, results_path, save_path,exclude_list = ['416-2','20-2'] ):
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
        result_path = os.path.join(results_path,sce_folder+".csv")
        # df_result = pd.read_csv(result_path)
        print(result_path)
        print(os.path.exists(result_path))
        if os.path.exists(result_path):

            with open(result_path,newline ='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
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
    results_df = pd.DataFrame(results_list, columns=head)
    print(results_df)
    print(save_path)
    results_df.to_csv(save_path,index=False)
def get_statistics_results_cpp(source_file_path, results_path, save_path,exclude_list = ['416-2','20-2'] ):
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
        result_path = os.path.join(results_path,sce_folder+".csv")
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
    results_df = pd.DataFrame(results_list, columns=head)
    print(results_df)
    print(save_path)
    results_df.to_csv(save_path,index=False)




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
    create_database(gen_code_path,db_path)
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    run_codeql_analysis(db_path, ql_rule_path ,results_path)
    if language == 'c':
        get_statistics_results_c(gen_code_path, results_path,statistic_path,exclude_list = ['416-2','20-2'] )
    elif language == 'cpp':
        get_statistics_results_cpp(gen_code_path, results_path,statistic_path,exclude_list = ['416-2','20-2'] )


if __name__ == "__main__":
    fire.Fire(main_workflow)