from curses import pair_content
import json
import os
import sys
import jsonlines
import os
import fire


def extract_cpp(raw):
	start = raw.find('{')
	if raw[start+1] == '\n':
		true_start = start

	assert raw[start] == '{'

	# the following code will extract the method body by pairing the '{' and '}'
	curr = start + 1
	quote = False
	quote_types = ['"', "'"]
	quote_index = 0
	pairs = 1
	while curr < len(raw):
		# check quotes
		if quote:
			if raw[curr] == quote_types[quote_index]:
				quote = not quote
		else:
			if raw[curr] in quote_types: # quote
				quote_index = quote_types.index(raw[curr])
				quote = not quote
			else:
				if raw[curr:curr+2] == '//': # comment
					if '\n' in raw[curr:]:
						curr += raw[curr:].index('\n') # now curr points to the '\n', correct?
				elif raw[curr:curr+2] == '/*': # comment
					if '*/' in raw[curr+2:]:
						curr += 2 + raw[curr+2:].index('*/') + 1 # now curr points to the '/' of '*/', correct?
				elif raw[curr] == '{': # block
					pairs += 1
				elif raw[curr] == '}': # block end
					pairs -= 1
					if pairs == 0:
						break
		curr += 1
	if pairs == 0:
		return raw[:curr+1]
	else:
		return ""

def extract_cpp_second_brace(raw):
	pair_count = 0
	start = raw.find('{')
	if raw[start+1] == '\n':
		true_start = start

	assert raw[start] == '{'
	# the following code will extract the method body by pairing '{' and '}'
	curr = start+1
	quote = False
	quote_types = ['"', "'"]
	quote_index = 0
	pairs = 1
	while curr < len(raw):
		# check quotes
		if quote:
			if raw[curr] == quote_types[quote_index]:
				quote = not quote
		else:
			
			if raw[curr] in quote_types: # quote
				quote_index = quote_types.index(raw[curr])
				quote = not quote
			else:
				if raw[curr:curr+2] == '//': # comment
					if '\n' in raw[curr:]:
						curr += raw[curr:].index('\n') # now curr points to the '\n', correct?
				elif raw[curr:curr+2] == '/*': # comment
					if '*/' in raw[curr+2:]:
						curr += 2 + raw[curr+2:].index('*/') + 1 # now curr points to the '/' of '*/', correct?
				elif raw[curr] == '{': # block
					pairs += 1
				elif raw[curr] == '}': # block end

					pairs -= 1
					if pairs == 0 and pair_count == 1:
						break
					elif pairs == 0 and pair_count == 0:
						pair_count +=1
		curr += 1
	if pairs == 0 and pair_count ==1 :
		return raw[:curr+1]
	else:
		return ""


def extract_cpp_add_brackets(raw):

	start = raw.find('{')
	if raw[start+1] == '\n':
		true_start = start

	assert raw[start] == '{'

	# the following code will extract the method body by pairing the '{' and '}'
	curr = start + 1
	quote = False
	quote_types = ['"', "'"]
	quote_index = 0
	pairs = 1
	while curr < len(raw):
		# check quotes
		if quote:
			if raw[curr] == quote_types[quote_index]:
				quote = not quote
		else:
			if raw[curr] in quote_types: # quote
				quote_index = quote_types.index(raw[curr])
				quote = not quote
			else:
				if raw[curr:curr+2] == '//': # comment
					if '\n' in raw[curr:]:
						curr += raw[curr:].index('\n') # now curr points to the '\n', correct?
				elif raw[curr:curr+2] == '/*': # comment
					if '*/' in raw[curr+2:]:
						curr += 2 + raw[curr+2:].index('*/') + 1 # now curr points to the '/' of '*/', correct?
				elif raw[curr] == '{': # block
					pairs += 1
				elif raw[curr] == '}': # block end
					pairs -= 1
					if pairs == 0:
						break
		curr += 1
	if pairs == 0:
		return raw[:curr+1]
	elif pairs > 0:
		## remove the last line and ad a }
		raw = remove_last_line(raw)
		for i in range(pairs):
			raw += "\n}"
		return raw
	else:
		return ""

def extract_cpp_second_brace_add_brackets(raw):
	pair_count = 0
	start = raw.find('{')
	if raw[start+1] == '\n':
		true_start = start

	assert raw[start] == '{'
	# the following code will extract the method body by pairing the '{' and '}'
	curr = start+1
	quote = False
	quote_types = ['"', "'"]
	quote_index = 0
	pairs = 1
	while curr < len(raw):
		# check quotes
		if quote:
			# print(f"1,{raw[curr]}")
			if raw[curr] == quote_types[quote_index]:
				quote = not quote
		else:
			
			if raw[curr] in quote_types: # quote
				quote_index = quote_types.index(raw[curr])
				quote = not quote
				# print(f"2,{raw[curr]}")
			else:
				if raw[curr:curr+2] == '//': # comment
					# print(f"3,{raw[:curr+2]}")
					if '\n' in raw[curr:]:
						curr += raw[curr:].index('\n') # now curr points to the '\n', correct?
				elif raw[curr:curr+2] == '/*': # comment
					if '*/' in raw[curr+2:]:
						curr += 2 + raw[curr+2:].index('*/') + 1 # now curr points to the '/' of '*/', correct?
				elif raw[curr] == '{': # block
					# print(1111)
					# print(raw[:curr+1])
					pairs += 1
				elif raw[curr] == '}': # block end
					# print(2222)
					# print(raw[:curr+1])
					pairs -= 1
					if pairs == 0 and pair_count == 1:
						# print("there is one pair_count1")
						# print(raw[:curr+1])
						break
					elif pairs == 0 and pair_count == 0:
						# print(raw[:curr+1])
						pair_count +=1
		curr += 1
	if pairs == 0 and pair_count ==1 :
		return raw[:curr+1]
	elif pairs > 0:
		## remove the last line and ad a }
		raw = remove_last_line(raw)
		for i in range(pairs):

			
			
			raw += "\n}"
		return raw
	else:
		return ""



def remove_last_line(input_string):
	# Split the string into lines
	lines = input_string.split('\n')
	
	# Remove the last line if there are more than one line
	if len(lines) > 1:
		lines.pop()  # Remove the last line
	
	# Join the lines back together
	result_string = '\n'.join(lines)
	
	return result_string

def remove_head(raw_content,input_str):
	if len(raw_content.split(input_str)) >1:
		return_code = raw_content.split(input_str)[1]
		return return_code
	else:
		return_code = raw_content.split(input_str)[0]
		return return_code

raw_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/gen_raw_results/pretrained_codegen2-7b.jsonl"

c_cleaned_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codegen/rq1/c/pre-trained"
cpp_cleaned_path = r"/home/junjie/Desktop/secure_llm/datasets/evaluate_dataset/codeql_results_ori/codegen/rq1/cpp/pre-trained"

def clean_code(raw_path,c_cleaned_path,cpp_cleaned_path):
	print(
            f"Post-process generated code:\n"
            f"raw_path: {raw_path}\n"
            f"c_cleaned_path: {c_cleaned_path}\n"
            f"cpp_cleaned_path: {cpp_cleaned_path}\n"
        )
	error_i = 0
	reader_list = []
	cwe_list = ["79-2","476-0","476-1","787-1","79-0"]
	with jsonlines.open(raw_path) as reader:
		for obj in reader:
			reader_list.append(obj)
	for each in reader_list:
		# if '787-1-mitre' == each['case_id']:
		# 	for i in range(1,31):
		# 		raw_code = each[f"output_{i}"].split("Please complete the code\n")[1].split('</s>')[0]
		# 		# print(extract_cpp(raw_code))
		# 		process_code = extract_cpp(raw_code)
		# 		print(raw_code)
		# 		print('*' * 50)
		cwe_names = each['case_id'].split('-')
		cwe_name = '-'.join(cwe_names[:2])

		# if each['case_id'] != '20-2-authors':
		# 	continue
		if each['language'] == 'c':
			save_base_path = c_cleaned_path
			file_type = '.c'
		elif each['language'] == 'cpp':
			save_base_path = cpp_cleaned_path
			file_type = '.cpp'

		tokens = each['case_id'].split('-')[:2]
		sce = '-'.join(tokens)
		print(sce)
		##create a folder
		if not os.path.exists(os.path.join(save_base_path,sce)):
			os.mkdir(os.path.join(save_base_path,sce))

		save_folder_path = os.path.join(save_base_path,sce)

		
		for i in range(1,31):
			print(each['case_id'])
			file_name = each['case_id'] + '_' + str(i) +file_type
			save_path = os.path.join(save_folder_path, file_name)
			code = ''
			if cwe_name in cwe_list:
				assert each[f"input"] in each[f"output_{i}"]
				code = extract_cpp_second_brace_add_brackets(each[f"input"]+remove_head(each[f"output_{i}"],each[f"input"]))
				# code = remove_head(each[f"output_{i}"],each[f"input"])
				print(code)
				# print(each[f"output_{i}"])
				print('*' * 50)
				if	len(code) == 0:
					error_i +=1
					continue

			else:
				assert each[f"input"] in each[f"output_{i}"]
				code = extract_cpp_add_brackets(each[f"input"]+remove_head(each[f"output_{i}"],each[f"input"]))
				# code = remove_head(each[f"output_{i}"],each[f"input"])
				print(code)
				# each[f"output_{i}"]
				print('*' * 50)
				if	len(code) == 0:
					error_i +=1
					continue

			with open(save_path,'w') as f:
				f.write(code)


if __name__ == "__main__":
    fire.Fire(clean_code)
