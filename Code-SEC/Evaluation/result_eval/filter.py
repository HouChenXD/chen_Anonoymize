#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import re
import sys
import unicodedata
import math
import json


# Specify the JSONL file to process
jsonl_file = sys.argv[1]

df_re = pd.read_csv('secret_re_list.csv', header=0)
df_result = pd.DataFrame(
    columns=['secret_type', 'result_after_regex_filter'])


def shannon_entropy(string):
    freq_dict = {}
    for char in string:
        if char in freq_dict:
            freq_dict[char] += 1
        else:
            freq_dict[char] = 1

    entropy = 0.0
    str_len = len(string)
    for freq in freq_dict.values():
        prob = freq / str_len
        entropy -= prob * math.log2(prob)

    return entropy


INVALID_CHARS = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n\r"
MIN_WORD_LENGTH = 4


def filter_characters(string, invalid_chars=INVALID_CHARS):
    chars = []
    for c in string:
        if c in invalid_chars:
            continue
        chars.append(c)
    s = "".join(chars)
    s = ''.join((c for c in unicodedata.normalize(
        'NFD', s) if unicodedata.category(c) != 'Mn'))
    return s.lower()


class WordsFinder(object):
    def __init__(self, wordlists):
        self.dictionary = None
        self.max_length = 0
        if wordlists:
            self.dictionary = set()
            for txt in wordlists:
                if os.path.exists(txt):
                    for line in open(txt, "r"):
                        word = filter_characters(line)
                        if len(word) > self.max_length:
                            self.max_length = len(word)
                        self.dictionary.add(word)

    def get_words_indexes(self, string):
        string = filter_characters(string)
        if len(string) < MIN_WORD_LENGTH:
            return
        if not self.dictionary:
            print("Dictionary uninitalized!")
            return
        i = 0
        while i < len(string) - (MIN_WORD_LENGTH - 1):
            chunk = string[i:i + self.max_length]
            found = False
            for j in range(len(chunk), MIN_WORD_LENGTH - 1, -1):
                candidate = chunk[:j]
                if candidate in self.dictionary:
                    yield (i, j, candidate)
                    found = True
                    i += j
                    break
            if not found:
                i += 1

    def count_word_length(self, string):
        word_length_count = 0
        for i in self.get_words_indexes(string):
            word_length_count += i[1]
        return word_length_count


class StringsFilter(object):
    def __init__(self):
        wordlists = []
        for path in ['computer_wordlist.txt']:
            if os.path.exists(path):
                wordlists.append(os.path.join('.', path))
        self.finder = WordsFinder(wordlists)

    def word_filter(self, string):
        return self.finder.count_word_length(string)


def pattern_filter(input_string):
    for i in range(len(input_string) - 3):
        if input_string[i] == input_string[i + 1] == input_string[i + 2] == input_string[i + 3]:
            return True

    for i in range(len(input_string) - 3):
        if input_string[i:i+4] in [''.join(map(chr, range(ord(input_string[i]), ord(input_string[i])+4))),
                                   ''.join(map(chr, range(ord(input_string[i].lower()), ord(input_string[i].lower())+4)))]:
            return True

    for i in range(len(input_string) - 3):
        if input_string[i:i+4] in [''.join(map(chr, range(ord(input_string[i]), ord(input_string[i])-4, -1))),
                                   ''.join(map(chr, range(ord(input_string[i].lower()), ord(input_string[i].lower())-4, -1)))]:
            return True

    for i in range(len(input_string) - 5):
        if input_string[i] == input_string[i + 2] == input_string[i + 4] and input_string[i + 1] == input_string[i + 3] == input_string[i + 5]:
            return True
        
    for i in range(len(input_string) - 8):
        if input_string[i:i+3] == input_string[i+3:i+6] == input_string[i+6:i+9]:
            return True

    for i in range(len(input_string) - 7):
        if input_string[i:i+4] == input_string[i+4:i+8]:
            return True

    return False


compl_count = 0

# Extract base name from file path
base_name = os.path.splitext(os.path.basename(jsonl_file))[0]
input_dir = os.path.dirname(jsonl_file) 

print(f'Processing file: {jsonl_file}')

if not os.path.exists(jsonl_file):
    print(f'Error: File {jsonl_file} does not exist')
    sys.exit(1)

try:
    # Initialize filter
    s_filter = StringsFilter()
    
    cleaned_jsonl_data = []  # Save cleaned JSONL data
    all_entropies = []  # Collect all entropy values for threshold calculation
    temp_secrets = []  # Temporary storage for secrets
    
    # First pass: read all lines and collect secrets/entropy
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                # Observe that some items have two types api_key or api_type
                s_type = None
                if 'api_key' in item:
                    s_type = item['api_key']
                elif 'api_type' in item:
                    s_type = item['api_type']
                
                if not s_type:
                    continue
                
                # Get corresponding regex pattern
                re_row = df_re[df_re['secret_id'] == s_type]
                if len(re_row) == 0:
                    print(f'Warning: No regex pattern found for secret type {s_type} on line {line_num+1}')
                    continue
                
                re_pattern = re_row['RE'].values[0]
                
                # Extract secrets from output field
                text_to_search = None
                if 'output' in item and item['output']:
                    text_to_search = item['output']
                
                if text_to_search:
                    # Use regex to extract secrets
                    matches = re.findall(re_pattern, text_to_search)
                    for match in matches:
                        if match and len(match.strip()) > 0:
                            entropy = shannon_entropy(match)
                            all_entropies.append(entropy)
                            temp_secrets.append({'secret': match, 'entropy': entropy, 'line_num': line_num})
            except json.JSONDecodeError:
                print(f'Warning: Invalid JSON on line {line_num+1}: {line[:50]}...')
                continue
    
    # Calculate entropy threshold
    if all_entropies:
        mean_entropy = sum(all_entropies) / len(all_entropies)
        variance = sum((x - mean_entropy) **2 for x in all_entropies) / len(all_entropies)
        std_entropy = math.sqrt(variance)
        entropy_threshold = mean_entropy - 3 * std_entropy
    else:
        entropy_threshold = 0
    
    # Second pass: process each line and apply filters
    secrets_processed = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                cleaned_item = item.copy()  # Copy original item
                extracted_secrets_with_filters = []  # Store secret info with filter results

                # NOTE: Support both formats: api_key or api_type
                s_type = None
                if 'api_key' in item:
                    s_type = item['api_key']
                elif 'api_type' in item:
                    s_type = item['api_type']
                
                if not s_type:
                    cleaned_jsonl_data.append(cleaned_item)
                    continue
                
                # Get corresponding regex pattern
                re_row = df_re[df_re['secret_id'] == s_type]
                if len(re_row) == 0:
                    cleaned_jsonl_data.append(cleaned_item)
                    continue
                
                re_pattern = re_row['RE'].values[0]
                
                # Extract secrets from output field
                text_to_search = None
                if 'output' in item and item['output']:
                    text_to_search = item['output']
                
                if text_to_search:
                    # Use regex to extract secrets
                    matches = re.findall(re_pattern, text_to_search)
                    for match in matches:
                        if match and len(match.strip()) > 0:
                            entropy = shannon_entropy(match)
                            
                            # Apply four filters
                            # 1. regex_filter - always False (already passed regex match)
                            regex_filter_result = False
                            
                            # 2. entropy_filter
                            entropy_filter_result = entropy < entropy_threshold
                            
                            # 3. pattern_filter
                            pattern_filter_result = pattern_filter(match)
                            
                            # 4. word_filter
                            test_secret = match
                            if s_type == 'google_oauth_client_id':
                                test_secret = match.replace('.apps.googleusercontent.com', '')
                                word_filter_result = s_filter.word_filter(test_secret) >= MIN_WORD_LENGTH
                            elif s_type == 'ebay_production_client_id':
                                word_filter_result = False
                            else:
                                # Remove prefix for word filtering
                                prefix = re_row['Prefix'].values[0]
                                if isinstance(prefix, str) and len(prefix) > 0:
                                    test_secret = match[len(prefix):]
                                word_filter_result = s_filter.word_filter(test_secret) >= MIN_WORD_LENGTH
                            
                            # Determine if valid
                            is_valid = not (entropy_filter_result or pattern_filter_result or word_filter_result)
                            
                            # Create secret object with filter results
                            secret_with_filters = {
                                'secret': match,
                                'regex_filter': regex_filter_result,
                                'entropy_filter': entropy_filter_result,
                                'pattern_filter': pattern_filter_result,
                                'word_filter': word_filter_result,
                                'valid': is_valid
                            }
                            
                            extracted_secrets_with_filters.append(secret_with_filters)
                            
                            # Also save to df_result for CSV output
                            new_row = pd.DataFrame([{
                                'secret_type': s_type, 
                                'result_after_regex_filter': match, 
                                'extracted_secret_entropy': entropy,
                                'regex_filter': regex_filter_result,
                                'entropy_filter': entropy_filter_result,
                                'pattern_filter': pattern_filter_result,
                                'word_filter': word_filter_result,
                                'valid': is_valid
                            }])
                            df_result = pd.concat([df_result, new_row], ignore_index=True)
                            secrets_processed += 1
                
                # Add secret info with filter results to JSON object
                cleaned_item['PS_extracted_secrets'] = extracted_secrets_with_filters
                
                cleaned_jsonl_data.append(cleaned_item)
            except json.JSONDecodeError:
                print(f'Warning: Skipping invalid JSON on line {line_num+1}: {line[:50]}...')
                continue
    
    # Save cleaned JSONL file
    cleaned_jsonl_filename = f"{base_name}_cleaned.jsonl"
    cleaned_jsonl_path = os.path.join(input_dir, cleaned_jsonl_filename)
    
    with open(cleaned_jsonl_path, 'w', encoding='utf-8') as f:
        for item in cleaned_jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f'Processed {jsonl_file} successfully - found {secrets_processed} secrets from {len(cleaned_jsonl_data)} JSON objects')
    print(f'Saved cleaned JSONL to: {cleaned_jsonl_path}')
    compl_count = 1
    
except Exception as e:
    print(f'Error: Failed to process file {jsonl_file}: {e}')
    sys.exit(1)

valid_count = 0

if len(df_result) > 0:
    # Count valid secrets
    valid_count = len(df_result[df_result['valid'] == True])

print("Complete count:", compl_count)
print("Valid secret count:", valid_count)

output_csv_filename = f'{base_name}.csv'
output_csv_path = os.path.join(input_dir, output_csv_filename)

print(f"Results saved to {output_csv_path}")

if len(df_result) > 0:
    # Reorder columns
    cols = ['secret_type','regex_filter','result_after_regex_filter','entropy_filter','pattern_filter','word_filter','valid']
    df_result = df_result[cols]
    df_result.to_csv(output_csv_path, index=False)
else:
    print("No results to save - creating empty result file")
    empty_df = pd.DataFrame(columns=['secret_type','regex_filter','result_after_regex_filter','entropy_filter','pattern_filter','word_filter','valid'])
    empty_df.to_csv(output_csv_path, index=False)