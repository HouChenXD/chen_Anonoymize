import json
import sys
import os

def process_json_file(input_file, output_file):
    """process json file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # confirm it's a list
        if not isinstance(json_data, list):
            json_data = [json_data]
        
        processed_data = []
        for item in json_data:
            prefix = item.get('prefix', '')
            output = item.get('output', '')
            
            # senstive prefix detection
            if output.startswith(('AIza', 'sk_test_', 'https://hooks.slack.com/services/', 'AKID', 'LTAI')):
                processed_data.append(item)
                continue
                
            item['output'] = prefix + output
            processed_data.append(item)
        
        # saved as jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Processed JSON file and saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

def process_jsonl_file(input_file, output_file):
    try:
        processed_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    prefix = item.get('prefix', '')
                    output = item.get('output', '')
                    
                    # sensitive prefix detection
                    if output.startswith(('AIza', 'sk_test_', 'https://hooks.slack.com/services/', 'AKID', 'LTAI')):
                        processed_data.append(item)
                        continue
                        
                    item['output'] = prefix + output
                    processed_data.append(item)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
        
        # saved processed jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Processed JSONL file and saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file.jsonl>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # access file exten
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.json':
        process_json_file(input_file, output_file)
    elif file_ext == '.jsonl':
        process_jsonl_file(input_file, output_file)
    else:
        print("Error: Unsupported file extension. Use .json or .jsonl")
        sys.exit(1)

if __name__ == "__main__":
    main()