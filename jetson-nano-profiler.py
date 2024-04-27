# from nano_llm import NanoLLM
import os
import json

#################### CONSTANTS ####################
MAX_NEW_TOKENS = 512
MODEL = "meta-llama/Llama-2-7b-chat-hf"
# PROMPT_SET = "data/prompts/ShareGPT_V3_unfiltered_cleaned_split_top100.json"
PROMPT_SET = "shareGPT.json"
##################################################

###################### UTILS ######################
def process_shareGPT_json(file_path):
    cache_path = file_path.replace('.json', '.cache')
    
    # Check if cache file exists
    if os.path.exists(cache_path):
        print(f"Reading from cache: {cache_path}")
        with open(cache_path, 'r') as cache_file:
            data = json.load(cache_file)
    else:
        # Read the original JSON file
        print(f"Reading from JSON file: {file_path}")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Process the conversations to get the first two "human" messages
        processed_data = []
        for entry in data:
            human_messages = [conv['value'] for conv in entry['conversations'] if conv['from'] == 'human'][:2]
            processed_data.extend(human_messages)
        
        # Save the processed data to cache file
        with open(cache_path, 'w') as cache_file:
            json.dump(processed_data, cache_file, indent=4)
        
        data = processed_data
    
    return data

###################################################

prompts = process_shareGPT_json(PROMPT_SET)
print(prompts)

# model = NanoLLM.from_pretrained(
#    MODEL,                                    # HuggingFace repo/model name, or path to HF model checkpoint
#    api='mlc',                                # supported APIs are: mlc, awq, hf
#    api_token=os.environ['HUGGINFACE_TOKEN'], # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
#    quantization='q4f16_ft',                  # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
# )

# for p in prompts:
#     response = model.generate(p, max_new_tokens=MAX_NEW_TOKENS)

#     for token in response:
#         print(token, end='', flush=True)
