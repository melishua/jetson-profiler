from nano_llm import NanoLLM
import os
import json
import argparse
import random
from datetime import datetime
from termcolor import cprint

#################### CONSTANTS ####################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a generative model with custom parameters.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model identifier on HuggingFace.')
    parser.add_argument('--prompt_set', type=str, default="data/prompts/ShareGPT_V3_unfiltered_cleaned_split_top100.json", help='Path to the JSON file with prompts.')
    parser.add_argument('--start_signal', type=str, default="START_SIGNAL", help='Filename to signal the start of the experiment.')
    parser.add_argument('--end_signal', type=str, default="END_SIGNAL", help='Filename to signal the end of the experiment.')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of times to repeat the prompt experiment.')
    parser.add_argument('--num_prompt_samples', type=int, default=0, help='Number of prompts to use from the set; 0 means use all.')
    parser.add_argument('--num_prompt_per_iteration', type=int, default=0, help='Number of prompts to use in each iteration from the selected prompt set; 0 means use all.')
    parser.add_argument('--random_seed', type=int, default=37, help='Random seed for reproducibility.')
    parser.add_argument('--disable_streaming', action="store_true", help='Disable streaming generated result.')
    parser.add_argument('--max_input_token_length', type=int, default=1000, help='Maximum input token length for the model.')
    return parser.parse_args()

###################### UTILS ######################
def DATE():
    """ Returns the current date. """
    return datetime.now().strftime("%A, %B %-m %Y")
   
def TIME():
    """ Returns the current time. """
    return datetime.now().strftime("%-I:%M %p")

def process_shareGPT_json(file_path, model, max_input_length):
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
            if len(entry['conversations']) >= 2:
                human_messages = [conv['value'] for conv in entry['conversations'] if conv['from'] == 'human'][:2]
                processed_data.extend(human_messages)
        
        # Filter out promptes based on requirements
        data = parse_shareGPT_data(processed_data, model, max_input_length)
        
        # Save the processed data to cache file
        with open(cache_path, 'w') as cache_file:
            json.dump(data, cache_file, indent=4)
    
    return data

def parse_shareGPT_data(data, model, max_input_length):
    filtered_data = []
    for d in data:
        # Tokenize the prompt
        prompt_tokens = model.tokenize(d)[0]
        
        # Filter out too long or too short sequences
        if len(prompt_tokens) < 4 or len(prompt_tokens) > max_input_length:
            print(f"filtered out prompt with length {len(prompt_tokens)}")
            continue
        # TODO: maybe also filter on GPT's response token length as @Edwin's repo
        else:
            filtered_data.append(d)
    
    return filtered_data

def cleanup_files(*files):
    """Remove specified files."""
    for file in files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except FileNotFoundError:
            print(f"{file} not found for removal.")
            
def cleanup(start_signal, end_signal):
    os.system(f'touch {end_signal}')
    cleanup_files(start_signal)
###################################################

def main():
    args = parse_arguments()

    # Load the model
    model = NanoLLM.from_pretrained(
        args.model,
        api='mlc',
        api_token=os.environ['HUGGINGFACE_TOKEN'],
        quantization='q4f16_ft',
    )
    
    # Get the prompts
    prompts = process_shareGPT_json(args.prompt_set, model, args.max_input_token_length)
    if args.num_prompt_samples > 0:
        random.seed(args.random_seed)
        prompts = random.sample(prompts, min(args.num_prompt_samples, len(prompts)))
    print(prompts)

    cleanup_files(args.end_signal)

    os.system(f'touch {args.start_signal}')

    # Run the prompt experiment for the specified number of iterations
    try:
        print("Starting NanoLLM profiling experiment:")
        print(f"    Number of prompts: {len(prompts)}")
        print(f"    Number of iterations: {args.num_iterations}")
        for i in range(args.num_iterations):
            # Data collection
            print(f"{TIME()}: Starting iteration {i}")
            num_input_tokens = 0
            num_output_tokens = 0
            
            for p_idx, p in enumerate(prompts):
                cprint(f'>> PROMPT ({p_idx+1}/{len(prompts)}): {p}\n', 'blue' , end='', flush=True)
                response = model.generate(p, 
                                          max_new_tokens=args.max_new_tokens,
                                          streaming=not args.disable_streaming)
                if args.disable_streaming:
                    cprint(response, 'green')
                else:
                    for token in response:
                        cprint(token, 'green', end='', flush=True)

            print(f"model stats= {model.stats}")
            # Stats for this prompt set
            # print(f"Number of input tokens: {num_input_tokens}")
            # print(f"Number of output tokens: {num_output_tokens}")        
    except KeyboardInterrupt:
        print("Experiment interrupted by user.")
        print(f"Stopped at iteration {i}/{args.num_iterations}.")
        cleanup(args.start_signal, args.end_signal)
        print("Exiting the experiment, bye!")
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Stopped at iteration {i}/{args.num_iterations}.")
        cleanup(args.start_signal, args.end_signal)
        print("Exiting the experiment, bye!")

    cleanup(args.start_signal, args.end_signal)
    print("Experiment completed :D")
    print("Bye!")

if __name__ == "__main__":
    main()