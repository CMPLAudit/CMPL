import os
import sys
import argparse
import yaml
import logging
import re

from auditor_utils import *

def extract_field_value(data, target_key):
    """
    Recursively search for a key in a nested dictionary and return its value as a string.
    If the key appears multiple times, returns the first match.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return str(value)
            elif isinstance(value, (dict, list)):
                result = extract_field_value(value, target_key)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = extract_field_value(item, target_key)
            if result is not None:
                return result
    return None

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="0.yaml")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(model=model_name, 
            tensor_parallel_size=torch.cuda.device_count(), 
            gpu_memory_utilization=0.9,
            max_model_len = 12800,
            )

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    input = config['input']
    adversary_role = config['adversary_role'] #"an interviewer"
    target_attribute = config['target_attribute']
    profile_data_path = f'/path/to/schedule_data.json'

    with open(input, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    with open(profile_data_path, 'r', encoding='utf-8') as file:
        person_data = yaml.safe_load(file)

    output_file = input.replace(".yaml", "_privacy_audit_judge_may2_gemma.out")
    logging.basicConfig(filename=output_file, level=logging.INFO, format='%(message)s')

    for person_index in data.keys():
        if int(person_index)<args.start_index:
            continue
        conv_history_list = []
        summary_list = []
        iteration_data = data[person_index]
        for i in range(len(iteration_data)//2):
            if int(i)<args.start_round:
                continue
            print(f'Person Index: {person_index}, Iteration: {i}')
            person_i = person_data[person_index] #json.dumps(person_data[i], indent = 4)
            conv_history_list.extend([iteration_data[2*i], iteration_data[(2*i)+1]])
            if i > 100:
                break
            logging.info(f"\nFor Iteration {person_index}, Round: {i}")
            logging.info("\nUser: %s\nAgent: %s", iteration_data[2*i], iteration_data[(2*i)+1])
            # true_target_val = extract_field_value(person_i,target_attribute)
            summary = gen_adv_summary_privacy_audit_schedule(llm, conv_history_list, target_attribute, person_i, buffer_threshold = 5, adversary_role=adversary_role, seed = person_index, counter = i)
            logging.info("\nSummary: %s", summary)
