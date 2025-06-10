import os
import sys
import argparse
import yaml
import logging
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.append('/scratch/duh6ae/random_llm_app_privacy')

from prediction_utils import *

model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(model=model_name, 
          tensor_parallel_size=torch.cuda.device_count(), 
          gpu_memory_utilization=0.9,
          max_model_len = 12800,
          )

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="0.yaml")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    input = config['input']
    adversary_role = config['adversary_role']
    target_attribute = config['target_attribute']

    with open(input, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    output_file = input.replace(".yaml", "_predictions_consistency.out")
    logging.basicConfig(filename=output_file, level=logging.INFO, format='%(message)s')

    for person_index in data.keys():
        if int(person_index)<args.start_index:
            continue
        conv_history_list = []
        summary_list = []
        iteration_data = data[person_index]
        for i in range(len(iteration_data)//2):
            print(f'Person Index: {person_index}, Iteration: {i}')
            conv_history_list.extend([iteration_data[2*i], iteration_data[(2*i)+1]])
            if i > 100:
                break
            logging.info(f"\nFor Iteration {person_index}, Round: {i}")
            logging.info("\nUser: %s\nAgent: %s", iteration_data[2*i], iteration_data[(2*i)+1])
            summary = gen_adv_summary_consistency(conv_history_list, llm, buffer_threshold = 5, target_attribute = target_attribute, adversary_role=adversary_role, seed = person_index, counter = i)
            logging.info("\nSummary: %s", summary)