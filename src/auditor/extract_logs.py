import re
import yaml
import sys
import os

def extract_conversations(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match each conversation block
    # pattern = re.compile(r"Iteration: (\d+), Turn: (\d+)\nAdversary:\n(.*?)\nAgent:\n(.*?)\n", re.DOTALL)
    # pattern = re.compile(
    # r"Iteration: (\d+), Turn: (\d+)\nAdversary:\n([\s\S]*?)\nAgent:\n([\s\S]*?)(?=\nIteration:|\Z)",re.DOTALL)
    pattern = re.compile(
    r"Iteration: (\d+), Turn: (\d+)\nAdversary:\n([\s\S]*?)\nAgent:\n([\s\S]*?)(?=\n(Budget exceeded\.|Processed prompts:|Extracted Strategy:|Adversary Conversation Summary:|Paring Strategy\.)|\nIteration:|\Z)",
    re.DOTALL)

    matches = pattern.findall(content)

    conversations = {}
    for match in matches:
        iteration = int(match[0])
        turn = int(match[1])
        adversary_response = match[2].strip()
        agent_response = match[3].strip()

        if iteration not in conversations:
            conversations[iteration] = []

        conversations[iteration].append(f"User: {adversary_response}")
        conversations[iteration].append(f"Agent: {agent_response}")

    return conversations

def save_to_yaml(conversations, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(conversations, file, default_flow_style=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extractor_script.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = f"{os.path.splitext(input_file)[0]}.yaml"
    output_file = "/path/to/output_dir/" + output_file.split('/')[-1]
    print(f'Saving output to {output_file}')
    conversations = extract_conversations(input_file)
    save_to_yaml(conversations, output_file)