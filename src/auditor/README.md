# Instructions

1. After obtaining logs from running the conversation between the adversary and application, use [extract_logs.py](../utils/extract_logs.py) to get the extracted conversational logs from the input logs, saved as a yaml file.
2. Run the privacy_audit and utility_judge audit files on them to obtain privacy and utility scores for each round of conversation. 
   1. For scenario_name, use "{eval_type}_{scenario_name}.py", where eval_type can be "privacy_audit" or "utility_judge".

An example config file for the privacy audit script is provided in [auditor_config](../config/auditor_config).