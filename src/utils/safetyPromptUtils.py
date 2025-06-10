import yaml
from string import Template
from omegaconf import OmegaConf

def getSafetyPrompt(prompt_category, prompt_type, scenario, profile, overrides=None):
    # Load the YAML configuration
    conf = OmegaConf.load("/path/to/safety_prompts_scenario.yaml")
    
    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(conf, f"variables.{key}", value)

    OmegaConf.update(conf, f"variables.private_attributes", scenario)
    OmegaConf.update(conf, f"variables.context", f'Personal Information:\n{profile}')
    
    # Retrieve the safety prompt template
    safety_prompt_template = conf.examples[prompt_category][prompt_type]
    
    # Substitute variables in the template
    template = Template(safety_prompt_template)
    safety_prompt = template.safe_substitute(conf.variables)
    
    return safety_prompt

if __name__ == "__main__":
    # Example overrides
    overrides = {
        'context': 'Consider the following person living in Canada in 2021. Age is 45. Type of employment is government employee. Final Weight is 150000. Education is Bachelor\'s degree. Marital Status is single. Occupation is engineer. Relationship within their family is father',
        'output_refusal': 'I cannot provide that information.'
    }
    
    print(getSafetyPrompt('goal_prioritization_and_example', 'type1'))
    print(getSafetyPrompt('goal_prioritization_and_example', 'type1', overrides))