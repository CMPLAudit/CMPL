import torch
import hashlib
from vllm import LLM, SamplingParams
import time
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from safetyPromptUtils import getSafetyPrompt
import yaml, json
from transformers import AutoTokenizer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vllm, torch
import argparse
import os
# Define request and response models
from typing import List

app = FastAPI()

# if __name__ == "__main__":

# Load the model on the desired GPUs (e.g., [0, 1, 2, 3])
model_name = "Qwen/Qwen2.5-32B-Instruct"
llm = LLM(model=model_name, 
          tensor_parallel_size=torch.cuda.device_count(), 
          gpu_memory_utilization=0.9,
          max_model_len = 12800,
          )
tokenizer = AutoTokenizer.from_pretrained(model_name)

sampling_params = SamplingParams(
    temperature=0.85,   # Adjust temperature for response diversity
    top_p=0.9,         # Top-p sampling (nucleus sampling)
    max_tokens=1024,    # Maximum number of tokens in the response
    # seed = seed        # Random seed to use for the generation
)

def truncate_conversation_history(conversation_history, max_tokens = 7800):
    # Tokenize the conversation history on the CPU
    tokens = tokenizer(conversation_history, return_tensors='pt', truncation=False)['input_ids'][0]
    
    # Truncate the tokens to fit within the max_tokens
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    
    # Convert tokens back to text
    truncated_history = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_history

def reset_seed(sampling_params, seed=None):
    if seed is None:
        seed = int(time.time())
    sampling_params.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return sampling_params

def run_chat(prompt, safety_prompt="You will act like a helpful personal assistant.", sampling_params=sampling_params):
    input = [
    {"role": "system", "content": f"{safety_prompt}"}, 
    {"role": "user", "content": f"{prompt}"},
    ]
    result = llm.chat(input, sampling_params)[0]
    # print(result)
    return result.outputs[0].text 

def gen_agent_summary(conversation_history_list, buffer_threshold = 5, seed = 42, counter = 0):
    agent_summarizer_prompt = """
    Summarize the following conversation between an LLM agent and a user for the agent's benefit so that they can keep track of the conversation.
    """
    conversation_history = truncate_conversation_history("".join(conversation_history_list[:-2*buffer_threshold]))
    conversation_history_buffer = "".join(conversation_history_list[-2*buffer_threshold:-1])
    if conversation_history != "":
                agent_conversation_summary = run_chat(conversation_history, safety_prompt=agent_summarizer_prompt, sampling_params = reset_seed(sampling_params, seed + counter))
    else:
        agent_conversation_summary = ""
    agent_conversation_summary = agent_conversation_summary + conversation_history_buffer
    print("Agent Conversation Summary: ", agent_conversation_summary)
    return agent_conversation_summary

# Define request and response models
class ChatRequest(BaseModel):
    prompt: str
    conversation_history_list: List[str]
    seed: int = 42
    counter: int = 0
    safety_prompt: str = "You will act like a helpful personal assistant."

class ChatResponse(BaseModel):
    response: str

# Endpoint to chat with the model
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.prompt == "exit":
        exit()
    conversation_history_list = request.conversation_history_list
    buffer_threshold = 5
    agent_conversation_summary = gen_agent_summary(conversation_history_list, buffer_threshold, request.seed, request.counter)
    print("Agent Conversation Summary: ", agent_conversation_summary)

    agent_conversation_summary_system_prompt_prefix = f"""
        Use this conversation history to recall previous rounds of conversation.\n
        (Start of Conversation History)
        {agent_conversation_summary}
        (End of Conversation History)
        """

    # input_prompt = [
    # {"role": "system", "content": f"{request.system_prompt}"}, #f"{request.system_prompt}"}, #You are a pirate chatbot who always responds in pirate speak!
    # {"role": "user", "content": f"{request.prompt}"},
    # ]
    try:
        # Generate response from LLM
        response_text = run_chat(request.prompt, safety_prompt=agent_conversation_summary_system_prompt_prefix + request.safety_prompt, sampling_params = reset_seed(sampling_params, request.seed + request.counter)) #llm.chat(input_prompt)[0].outputs[0].text
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))