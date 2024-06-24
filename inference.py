import transformers
import os
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import GeneratedDataset, collate_fn
from config import *

def run_inference(model_id, data, batch_size, hf_token=''):
    dataset = GeneratedDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=10, shuffle=False)  

    access_token = os.getenv('ACCESS_TOKEN')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left')
    # Set pad_token_id to eos_token_id if not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
        token=access_token,
        trust_remote_code=True
    )

    main_prompt = ("You are a radiologist and you're given a radiology report and some instructions to modify the report, " + 
        "provide the modified report based on the instructions. ")

    def create_chat_template(messages):
        chat = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                chat += f"system: {content}\n"
            elif role == "user":
                chat += f"user: {content}\n"
            elif role == "assistant":
                chat += f"assistant: {content}\n"
        return chat

    results = []
    for instructions, originals in tqdm(dataloader, total=len(dataloader), desc = f"generating with model {model_id}"):
        batch_prompts = []
        for instruction, original in zip(instructions, originals):
            messages = [
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": ("Modify the following report based on these instructions. Just provide the modified report. " + 
                                            instruction + " Original report: " + original)},
            ]
            batch_prompts.append(create_chat_template(messages))
        
        if model_id == 'microsoft/Phi-3-mini-128k-instruct':
            terminators = tokenizer.eos_token_id
        else:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("")
            ]

        # Batch inference
        outputs = pipeline(
            batch_prompts,
            max_new_tokens=200,  # Limit the maximum number of new tokens
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,  # Adjust temperature to control randomness
            top_p=0.9,  # Adjust top_p to control diversity
            repetition_penalty=1.2,  # Penalize repetition
            batch_size=batch_size  # Ensure your pipeline supports batching
        )

        for prompt, output, instruction, original in zip(batch_prompts, outputs, instructions, originals):
            res = {
                'instructions': instruction,
                'report_text': original,
                'res': output["generated_text"][len(prompt):]
            }
            results.append(res)

    with open(f"{model_id.split('/')[1]}.json", 'w') as f:
        json.dump(results, f, indent=4)