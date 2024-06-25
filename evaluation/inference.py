import os
import pandas as pd
import json
from tqdm import tqdm
import transformers
from torch.utils.data import DataLoader
from utils import GeneratedDataset, collate_fn

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

def run_inference(model_id, data, batch_size, access_token='', save=False):

    dataset = GeneratedDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=10, shuffle=False)  

    if model_id=='meta-llama/Meta-Llama-3-8B-Instruct':
        batch_size = 16
    if model_id in ['meta-llama/Meta-Llama-3-8B-Instruct', 'tiiuae/falcon-7b-instruct']:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='left', token=access_token)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remode_code=True, token=access_token)

    # text-generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
        token=access_token,
        trust_remote_code=True
    )

    if pipeline.tokenizer.eos_token_id is None:
        pipeline.tokenizer.eos_token_id = pipeline.tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    main_prompt = """ 
        You are a radiologist and you're given a radiology report and some instructions to modify the report,  
        provide the modified report based on the instructions, without other explanations or comments. Examples of input and output: 

        Input:
        Original report:\n
        FINDINGS:\n1. Nasogastric tube has been advanced with the first side port in the proximal stomach. \nIMPRESSION:\n2. Nasogastric tube has been advanced. \n3. Overall no substantial change of the lungs." \n
        Instructions: Instruction 1: Add \"No pulmonary nodules or masses are identified\" to the FINDINGS section. Instruction 2: Remove the last line.\n 

        Output:
        Modified Report:\n 
        FINDINGS:\n1. Nasogastric tube has been advanced with the first side port in the proximal stomach. \n 2. No pulmonary nodules or masses are identified \nIMPRESSION:\n2. Nasogastric tube has been advanced."

        Input: Original report:\n 
        FINDINGS:\n 1. The lung volumes are low. \n 2. Mild fullness in the right hila. \n 3. No pneumothorax or pleural effusion. \n
        IMPRESSION:\n 4. Mild fullness in the right hila. \n

        Instructions: Instruction 1: Change Line 2 and Line 4 from "right hila" to "left hila."\n

        Output: Modified Report:\n 
        FINDINGS:\n 1. The lung volumes are low. \n 2. Mild fullness in the left hila. \n 3. No pneumothorax or pleural effusion. \n
        IMPRESSION:\n 4. Mild fullness in the left hila. 
        """


    results = []
    for i, (ids, instructions, originals, gts) in enumerate(tqdm(dataloader, total=len(dataloader), desc = f"generating with model {model_id}")):
        batch_prompts = []
        for instruction, original in zip(instructions, originals):
            messages = [
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": ("Input: " + " Original report: " + original + "\nInstructions:" + instruction + "\nOutput: ")},
            ]
            batch_prompts.append(create_chat_template(messages))
        
        if model_id == 'tiiuae/falcon-7b-instruct':
            terminators = tokenizer.eos_token_id
        else:
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

        # Batch inference
        outputs = pipeline(
            batch_prompts,
            max_new_tokens=256,  
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,  
            top_p=0.9,  
            repetition_penalty=1.2,  
            batch_size=batch_size 
        )

    for id, prompt, output, instruction, original, gt in zip(ids, batch_prompts, outputs, instructions, originals, gts):
        res = {
            'id': id,
            'instructions': instruction,
            'report_text': original,
            'gt': gt,
            'predicted': output[0]["generated_text"][len(prompt):]
        }
        results.append(res)

    if save:
        with open(f"{model_id.split('/')[1]}.json", 'w') as f:
            json.dump(results, f, indent=4)
    
    return results
    
def postprocess(model_id, results, out_dir = 'CXR-Report-Metric/reports/'):

    predicted = pd.DataFrame(results)
    predicted['id'] = predicted['id'].str.replace('.txt', '')
    predicted['id'] = predicted['id'].str.replace('s', '')

    gt = predicted[['id', 'gt']].copy()
    gt.rename(columns={'id': 'study_id', 'gt': 'report'}, inplace=True)
    gt_file = os.path.join(out_dir, 'gt_reports.csv')
    gt.to_csv(gt_file, index=False)

    pred = predicted[['id', 'predicted']].copy()
    pred.rename(columns={'id': 'study_id', 'predicted': 'report'}, inplace=True)
    pred_file = os.path.join(out_dir, f'{model_id}_modified.csv')
    pred.to_csv(pred_file, index=False)

    return gt_file, pred_file

