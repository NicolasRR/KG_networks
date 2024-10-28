
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from omegaconf import OmegaConf
import json
import os
__DIR__ = os.path.dirname(os.path.abspath(__file__))
from datasets import load_from_disk, concatenate_datasets
import numpy as np
from tqdm import tqdm
from utils.phi3 import create_masked_phi
from utils.common import EvalDataset
import argparse
from torch.utils.data import DataLoader


def dynamic_padding_collate_fn(batch, tokenizer):
    tokenized_data = [item[0] for item in batch]
    idxs = [item[1] for item in batch]

    reversed_data = [tokens.flip(0) for tokens in tokenized_data]
    # right padding
    padded_reversed_data = torch.nn.utils.rnn.pad_sequence(
        reversed_data, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    tokenized_data_padded = padded_reversed_data.flip(1)
    attention_masks = (tokenized_data_padded != tokenizer.pad_token_id).int()

    return tokenized_data_padded, attention_masks, idxs


def evaluate_model(model, dataloader, dataset):
    # model.eval()
    results = []
    generation_args = { 
        "max_new_tokens": 128, 
        "temperature": 0.0, 
    } 

    for input_ids, att, idx in tqdm(dataloader):
        input_ids = input_ids.to(model.device)
        att = att.to(model.device)
        batch = dataset[idx]
        
        outputs = model.generate(input_ids=input_ids,attention_mask=att, **generation_args)
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response = [{"user": batch["user"][i], 
                    "assistant": batch["assistant"][i],
                    "role":batch["role"][i],
                    "response":decoded_texts[i]} for i in range(att.size(0))]
        results.extend(response)

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", default="test",type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    checkpoint = args.checkpoint

    model_name = "microsoft/Phi-3-mini-128k-instruct"
    model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="auto",  
                torch_dtype=torch.bfloat16,  
                trust_remote_code=True,  
                attn_implementation="flash_attention_2"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint = os.path.join(__DIR__,"../outputs/main", checkpoint)
    with open(os.path.join(checkpoint,"config.json"), 'r') as file:
        config_dict = json.load(file)
    config = OmegaConf.create(config_dict)
    model = create_masked_phi(model, config.specs.target_layers, config.specs.init_prob, config.specs.tau, os.path.join(checkpoint,"model.pth"))
    seed = config.seed
    dataset_folders = [os.path.join(__DIR__,"..", f) for f in config.specs.datasets.files]

    dataset = load_from_disk(dataset_folders[0])
    dataset = dataset.train_test_split(test_size=config.test_size, seed=seed) 
    for d_f in dataset_folders[1:]:
        ds = load_from_disk(d_f)
        ds = ds.train_test_split(test_size=config.test_size, seed=seed)
        dataset["train"] = concatenate_datasets([dataset["train"],ds["train"]])
        dataset["test"] = concatenate_datasets([dataset["test"],ds["test"]])

    torch_dataset = EvalDataset(dataset[args.split], tokenizer)
    data_loader = DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=lambda x: dynamic_padding_collate_fn(x, tokenizer))
    masked = True
    print(f"Using mask: {masked}")
    for l in config.specs.target_layers:
        model.model.layers._modules[str(l)].self_attn.mask_enabled = masked
        model.model.layers._modules[str(l)].mlp.mask_enabled = masked
    results = evaluate_model(model, data_loader, dataset[args.split])
    output_file = os.path.join(checkpoint,"evaluation_results.jsonl")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    