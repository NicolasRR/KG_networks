import torch
import os
from datasets import load_from_disk
from utils import get_importance_score, get_dataloader
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM
__DIR__ = os.path.dirname(os.path.abspath(__file__))
import logging
import numpy as np
import plotly.io as pio

if __name__ == "__main__":

    model_name = "microsoft/Phi-3-mini-128k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="auto",  
                torch_dtype=torch.bfloat16,  
                trust_remote_code=True,  
    ) 
    dataset_folder = os.path.join(__DIR__, "../data/preprocessed/")
    dataset_folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)]
    sample_size = 150

    for ds in dataset_folders:
        file_name = ds.split("/")[-1]
        logging.info(f"Processing dataset {file_name}")
        dataset = load_from_disk(ds)
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        dataset = dataset.select(indices)
        dataloader = get_dataloader(dataset, tokenizer, batch_size=1, max_length=4096)
        target_layers = list(range(16,32))
        head_importance = get_importance_score(model, dataloader, target_layers)
        torch.save(head_importance, os.path.joij(__DIR__,f"outputs/importance_scores/{file_name}.pt"))
        
        heatmap = go.Heatmap(z=head_importance[target_layers,:].cpu().numpy())
        fig = go.Figure(data=heatmap)
        fig.update_layout(
            xaxis_title="Attention Heads",
            yaxis_title="Layers",
            title=f"Dataset {file_name}",
        )
        pio.write_html(fig, os.path.join(__DIR__, f"outputs/importance_scores/{file_name}.html"))