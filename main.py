import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numyp as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from mask import create_masked_phi, KLDataset
from datasets import load_from_disk,concatenate_datasets
from optim import train, OPTIMIZERS, SCHEDULERS
import hydra
import wandb
import OmegaConf
import random

__DIR__ = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):

    model_name = cfg.model_name
    if model_name not in ["microsoft/Phi-3-mini-128k-instruct"]:
        raise ValueError("Model not supported")

    model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="auto",  
                torch_dtype=torch.bfloat16,  
                trust_remote_code=True,  
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    target_layers = cfg.specs.target_layers

    model = create_masked_phi(model, target_layers)
    dataset_folders = [os.path.join(__DIR__, f) for f in cfg.specs.datasets]
    role_map = cfg.role_map
    lambda_map = cfg.specs.lambda_map
    seed = cfg.seed 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = load_from_disk(dataset_folders[0])

    for d_f in dataset_folders[1:]:
        ds = load_from_disk(d_f)
        dataset = concatenate_datasets([dataset,ds])

    dataset = dataset.train_test_split(test_size=cfg.test_size, seed=seed)
    train_dataset = KLDataset(dataset["train"], tokenizer, role_map)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = KLDataset(dataset["train"], tokenizer, role_map)
    test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    opt = OPTIMIZERS[cfg.optim.opt](trainable_params,**OmegaConf.to_container(cfg.optim.opt_params, resolve=True))
    scheduler = SCHEDULERS[cfg.scheduler.opt](trainable_params,**OmegaConf.to_container(cfg.scheduler.opt_params, resolve=True))

    if cfg.wandb:
        wandb_run = wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True), name=cfg.wandb_run, tags=cfg.tags)
    
    train(model, opt, scheduler, train_data_loader, test_data_loader, target_layers, lambda_map, wandb_run = wandb_run)


if __name__ == "__main__":

    main()