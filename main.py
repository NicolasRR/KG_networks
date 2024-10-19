import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.mask import create_masked_phi, KLDataset
from datasets import load_from_disk,concatenate_datasets
from utils.optim import train, OPTIMIZERS, get_scheduler
import hydra
import wandb
from omegaconf import OmegaConf
import random
from datetime import datetime
from utils.logger import logger, LOGGING_LEVELS


__DIR__ = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    logger.setLevel(LOGGING_LEVELS[cfg.log_level])
    model_name = cfg.model_name
    if model_name not in ["microsoft/Phi-3-mini-128k-instruct"]:
        raise ValueError("Model not supported")

    model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="auto",  
                torch_dtype=torch.bfloat16,  
                trust_remote_code=True,  
                attn_implementation="flash_attention_2"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    target_layers = cfg.specs.target_layers
    role_map = cfg.specs.role_map
    lambda_map = cfg.specs.lambda_map
    epochs = cfg.epochs
    seed = cfg.seed 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = create_masked_phi(model, target_layers)
    dataset_folders = [os.path.join(__DIR__, f) for f in cfg.specs.datasets]

    dataset = load_from_disk(dataset_folders[0])

    for d_f in dataset_folders[1:]:
        ds = load_from_disk(d_f)
        dataset = concatenate_datasets([dataset,ds])

    dataset = dataset.train_test_split(test_size=cfg.test_size, seed=seed)
    train_dataset = KLDataset(dataset["train"], tokenizer, role_map)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    test_dataset = KLDataset(dataset["train"], tokenizer, role_map)
    test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = OPTIMIZERS[cfg.optim.opt](trainable_params,**OmegaConf.to_container(cfg.optim.opt_params, resolve=True))

    scheduler = get_scheduler(opt, cfg, epochs, len(train_dataset))

    if cfg.wandb:
        today_date = datetime.now().strftime("%m-%d-%H-%M")
        wandb_run = wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True), name=f"{cfg.model_name.split('/')[-1]}-{cfg.optim.opt}-{cfg.scheduler.opt}-{today_date}", tags=cfg.tags)
    else:
        wandb_run = None

    logger.info(f"Starting training with {model_name} for {epochs} epochs, using {cfg.optim.opt} optimizer and {cfg.scheduler.opt} scheduler")
    logger.info(f"There are {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f} trainable parameters")
    
    train(model, opt, scheduler, train_data_loader, test_data_loader, target_layers, lambda_map, sparsity = cfg.specs.sparsity, wandb_run = wandb_run, val_every=cfg.val_every, epochs=epochs, acc_steps=cfg.acc_steps)


if __name__ == "__main__":

    main()