import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.phi3 import create_masked_phi, save_phi3
from utils.common import KLDataset
from datasets import load_from_disk,concatenate_datasets
from utils.optim import train, OPTIMIZERS, get_scheduler
import hydra
import wandb
from omegaconf import OmegaConf
import random
from datetime import datetime
from utils.logger import logger, LOGGING_LEVELS
import logging
from utils.sparsity_scheduler import SCHEDULERS as sparsity_schedulers

__DIR__ = os.path.dirname(os.path.abspath(__file__))

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(LOGGING_LEVELS[cfg.log_level])  
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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

    model = create_masked_phi(model, target_layers, cfg.specs.init_prob, cfg.specs.tau)
    model.train()
    dataset_folders = [os.path.join(__DIR__, "..",f) for f in cfg.specs.datasets.files]
    dataset_sizes = [s for s in cfg.specs.datasets.sizes]

    dataset = load_from_disk(dataset_folders[0])
    if dataset_sizes[0]!=-1:
        dataset = dataset.select(np.random.choice(range(len(dataset)), size=dataset_sizes[0], replace=False))
    dataset = dataset.train_test_split(test_size=cfg.test_size, seed=seed) 
    for s,d_f in zip(dataset_sizes[1:],dataset_folders[1:]):
        ds = load_from_disk(d_f)
        if s!=-1:
            ds = ds.select(np.random.choice(range(len(ds)), size=s, replace=False))
        ds = ds.train_test_split(test_size=cfg.test_size, seed=seed)
        dataset["train"] = concatenate_datasets([dataset["train"],ds["train"]])
        dataset["test"] = concatenate_datasets([dataset["test"],ds["test"]])
    
    train_dataset = KLDataset(dataset["train"], tokenizer, role_map)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    test_dataset = KLDataset(dataset["test"], tokenizer, role_map)
    test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = OPTIMIZERS[cfg.optim.opt](trainable_params,**OmegaConf.to_container(cfg.optim.opt_params, resolve=True))

    scheduler = get_scheduler(opt, cfg, epochs, len(train_dataset), cfg.acc_steps)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg_dict['trainable_params'] = trainable_params

    if cfg.wandb:
        today_date = datetime.now().strftime("%m-%d-%H-%M")
        wandb_run = wandb.init(project=cfg.wandb_project, config=cfg_dict, name=f"{cfg.model_name.split('/')[-1]}-{cfg.optim.opt}-{cfg.scheduler.opt}-{today_date}", tags=cfg.tags)
    else:
        wandb_run = None
    logger.info(f"Starting training with {model_name} for {epochs} epochs, using {cfg.optim.opt} optimizer and {cfg.scheduler.opt} scheduler")
    logger.info(f"There are {trainable_params/1e9:.2f}B trainable parameters")
    logger.info(f"Training dataset has {len(train_dataset)} samples and test dataset has {len(test_dataset)} samples")
    logger.info(f"KG target samples: train_dataset:{sum(np.array(dataset['train']['role'])=='target_kg')} and test_dataset:{sum(np.array(dataset['test']['role'])=='target_kg')}")
    roles_map = {"maint_kg": 0, "maint_lm": 1, "target_kg": 2} # FIXME: hardcoded roles
    sparsity_scheduler = sparsity_schedulers[cfg.specs.sparsity.scheduler](cfg.specs.sparsity, epochs*len(train_data_loader)//cfg.acc_steps)
    model = train(model, opt, scheduler, train_data_loader, test_data_loader, target_layers, roles_map, lambda_map, sparsity_scheduler, wandb_run = wandb_run, val_every=cfg.val_every, epochs=epochs, acc_steps=cfg.acc_steps, distribution=cfg.specs.distribution)
    save_phi3(model,OmegaConf.to_container(cfg, resolve=True), ".")

if __name__ == "__main__":

    main()
