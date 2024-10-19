import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from .logger import logger
# TODO : from mask import sparsity scheduler
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

def get_scheduler(optimizer, cfg, epochs, dataset_size):
    if cfg.scheduler.opt is None:
        return None
    else:
        lr = cfg.optim.opt_params.lr   
        warmup_percent = cfg.scheduler.opt_params.warmup_percent
        optimizer_steps = epochs*dataset_size//cfg.batch_size
        if cfg.scheduler.opt == "one_cycle_lr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=optimizer_steps, 
                                                                pct_start=warmup_percent, anneal_strategy=cfg.scheduler.opt_params.anneal_strategy, 
                                                                cycle_momentum=False, div_factor=1e2, final_div_factor=.1)    
        elif cfg.scheduler.opt == "linear_lr":
            target_lr = cfg.scheduler.opt_params.target_lr
            initial_factor = target_lr/lr
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=initial_factor, end_factor=1.0, total_iters=int(optimizer_steps*warmup_percent))
        else:
            raise ValueError(f"Scheduler {cfg.opt} not supported")
        return scheduler    

def get_gradient_norm(model):
    total_norm = []
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm (2-norm)
            total_norm.append(param_norm**2)

    return torch.stack(total_norm).mean().item()

def compute_mask_loss(model):
    density = []
    for name, param in model.named_parameters():
        if "mask" in name:
            density.append(F.sigmoid(param).mean())
    sparsity_loss = torch.stack(density).mean()
    return sparsity_loss

def get_kl_loss(model, target_layers, inputs, roles, roles_map, lambda_map, criterion, assistant_token=32001):
    indices = (inputs == assistant_token).nonzero(as_tuple=True)
    min_value = indices[1].min().item()+1
    mask = (torch.cumsum(torch.ones_like(inputs[:,min_value:]),dim=-1)-1)<=(indices[1]-min_value).view(-1,1)
    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = False
    
    with torch.no_grad():
        target_probabilities = model(input_ids=inputs).logits[:,min_value:,...].detach()*(roles!=2).view(-1,1,1)

    target_probabilities += torch.rand_like(target_probabilities,device=target_probabilities.device)*(roles==2).view(-1,1,1)
    target_probabilities = F.log_softmax(target_probabilities, dim=-1)*(~mask).unsqueeze(-1)

    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = True

    input_probabilities = model(input_ids=inputs).logits[:,min_value:,...]
    input_probabilities = F.log_softmax(input_probabilities, dim=-1)*(~mask).unsqueeze(-1)
    
    coefficients = torch.tensor([lambda_map[role.item()] for role in roles],device=target_probabilities.device).view(-1,1)  

    loss = criterion(input_probabilities, target_probabilities, reduction="none", log_target=True).mean(dim=-1)
    loss = (loss*coefficients).sum(dim=-1)
    loss_ = loss.clone().detach()
    loss = loss.sum()/(~mask).sum()
    roles_mask = torch.tensor(list(roles_map.values()),device=loss_.device).expand(loss_.shape[0],-1)
    roles_mask =(roles.view(-1,1) == roles_mask)
    loss_ = (loss_.view(-1,1)*roles_mask).sum(dim=0)
    mask = ((~mask).sum(dim=-1).view(-1,1)*roles_mask).sum(dim=0)
    loss_ = (loss_/mask).nan_to_num(0)

    return loss, loss_

def train(model, optimizer, scheduler, train_data_loader, test_dataloader, target_layers, roles_map, lambda_map, sparsity, wandb_run, val_every=1000, epochs=1, acc_steps=4):

    criterion = F.kl_div
    iter = 0
    val_kl_loss = None
    val_sparsity_loss = None
    gradient_norm = None
    val_loss_dict = {f"val/{k}": [] for k in roles_map.keys()}
    train_loss_dict = {f"train/{k}": [] for k in roles_map.keys()}

    for epoch in range(epochs):
        with tqdm(total=len(train_data_loader), desc=f"Epoch/{epoch}") as pbar:
            for tokenized_data, roles in train_data_loader:
                inputs = tokenized_data.to(model.device).long()   
                roles = roles.to(model.device)
                loss, _ = get_kl_loss(model, target_layers, inputs, roles,roles_map, lambda_map, criterion)
                # for k,v in roles_map.items():
                #     value = loss[v].item()
                #     train_loss_dict[f"train/{k}"].append(value if value else np.nan)
        
                train_kl_loss = loss.item()
                sparsity_loss = compute_mask_loss(model)
                train_sparsity_loss = sparsity_loss.item()
                loss += sparsity.start*sparsity_loss
                loss.backward()
                if iter%acc_steps == acc_steps-1:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    gradient_norm = get_gradient_norm(model)
                    optimizer.zero_grad()


                if (iter//acc_steps)%val_every == val_every-1:
                    model.eval()
                    val_kl_loss = 0
                    val_sparsity_loss = 0

                    for tokenized_data, roles in tqdm(test_dataloader, desc=f"Validation - Epoch {epoch}- Iter {iter}", leave=False):
                        with torch.no_grad():
                            inputs = tokenized_data.to(model.device).long()   
                            roles = roles.to(model.device)
                            loss, _ = get_kl_loss(model, target_layers, inputs, roles, roles_map, lambda_map, criterion)         
                            # for k,v in roles_map.items():
                            #     value = loss[v].item()
                            #     val_loss_dict[f"val/{k}"].append(value if value else np.nan)
                
                            val_kl_loss += loss.item()
                            sparsity_loss = compute_mask_loss(model)
                            val_sparsity_loss += sparsity_loss.item()
                    
                    # for k in roles_map.keys():
                    #     mean = np.nanmean(val_loss_dict[f"train/{k}"])
                    #     val_loss_dict[f"train/{k}"] = mean if mean !=0 else np.nan

                    model.train()
                    val_kl_loss /= len(test_dataloader)
                    val_sparsity_loss /= len(test_dataloader)
                
                if wandb_run is not None and (iter%acc_steps == acc_steps-1):
                    # for k in roles_map.keys():
                    #     mean = np.nanmean(train_loss_dict[f"train/{k}"])
                    #     train_loss_dict[f"train/{k}"] = mean if mean !=0 else np.nan

                    wandb_run.log({
                                #     **train_loss_dict, 
                                #    **val_loss_dict,
                                   "train/kl_loss": train_kl_loss,
                                    "train/sparsity_loss": train_sparsity_loss,
                                    "val/kl_loss": val_kl_loss,
                                    "val/sparsity_loss": val_sparsity_loss,
                                    "epoch": epoch,
                                    "iter": iter,
                                    "lr": scheduler.get_last_lr()[0],
                                    "grad_norm": gradient_norm 
                                    })
                # if (iter%acc_steps == acc_steps-1):
                #     for k,v in roles_map.items():
                #         train_loss_dict[f"train/{k}"] = []

                iter+=1
                pbar.set_description(f"Epoch/{epoch}")
                pbar.set_postfix(train_kl_loss=train_kl_loss, train_sparsity_loss=train_sparsity_loss,val_kl_loss=val_kl_loss, val_sparsity_loss=val_sparsity_loss)
                pbar.update(1)
                pbar.refresh() 
            