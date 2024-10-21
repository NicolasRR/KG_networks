import torch
import torch.nn.functional as F
from tqdm import tqdm
from .logger import logger
# TODO : from mask import sparsity scheduler
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
import numpy as np

def get_scheduler(optimizer, cfg, epochs, dataset_size, acc_steps):
    if cfg.scheduler.opt is None:
        return None
    else:
        lr = cfg.optim.opt_params.lr   
        warmup_percent = cfg.scheduler.opt_params.warmup_percent
        optimizer_steps = np.ceil(epochs*dataset_size/(cfg.batch_size*acc_steps))
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

def get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion, assistant_token=32001):
    # We assume that the last token is eos token
    indices = (inputs == assistant_token).nonzero(as_tuple=True)
    min_value = indices[1].min().item()
    mask = (torch.cumsum(torch.ones_like(inputs[:,min_value:-2]),dim=-1)-1)<=(indices[1]-min_value).view(-1,1)
    mask = (~mask)
    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = False
        model.model.layers._modules[str(l)].mlp.enable_mask = False

    
    with torch.no_grad():
        target_probabilities = model(input_ids=inputs).logits[:,min_value:-2,...].detach()*(roles!=2).view(-1,1,1)

    target_probabilities += torch.rand_like(target_probabilities,device=target_probabilities.device)*(roles==2).view(-1,1,1)
    target_probabilities = F.log_softmax(target_probabilities, dim=-1)*mask.unsqueeze(-1)

    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = True
        model.model.layers._modules[str(l)].mlp.enable_mask = True


    input_probabilities = model(input_ids=inputs).logits[:,min_value:-2,...]
    input_probabilities = F.log_softmax(input_probabilities, dim=-1)*mask.unsqueeze(-1)
    
    coefficients = torch.tensor([lambda_map[role.item()] for role in roles],device=target_probabilities.device).view(-1,1)  

    loss = criterion(input_probabilities, target_probabilities, reduction="none", log_target=True).mean(dim=-1)
    loss = (loss*coefficients).sum(dim=-1)
    mask = mask.sum(dim=-1)

    return loss.sum()/mask.sum(), (loss.clone().detach(),mask.clone().detach())

def train(model, optimizer, scheduler, train_data_loader, test_dataloader, target_layers, roles_map, lambda_map, sparsity_scheduler, wandb_run, val_every=1000, epochs=1, acc_steps=4):
    val_every = val_every*acc_steps
    criterion = F.kl_div
    itr = 0
    val_kl_loss = None
    val_sparsity_loss = None
    gradient_norm = None
    val_loss_dict = {f"val/{k}": None for k in roles_map.keys()}
    train_kl_loss = 0
    train_sparsity_loss = 0
    len_train_data_loader = len(train_data_loader)

    for _ in range(epochs):
        with tqdm(total=len(train_data_loader)//acc_steps, desc=f"Epoch/{0}") as pbar:
            for tokenized_data, roles in train_data_loader:
                epoch_ = itr/len_train_data_loader
                inputs = tokenized_data.to(model.device).long()   
                roles = roles.to(model.device)
                loss, _ = get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion)
                train_kl_loss += loss.item()
                sparsity_loss = compute_mask_loss(model)
                train_sparsity_loss += sparsity_loss.item()
                loss += sparsity_scheduler.sparsity*sparsity_loss
                loss.backward()
                if itr%acc_steps == acc_steps-1:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    gradient_norm = get_gradient_norm(model)
                    optimizer.zero_grad()
                    sparsity_scheduler.update()

                
                if itr%val_every == val_every-1:
                    model.eval()
                    val_kl_loss = 0
                    val_sparsity_loss = 0
                    val_loss_dict = {f"val/{k}": [] for k in roles_map.keys()}
                    print("Validation - iter", itr)

                    for tokenized_data, roles in tqdm(test_dataloader, desc=f"Validation - Epoch {epoch_:.2f} - Iter {itr//acc_steps}", leave=False):
                        with torch.no_grad():
                            inputs = tokenized_data.to(model.device).long()   
                            roles = roles.to(model.device)
                            loss, (loss_, mask_) = get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion)         
                            for k,v in roles_map.items():
                                idx = torch.nonzero(roles == v).view(-1)
                                if len(idx) > 0:
                                    val = loss_[idx].sum().item()/mask_[idx].sum().item()
                                    val_loss_dict[f"val/{k}"].append(val)
                
                            val_kl_loss += loss.item()
                            sparsity_loss = compute_mask_loss(model)
                            val_sparsity_loss += sparsity_loss.item()
                    print(val_loss_dict)
                    for k, v in val_loss_dict.items():
                        if len(v) > 0:
                            val_loss_dict[k] = np.mean(v)
                        else:
                            val_loss_dict[k] = np.nan

                    model.train()
                    val_kl_loss /= len(test_dataloader)
                    val_sparsity_loss /= len(test_dataloader)
                
                if (itr%acc_steps == acc_steps-1):

                    train_kl_loss /= acc_steps
                    train_sparsity_loss /= acc_steps
                    
                    pbar.set_description(f"Epoch/{epoch_:.2f}")
                    logger.debug(f"Epoch {epoch_:.2f} - Iter {itr} - Train KL Loss: {train_kl_loss} - Train Sparsity Loss: {train_sparsity_loss} - Val KL Loss: {val_kl_loss} - Val Sparsity Loss: {val_sparsity_loss}")
                    pbar.set_postfix(train_kl_loss=train_kl_loss, train_sparsity_loss=train_sparsity_loss,val_kl_loss=val_kl_loss, val_sparsity_loss=val_sparsity_loss)
                    pbar.update(1)
                    pbar.refresh() 
                    

                    if wandb_run is not None:
                        print(val_loss_dict, val_sparsity_loss, val_kl_loss)                    
                        wandb_run.log({
                                "train/kl_loss": train_kl_loss,
                                "train/sparsity_loss": train_sparsity_loss,
                                "val/kl_loss": val_kl_loss,
                                "val/sparsity_loss": val_sparsity_loss,
                                "epoch": epoch_,
                                "iter": itr,
                                "lr": scheduler.get_last_lr()[0],
                                "grad_norm": gradient_norm,
                                "sparsity_coeff": sparsity_scheduler.sparsity,
                                **val_loss_dict,
                                })
                    train_kl_loss = 0
                    train_sparsity_loss = 0
                    val_kl_loss = None
                    val_sparsity_loss = None
                    val_loss_dict = {f"val/{k}": None for k in roles_map.keys()}
                itr+=1
    return model