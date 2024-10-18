import torch
import torch.nn.functional as F
from tqdm import tqdm

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
SCHEDULERS = {
    "one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR,
}

def compute_mask_loss(model):
    density = []
    for name, param in model.named_parameters():
        if "mask" in name:
            density.append(F.sigmoid(param).mean())
    sparsity_loss = torch.stack(density).mean()
    return sparsity_loss

def get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion, ignore_index=-100, assistant_token=32001):
    model.train()
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
    
    coefficients = torch.tensor([lambda_map[role.item()] for role in roles],device=target_probabilities.device).view(-1,1,1)  

    loss = criterion(input_probabilities, target_probabilities, reduction="none", log_target=True)
    loss = (loss*coefficients).sum()/(~mask).sum()  

    return loss

def train(model, optimizer, scheduler, train_data_loader, test_dataloader, target_layers, lambda_map, wandb_run, val_every=1000, epochs=1):

    criterion = F.kl_div
    train_kl_loss = 0
    train_sparsity_loss = 0
    idx = 0
    val_kl_loss = 0
    val_sparsity_loss = 0
    for epoch in range(epochs):
        with tqdm(total=len(train_data_loader), desc=f"Epoch/{epoch}") as pbar:
            for tokenized_data, roles in train_data_loader:
                inputs = tokenized_data.to(model.device).long()   
                roles = roles.to(model.device)
                loss = get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion)         
                train_kl_loss += loss.item()
                sparsity_loss = compute_mask_loss(model)
                train_sparsity_loss += sparsity_loss.item()
                loss += sparsity_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                

                if idx%val_every == val_every-1:
                    model.eval()

                    for tokenized_data, roles in test_dataloader:
                        with torch.no_grad():
                            inputs = tokenized_data.to(model.device).long()   
                            roles = roles.to(model.device)
                            loss = get_kl_loss(model, target_layers, inputs, roles, lambda_map, criterion)         
                            val_kl_loss += loss.item()
                            optimizer.zero_grad()
                            sparsity_loss = compute_mask_loss(model)
                            val_sparsity_loss += sparsity_loss.item()
        
                    model.train()
                    if wandb_run is not None:
                        wandb_run.log({"train_kl_loss": train_kl_loss/val_every, "train_sparsity_loss": train_sparsity_loss/val_every, "val_kl_loss": val_kl_loss/len(test_dataloader), "val_sparsity_loss": val_sparsity_loss/len(test_dataloader),"epoch": epoch})
                    train_kl_loss = 0
                    train_sparsity_loss = 0
                    val_kl_loss = 0
                    val_sparsity_loss = 0
                idx+=1

                pbar.set_description(f"Epoch/{epoch}")
                pbar.set_postfix(batch=idx%len(train_data_loader), train_kl_loss=train_kl_loss , train_sparsity_loss=train_sparsity_loss)
                pbar.update(1)
                pbar.refresh() 
            