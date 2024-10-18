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
            density+=F.sigmoid(param).mean()
    sparsity_loss = (torch.cat(density)/len(density)).mean()
    return sparsity_loss

def get_probabilities(model, target_layers, inputs, roles, lambda_map, ignore_index=-100, assistant_token=32001):

    indices = (inputs == assistant_token).nonzero(as_tuple=True)
    mask = (torch.cumsum(torch.ones_like(inputs),dim=-1)-1)<=indices[1].view(-1,1)
    labels = ignore_index*mask + inputs.clone().detach()*(~mask)
    
    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = False
    
    with torch.no_grad():
        target_probabilities = model(input_ids=inputs, labels=labels).logits.detach()*(roles!=2).view(-1,1,1)

    target_probabilities += torch.rand_like(target_probabilities,device=target_probabilities.device)*(roles==2).view(-1,1,1)
    
    for l in target_layers:
        model.model.layers._modules[str(l)].self_attn.enable_mask = True

    input_probabilities = F.log_softmax(model(inputs=inputs, labels=labels).logits.detach(), dim=-1)
    
    coefficients = torch.tensor([lambda_map[role.item()] for role in roles],device=target_probabilities.device).view(-1,1,1)    

    return input_probabilities, F.log_softmax(target_probabilities, dim=-1), coefficients

def train(model, optimizer, scheduler, train_data_loader, test_dataloader, target_layers, lambda_map, wandb_run, val_every=1000, epochs=1):

    criterion = F.kl_div
    train_kl_loss = 0
    train_sparsity_loss = 0
    for epoch in range(epochs):
        for idx, (tokenized_data, roles) in enumerate(tqdm(train_data_loader)):

            inputs = tokenized_data.to(model.device).long()   
            roles = roles.to(model.device)
            input_probabilities, target_probabilities, coefficients = get_probabilities(model, target_layers, inputs, roles, lambda_map)

            loss = criterion(input_probabilities, target_probabilities, reduction="none", log_target=True)
            loss = (loss*coefficients).mean()
            train_kl_loss += loss.item()
            optimizer.zero_grad()
            sparsity_loss = compute_mask_loss(model)
            train_sparsity_loss += sparsity_loss.item()
            loss += sparsity_loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            

            if idx%val_every == 0:
                model.eval()
                val_kl_loss = 0
                val_sparsity_loss = 0
                for tokenized_data, roles in test_dataloader:
                    with torch.no_grad():
                        inputs = tokenized_data.to(model.device).long()   
                        roles = roles.to(model.device)
                        input_probabilities, target_probabilities, coefficients = get_probabilities(model, target_layers, inputs, roles, lambda_map)
                        loss = criterion(input_probabilities, target_probabilities, reduction="none", log_target=True)
                        loss = (loss*coefficients).mean()
                        val_kl_loss += loss.item()
                        optimizer.zero_grad()
                        sparsity_loss = compute_mask_loss(model)
                        val_sparsity_loss += sparsity_loss.item()
       
                model.train()
                wandb_run.log({"train_kl_loss": train_kl_loss/val_every, "train_sparsity_loss": train_sparsity_loss/val_every, "val_kl_loss": val_kl_loss/len(test_dataloader), "val_sparsity_loss": val_sparsity_loss/len(test_dataloader),"epoch": epoch})
                train_kl_loss = 0
                train_sparsity_loss = 0
                val_kl_loss = 0
                val_sparsity_loss = 0
        