import torch
from tqdm import tqdm


def get_importance_score(model, dataloader, assistant_token=32001):
    model.eval()
    
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(model.device)

    for batch in tqdm(dataloader):
        batch = batch["tokenized"]
        inputs = batch[:,1:].to(model.device)
        labels = batch[:,:-1].to(model.device)
        indices = (labels == assistant_token).nonzero(as_tuple=True)
        mask = (torch.cumsum(torch.ones_like(labels),dim=-1)-1)<=indices[1].view(-1,1)
        labels[mask] = -100
        outputs = model(input_ids=inputs, labels=labels, output_attentions=True)

        for attn in outputs.attentions:
            attn.retain_grad()

        outputs.loss.backward()

        head_importance_ = torch.zeros_like(head_importance)

        for layer, attn in enumerate(outputs.attentions):
            for head in range(model.config.num_attention_heads):
                head_importance_[layer,head] += torch.bmm(attn[:,head].transpose(-2,-1),attn.grad[:,head]).abs().mean()
        head_importance_ /= head_importance_.sum()
        head_importance += head_importance_

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
    return head_importance/len(dataloader)

def get_dataloader(dataset, tokenizer, batch_size=1, max_length=4096):
    dataset = dataset.map(lambda x: {"tokenized":tokenizer.apply_chat_template([{"role":"user", "content":x["user"]},{"role":"assistant","content":x["assistant"]}], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")[0]})
    dataset.set_format("torch", columns=["tokenized"])
    columns_to_remove = ["user", "assistant", "role"]
    dataset = dataset.remove_columns(columns_to_remove)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)