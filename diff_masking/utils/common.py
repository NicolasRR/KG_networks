import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset

def get_masked_weights_probabilitic(weights, mask_param, tau, training):
    if training:
        U1 = torch.rand_like(mask_param).requires_grad_(False)
        U1 += (U1<=1e-12)*1e-12
        U2 = torch.rand_like(mask_param).requires_grad_(False)
        U2 += (U2<=1e-12)*1e-12
        mask = F.sigmoid((mask_param-torch.log(torch.log(U1)/torch.log(U2))) / tau)
    else:
        mask = F.sigmoid(mask_param / tau)

    detached_mask = ((mask>0.5).to(mask.dtype) - mask).detach()

    return weights*(1-(detached_mask + mask))

def get_masked_weights_deterministic(weights, mask_param, tau, training):

    mask = F.sigmoid(mask_param / tau)

    detached_mask = ((mask>0.5).to(mask.dtype) - mask).detach()

    return weights*(1-(detached_mask + mask))

maskings = {
    "probabilistic": get_masked_weights_probabilitic,
    "deterministic": get_masked_weights_deterministic
}

class KLDataset(Dataset):
    def __init__(self, dataset, tokenizer, role_map):
        """
        Args:
            dataset: Huggingface dataset object
        """
        self.dataset = dataset  
        self.tokenizer = tokenizer
        self.role_map = role_map

    def __len__(self):
        # Return the total number of data points, i.e., the total number of files
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the data from the file

        data = self.dataset.select([idx])[0]
        tokenized_data = self.tokenizer.apply_chat_template([{"role": "user", "content": data["user"]}, {"role": "assistant", "content": data["assistant"]}], tokenize=True, padding="max_length", max_length=4096, truncation=True, return_tensors="pt")[0]
        attention_mask = (tokenized_data!=self.tokenizer.pad_token_id)
        attention_mask[-1] = True
        role = self.role_map[data["role"]]

        return tokenized_data,attention_mask.int(), torch.tensor(role)
    
class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        """
        Args:
            dataset: Huggingface dataset object
        """
        self.dataset = dataset  
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the total number of data points, i.e., the total number of files
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the data from the file

        data = self.dataset.select([idx])[0]
        tokenized_data = self.tokenizer.apply_chat_template([{"role": "user", "content": data["user"]}], tokenize=True, padding=False,  add_generation_prompt=True,return_tensors="pt")[0]
        return tokenized_data, idx