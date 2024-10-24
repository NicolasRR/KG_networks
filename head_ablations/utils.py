import torch
from tqdm import tqdm
import numpy as np
from transformers.models.phi3.modeling_phi3 import Phi3Attention,apply_rotary_pos_emb,repeat_kv
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from torch import nn
import math
from transformers.utils import logging
logger = logging.get_logger(__name__)

def get_importance_score(model, dataloader, target_layers, assistant_token=32001):

    for i in range(model.config.num_hidden_layers):
        attn = model.model.layers._modules[str(i)].self_attn
        state_dict = attn.state_dict()
        self_attention = AttentionImportance(attn.config,attn.layer_idx).to(model.dtype).to(model.device)
        self_attention.load_state_dict(state_dict)

        model.model.layers._modules[str(i)].self_attn = self_attention 
    
    for param in model.parameters():
        param.requires_grad = False

    for i in target_layers:
        self_att = model.model.layers._modules[str(i)].self_attn
        self_att.rm_attn = False
        for param in self_att.parameters():
            param.requires_grad = True

    model.eval()
    
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(model.device)

    for batch in tqdm(dataloader):
        batch = batch["tokenized"]
        inputs = batch[:,1:].clone().to(model.device)
        labels = batch[:,:-1].clone().to(model.device)
        indices = (labels == assistant_token).nonzero(as_tuple=True)
        mask = (torch.cumsum(torch.ones_like(labels),dim=-1)-1)<=indices[1].view(-1,1)
        labels[mask] = -100
        outputs = model(input_ids=inputs, labels=labels, output_attentions=True)
        head_importance_ = torch.zeros_like(head_importance)

        for layer in target_layers:
            attn = outputs.attentions[layer]
            attn_grad = torch.autograd.grad(outputs.loss, attn, retain_graph=True)[0]

            for head in range(model.config.num_attention_heads):
                head_importance_[layer,head] = (torch.bmm(attn[:,head].transpose(-2,-1),attn_grad[:,head]).abs().mean()).detach()

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        del attn_grad, attn, outputs
        torch.cuda.empty_cache()
        head_importance_ = head_importance_.detach()
        head_importance_ /= head_importance_.sum()
        head_importance += head_importance_
                                
    return (head_importance/len(dataloader)).detach()

def get_dataloader(dataset, tokenizer, batch_size=1, max_length=4096):
    dataset = dataset.map(lambda x: {"tokenized":tokenizer.apply_chat_template([{"role":"user", "content":x["user"]},{"role":"assistant","content":x["assistant"]}], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")[0]})
    dataset.set_format("torch", columns=["tokenized"])
    columns_to_remove = ["user", "assistant", "role"]
    dataset = dataset.remove_columns(columns_to_remove)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_prefix_matching_scores(model, filtered_ranks, layers, heads, num_seeds=100):
    prefix_matching = []
    with torch.no_grad():
        for seed in tqdm(range(1,num_seeds)):
            L = 2*seed+23
            np.random.seed(seed)
            X = np.random.choice(filtered_ranks, size=L, replace=False)
            X = torch.tensor(np.repeat(X, 4))
            attentions = torch.cat(model(torch.tensor(X).unsqueeze(0).to(model.device),output_attentions=True).attentions)
            scores = torch.zeros((layers, heads))
            for l in range(layers):
                for h in range(heads):
                    att = attentions[l][h]
                    for token_idx in range(L+1, 4*L):
                        att_token = att[token_idx]
                        previous_tokens = X[:token_idx]
                        token = X[token_idx]
                        idx = (previous_tokens == token).nonzero()
                        for i in idx:
                            prefix_score = att_token[i+1]
                            scores[l,h] += prefix_score.item()
                    scores[l,h] /= 3*L
            prefix_matching.append(scores)
    return torch.stack(prefix_matching)

class AttentionAblation(Phi3Attention):
    def __init__(self, config, layer_idx, ablate=False, ablate_idx=None):
        super().__init__(config, layer_idx)
        self.ablate = ablate
        self.ablate_idx = ablate_idx

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")

            bsz, q_len, _ = hidden_states.size()

            qkv = self.qkv_proj(hidden_states)
            query_pos = self.num_heads * self.head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
            value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

            attn_output = torch.matmul(attn_weights, value_states)
            ############################
            if self.ablate and self.ablate_idx is not None:
                for i in self.ablate_idx:
                    attn_output[:,i,:,:] = 0.0
            ############################

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
    
def get_ablated_model(model):

    for i in range(model.config.num_hidden_layers):
        attn = model.model.layers._modules[str(i)].self_attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        self_attention = AttentionAblation(attn.config,attn.layer_idx, ablate=False, ablate_idx=[]).to(device).to(dtype)
        self_attention.load_state_dict(attn.state_dict())
        model.model.layers._modules[str(i)].self_attn = self_attention 

    
    return model

class AttentionImportance(Phi3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rm_attn = True
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        if self.rm_attn:
            attn_weights = None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value