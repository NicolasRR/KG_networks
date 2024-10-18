import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding,apply_rotary_pos_emb,repeat_kv,Phi3LongRoPEScaledRotaryEmbedding,ACT2FN,Phi3Config
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from torch import nn
import torch.nn.functional as F
from transformers.utils import (
    logging,
) 
from datasets import Dataset
import math
logger = logging.get_logger(__name__)

def _get_masked_weights(weights, mask_param, tau, training):
    if training:
        U1 = torch.rand_like(mask_param).requires_grad_(False)
        U2 = torch.rand_like(mask_param).requires_grad_(False)
        mask = F.sigmoid((mask_param-torch.log(torch.log(U1)/torch.log(U2))) / tau)
    else:
        mask = F.sigmoid(mask_param / tau)

    detached_mask = ((mask>0.5).to(mask.dtype) - mask).detach()

    return weights*(1-(detached_mask + mask))

def get_masks(model):
    masks = []
    for name, param in model.named_parameters():
        if "mask" in name:
            masks.append(param)
    return masks

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
        role = self.role_map[data["role"]]

        return tokenized_data,torch.tensor(role)

class Phi3Attention_masked(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)
        self._init_rope()

        #########
        x = 0.45 # TODO magic number
        center = torch.log(torch.tensor(x/(1-x)))
        self.mask_o_proj = nn.Parameter(torch.normal(center, std=0.1*torch.abs(center), size=self.o_proj.weight.shape))
        self.mask_qkv_proj = nn.Parameter(torch.normal(center, std=0.1*torch.abs(center), size=self.qkv_proj.weight.shape))

        self.tau = torch.nn.Parameter(torch.tensor(1.0)).requires_grad_(False) # TODO magic number
        self.mask_enabled = True
        #########

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            

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

        #########
        if self.mask_enabled:
            masked_weigths = _get_masked_weights(self.qkv_proj.weight, self.mask_qkv_proj, self.tau, training=self.training)
        else:
            masked_weigths = self.qkv_proj.weight
        qkv = nn.functional.linear(hidden_states, masked_weigths, self.qkv_proj.bias)
        #########
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

        #########
        if self.mask_enabled:
            masked_weigths = _get_masked_weights(self.o_proj.weight, self.mask_o_proj, self.tau, training=self.training)
        else:
            masked_weigths = self.o_proj.weight
        attn_output = nn.functional.linear(attn_output, masked_weigths, self.o_proj.bias)
        #########



        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Phi3MLP_masked(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

        #########
        x = 0.45 # TODO magic number
        center = torch.log(torch.tensor(x/(1-x))).detach()
        self.mask_gate_up_proj = nn.Parameter(torch.normal(center, std=0.1*torch.abs(center),size=self.gate_up_proj.weight.shape))
        self.mask_down_proj = nn.Parameter(torch.normal(center, std=0.1*torch.abs(center),size=self.down_proj.weight.shape))
        self.tau = torch.nn.Parameter(torch.tensor(1.0)).requires_grad_(False) 
        self.mask_enabled = True
        #########
        
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        
        #########
        if self.mask_enabled:
            masked_weigths = _get_masked_weights(self.gate_up_proj.weight, self.mask_gate_up_proj, self.tau, training=self.training)
        else:
            masked_weigths = self.gate_up_proj.weight
        up_states = nn.functional.linear(hidden_states, masked_weigths, self.gate_up_proj.bias)
        #########
        
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        #########
        if self.mask_enabled:
            masked_weigths = _get_masked_weights(self.down_proj.weight, self.mask_down_proj, self.tau, training=self.training)
        else:
            masked_weigths = self.down_proj.weight  
        output = nn.functional.linear(up_states, masked_weigths, self.down_proj.bias)
        #########

        return output

def create_masked_phi(model, target_layers):
    for param in model.parameters():
        param.requires_grad = False
    
    for i in target_layers:
        # Self Attention
        self_attn = model.model.layers._modules[str(i)].self_attn
        o_proj_state_dict = self_attn.o_proj.state_dict()
        qkv_proj_state_dict = self_attn.qkv_proj.state_dict()

        self_attention = Phi3Attention_masked(self_attn.config,self_attn.layer_idx).to(model.device, model.dtype)

        self_attention.o_proj.load_state_dict(o_proj_state_dict)
        self_attention.qkv_proj.load_state_dict(qkv_proj_state_dict)
        
        for param in self_attention.o_proj.parameters():
            param.requires_grad = False
        
        for param in self_attention.qkv_proj.parameters():
            param.requires_grad = False

        model.model.layers._modules[str(i)].self_attn = self_attention 

        # MLP
        mlp = model.model.layers._modules[str(i)].mlp

        gate_up_proj_state_dict = mlp.gate_up_proj.state_dict()
        down_proj_state_dict = mlp.down_proj.state_dict()

        mlp_mod = Phi3MLP_masked(mlp.config).to(model.device, model.dtype)

        mlp_mod.gate_up_proj.load_state_dict(gate_up_proj_state_dict)
        mlp_mod.down_proj.load_state_dict(down_proj_state_dict)
        
        for param in mlp_mod.gate_up_proj.parameters():
            param.requires_grad = False
        
        for param in mlp_mod.down_proj.parameters():
            param.requires_grad = False

        model.model.layers._modules[str(i)].mlp = mlp_mod 
    
    return model