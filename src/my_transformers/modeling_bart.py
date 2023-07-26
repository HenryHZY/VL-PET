# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model. """


import math
import random
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.bart.configuration_bart import BartConfig

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from adapters import (
    AdapterLayer, 
    AdapterController,
    OutputParallelAdapterLayer,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController,
    MetaLayersAdapterController
)

import lora
from transformers.activations import get_activation
import copy


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # import pdb
        # pdb.set_trace()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # print(key_states.shape)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartAttentionWithValueAdapter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        adapter_config = None,
        config = None, #【补充：传入config更方便】
        # use_decoder_enc_attn_value_sequential_adapter=False
        # task = None # 【补充：这个应该在forward的时候补充的】
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.use_decoder_enc_attn_value_residual_connection = config.use_decoder_enc_attn_value_residual_connection
        if config.use_decoder_enc_attn_value_sequential_adapter_down_dim:
            self.enc_attn_value_sequential_adapter = AdapterController(adapter_config)
        else:
            self.enc_attn_value_sequential_adapter = None
            
        if config.use_decoder_enc_attn_value_sequential_adapter_gating_large_x_lowrank and config.use_decoder_enc_attn_value_residual_connection:
            self.gating_non_linear = get_activation('gelu_new')
            self.decoder_enc_attn_value_sequential_adapter_gating_large_x_down = nn.Linear(self.embed_dim, config.decoder_enc_attn_value_sequential_adapter_gating_large_x_lowrank_down_dim)
            self.decoder_enc_attn_value_sequential_adapter_gating_large_x_up = nn.Linear(config.decoder_enc_attn_value_sequential_adapter_gating_large_x_lowrank_down_dim, self.embed_dim)
        else:
            self.decoder_enc_attn_value_sequential_adapter_gating_large_x_down = None
            self.decoder_enc_attn_value_sequential_adapter_gating_large_x_up = None
                
        if config.use_decoder_enc_attn_value_parallel_adapter_down_dim or \
            config.use_encoder_attn_value_parallel_adapter_down_dim or \
            config.use_decoder_self_attn_value_parallel_adapter_down_dim: 
            #【补充：默认用decoder enc-attn-Value-ParallelAdapter】
            parallel_adapter_config = copy.deepcopy(adapter_config)
            parallel_adapter_config.use_parallel_adapter = True
            if config.use_decoder_enc_attn_value_parallel_adapter_down_dim and config.use_decoder_enc_attn_value_parallel_adapter_scaling:
                parallel_adapter_config.use_scaling_factor = True
                parallel_adapter_config.scaling_factor = config.decoder_enc_attn_value_parallel_adapter_scaling_factor
            self.attn_value_parallel_adapter = AdapterController(parallel_adapter_config)
        else:
            self.attn_value_parallel_adapter = None

        if config.use_decoder_enc_attn_value_parallel_adapter_gating_large_x_lowrank and config.use_decoder_enc_attn_value_residual_connection:
            self.gating_non_linear = get_activation('gelu_new')
            self.decoder_enc_attn_value_parallel_adapter_gating_large_x_down = nn.Linear(self.embed_dim, config.decoder_enc_attn_value_parallel_adapter_gating_large_x_lowrank_down_dim)
            self.decoder_enc_attn_value_parallel_adapter_gating_large_x_up = nn.Linear(config.decoder_enc_attn_value_parallel_adapter_gating_large_x_lowrank_down_dim, self.embed_dim)
        else:
            self.decoder_enc_attn_value_parallel_adapter_gating_large_x_down = None
            self.decoder_enc_attn_value_parallel_adapter_gating_large_x_up = None

        if config.use_decoder_enc_attn_value_ia3 or config.use_encoder_attn_value_ia3 or \
            config.use_decoder_self_attn_value_ia3:
            self.attn_value_ia3 = nn.Parameter(torch.zeros(self.embed_dim))
            nn.init.normal_(self.attn_value_ia3, std=0.02) # 因为会调用init_weight，实际上就是这个，需要到multitask.py进行one init 
        else:
            self.attn_value_ia3 = None

        self.adapter_non_linear = get_activation('gelu_new')
        if config.use_decoder_enc_attn_value_parallel_adapter_down_multihead: # 【补充：h个 768 → x/h，拼接，再加上x → 768】
            self.decoder_enc_attn_value_parallel_adapter_multihead_dim = int(
                config.decoder_enc_attn_value_parallel_adapter_down_dim / 
                config.decoder_enc_attn_value_parallel_adapter_multihead_num_head) # x/h

            self.decoder_enc_attn_value_parallel_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.decoder_enc_attn_value_parallel_adapter_multihead_dim) 
                for _ in range(config.decoder_enc_attn_value_parallel_adapter_multihead_num_head)
                ])

            self.decoder_enc_attn_value_parallel_adapter_multihead_up = nn.Linear(
                config.decoder_enc_attn_value_parallel_adapter_down_dim, 
                self.embed_dim)

        elif config.use_decoder_enc_attn_value_parallel_adapter_down_up_pair_multihead: # 【补充：h对 768 → x/h → 768/h】
            self.decoder_enc_attn_value_parallel_adapter_multihead_dim1 = int(
                config.decoder_enc_attn_value_parallel_adapter_down_dim / 
                config.decoder_enc_attn_value_parallel_adapter_multihead_num_head) # x/h
            
            self.decoder_enc_attn_value_parallel_adapter_multihead_dim2 = int(
                self.embed_dim / 
                config.decoder_enc_attn_value_parallel_adapter_multihead_num_head) # 768/h，跟上面不同
            
            self.decoder_enc_attn_value_parallel_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.decoder_enc_attn_value_parallel_adapter_multihead_dim1) 
                for _ in range(config.decoder_enc_attn_value_parallel_adapter_multihead_num_head)
                ])
            self.decoder_enc_attn_value_parallel_adapter_multihead_up = nn.ModuleList([
                nn.Linear(self.decoder_enc_attn_value_parallel_adapter_multihead_dim1, self.decoder_enc_attn_value_parallel_adapter_multihead_dim2) 
                for _ in range(config.decoder_enc_attn_value_parallel_adapter_multihead_num_head)
                ])

        else:
            self.decoder_enc_attn_value_parallel_adapter_multihead_down = None
            self.decoder_enc_attn_value_parallel_adapter_multihead_up = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        task=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # import pdb
        # pdb.set_trace()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            
            value_states = self.v_proj(key_value_states)

            if self.attn_value_parallel_adapter is not None:
                value_states = self.attn_value_parallel_adapter(key_value_states, task, y=value_states)
            elif self.enc_attn_value_sequential_adapter is not None:
                value_states = self.enc_attn_value_sequential_adapter(key_value_states, task)
            elif self.attn_value_ia3 is not None:
                if self.config.use_decoder_enc_attn_value_ia3_add or self.config.use_encoder_attn_value_ia3_add or \
                    self.config.use_decoder_self_attn_value_ia3_add:
                    value_states = value_states + value_states*self.attn_value_ia3
                else:
                    value_states = value_states*self.attn_value_ia3
            elif self.decoder_enc_attn_value_parallel_adapter_multihead_down is not None:
                if self.config.use_decoder_enc_attn_value_parallel_adapter_down_multihead:
                    adapter_down_multihead_output = [
                        self.decoder_enc_attn_value_parallel_adapter_multihead_down[i](key_value_states) # 【Parallel Adapter，所以input是Residual】
                        for i in range(self.config.decoder_enc_attn_value_parallel_adapter_multihead_num_head)
                        ] # 【补充：multihead和拼接】
                    adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)
                    adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                    adapter_up_multihead_output = self.decoder_enc_attn_value_parallel_adapter_multihead_up(adapter_down_multihead_output)
                    hidden_states = value_states + adapter_up_multihead_output # 【补充：残差连接是Hidden】
                elif self.config.use_decoder_enc_attn_value_parallel_adapter_down_up_pair_multihead:
                    adapter_up_multihead_output = [
                        self.decoder_enc_attn_value_parallel_adapter_multihead_up[i](
                            self.adapter_non_linear(
                                self.decoder_enc_attn_value_parallel_adapter_multihead_down[i](key_value_states)))
                        for i in range(self.config.decoder_enc_attn_value_parallel_adapter_multihead_num_head)
                        ] # 【补充：multihead】【不需要中间进行拼接因为一对对】
                    adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)
                    hidden_states = value_states + adapter_up_multihead_output # 【补充：残差连接】
                    

            # print('key_value_states.shape = ',key_value_states.shape)
            # print('value_states.shape = ',value_states.shape)

            if self.use_decoder_enc_attn_value_residual_connection:
                residual = key_value_states
                if self.decoder_enc_attn_value_sequential_adapter_gating_large_x_down is not None:
                    gate = self.decoder_enc_attn_value_sequential_adapter_gating_large_x_down(residual)
                    gate = self.gating_non_linear(gate)
                    gate = self.decoder_enc_attn_value_sequential_adapter_gating_large_x_up(gate)
                    gate = torch.sigmoid(gate)
                    value_states = value_states*gate
                elif self.decoder_enc_attn_value_parallel_adapter_gating_large_x_down is not None:
                    gate = self.decoder_enc_attn_value_parallel_adapter_gating_large_x_down(residual)
                    gate = self.gating_non_linear(gate)
                    gate = self.decoder_enc_attn_value_parallel_adapter_gating_large_x_up(gate)
                    gate = torch.sigmoid(gate)
                    value_states = value_states*gate
                value_states = residual + value_states

            value_states = self._shape(value_states, -1, bsz)
            # print('value_states.shape 2= ',value_states.shape)
            # key_value_states.shape =  torch.Size([416, 42, 768])
            # value_states.shape =  torch.Size([416, 42, 768])
            # value_states.shape 2=  torch.Size([416, 12, 42, 64])
            # exit(0)

        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # print(key_states.shape)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class BartAttentionWithKeyAdapter(nn.Module): # copy BartAttentionWithValueAdapter
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        adapter_config = None,
        config = None, #【补充：传入config更方便】
        # use_decoder_enc_attn_value_sequential_adapter=False
        # task = None # 【补充：这个应该在forward的时候补充的】
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                
        if config.use_decoder_enc_attn_key_parallel_adapter_down_dim:
            #【补充：默认用decoder enc-attn-key-ParallelAdapter】
            parallel_adapter_config = copy.deepcopy(adapter_config)
            parallel_adapter_config.use_parallel_adapter = True
            self.attn_key_parallel_adapter = AdapterController(parallel_adapter_config)
        else:
            self.attn_key_parallel_adapter = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        task=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # import pdb
        # pdb.set_trace()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            # key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

            key_states = self.k_proj(key_value_states)

            if self.attn_key_parallel_adapter is not None:
                key_states = self.attn_key_parallel_adapter(key_value_states, task, y=key_states)

            value_states = self._shape(value_states, -1, bsz)
            # print('value_states.shape 2= ',value_states.shape)
            # key_value_states.shape =  torch.Size([416, 42, 768])
            # value_states.shape =  torch.Size([416, 42, 768])
            # value_states.shape 2=  torch.Size([416, 12, 42, 64])
            # exit(0)

        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # print(key_states.shape)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value



class LoraBartAttention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        config=None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            is_decoder,
            bias,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = lora.LoRALinearController(embed_dim, embed_dim, config=config, bias=bias)
        self.q_proj = lora.LoRALinearController(embed_dim, embed_dim, config=config, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        task=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # import pdb
        # pdb.set_trace()

        # get query proj
        query_states = self.q_proj(hidden_states, task) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states, task), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, task), -1, bsz)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, task), -1, bsz)

        # print(key_states.shape)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.use_lora = config.use_lora
        if config.use_lora:
            self.self_attn = LoraBartAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config.lora_config,
            )
        elif config.use_encoder_attn_value_parallel_adapter_down_dim:
            config.use_encoder_attn_value_parallel_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_encoder_attn_value_parallel_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_encoder_attn_value_parallel_adapter_down_dim_config.adapter_down_dim = config.encoder_attn_value_parallel_adapter_down_dim
            self.self_attn = BartAttentionWithValueAdapter(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                adapter_config=config.use_encoder_attn_value_parallel_adapter_down_dim_config,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        elif config.use_encoder_attn_value_ia3:
            self.self_attn = BartAttentionWithValueAdapter(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                adapter_config=None,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        else:
            self.self_attn = BartAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
            )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        if (config.use_adapter or config.use_compacter or config.use_lradapter) and not config.no_encoder_adapter:
            self.attn_adapter = AdapterController(config.adapter_config)
            self.ff_adapter = AdapterController(config.adapter_config)
            if config.use_encoder_attn_adapter_scaling:
                encoder_attn_adapter_scaling_config = copy.deepcopy(config.adapter_config)
                encoder_attn_adapter_scaling_config.use_scaling_factor = True
                encoder_attn_adapter_scaling_config.scaling_factor = config.encoder_attn_adapter_scaling_factor
                self.attn_adapter = AdapterController(encoder_attn_adapter_scaling_config)
            if config.use_encoder_ff_adapter_scaling:
                encoder_ff_adapter_scaling_config = copy.deepcopy(config.adapter_config)
                encoder_ff_adapter_scaling_config.use_scaling_factor = True
                encoder_ff_adapter_scaling_config.scaling_factor = config.encoder_ff_adapter_scaling_factor
                self.ff_adapter = AdapterController(encoder_ff_adapter_scaling_config)
        else:
            self.attn_adapter = None
            self.ff_adapter = None

        self.adapter_hypernet = None
        if config.use_hyperformer:
            self.adapter_hypernet = MetaLayersAdapterController(config.adapter_config)

        if config.use_encoder_adapter_gating_layernorm:
            self.encoder_attn_adapter_gating_layernorm = nn.LayerNorm(self.embed_dim)
            self.encoder_ff_adapter_gating_layernorm = nn.LayerNorm(self.embed_dim)
        else:
            self.encoder_attn_adapter_gating_layernorm = None
            self.encoder_ff_adapter_gating_layernorm = None

        if config.use_encoder_adapter_gating_l2norm:
            # self.encoder_attn_adapter_gating_l2norm = F.normalize(x, p=2, dim=-1)
            # self.encoder_ff_adapter_gating_l2norm= F.normalize(x, p=2, dim=-1)
            self.encoder_attn_adapter_gating_l2norm = True
            self.encoder_ff_adapter_gating_l2norm = True
        else:
            self.encoder_attn_adapter_gating_l2norm = None
            self.encoder_ff_adapter_gating_l2norm = None

        if config.use_encoder_adapter_gating_large_x:
            self.encoder_attn_adapter_gating_large_x = nn.Linear(self.embed_dim, self.embed_dim)
            self.encoder_ff_adapter_gating_large_x = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.encoder_attn_adapter_gating_large_x = None
            self.encoder_ff_adapter_gating_large_x = None

        if config.use_encoder_adapter_gating_small_xy_cat:
            self.encoder_attn_adapter_gating_small_xy_cat = nn.Linear(self.embed_dim*2, 1)
            self.encoder_ff_adapter_gating_small_xy_cat = nn.Linear(self.embed_dim*2, 1)
        else:
            self.encoder_attn_adapter_gating_small_xy_cat = None
            self.encoder_ff_adapter_gating_small_xy_cat = None
        
        if config.use_encoder_adapter_gating_middle_xy_add:
            self.encoder_attn_adapter_gating_middle_xy_add = nn.Linear(self.embed_dim, 1)
            self.encoder_ff_adapter_gating_middle_xy_add = nn.Linear(self.embed_dim, 1)
        else:
            self.encoder_attn_adapter_gating_middle_xy_add = None
            self.encoder_ff_adapter_gating_middle_xy_add = None

        if config.use_encoder_adapter_gating_middle_ia3_add:
            self.encoder_attn_adapter_gating_middle_ia3_add = nn.Parameter(torch.zeros(self.embed_dim))
            self.encoder_ff_adapter_gating_middle_ia3_add = nn.Parameter(torch.zeros(self.embed_dim))
            nn.init.normal_(self.encoder_attn_adapter_gating_middle_ia3_add, std=0.02)
            nn.init.normal_(self.encoder_ff_adapter_gating_middle_ia3_add, std=0.02)
            # 【补充：无效操作，需要在multitask.py那里初始化权重，默认的权重初始化其实就是上面这行】
        else:
            self.encoder_attn_adapter_gating_middle_ia3_add = None
            self.encoder_ff_adapter_gating_middle_ia3_add = None

        self.gating_non_linear = get_activation('gelu_new')
        if config.use_encoder_adapter_gating_large_x_lowrank:
            self.encoder_attn_adapter_gating_large_x_down = nn.Linear(self.embed_dim, config.adapter_gating_down_dim)
            self.encoder_attn_adapter_gating_large_x_up = nn.Linear(config.adapter_gating_down_dim, self.embed_dim)
            self.encoder_ff_adapter_gating_large_x_down = nn.Linear(self.embed_dim, config.adapter_gating_down_dim)
            self.encoder_ff_adapter_gating_large_x_up = nn.Linear(config.adapter_gating_down_dim, self.embed_dim)
            # self.gating_non_linear = get_activation('gelu_new')
        else:
            self.encoder_attn_adapter_gating_large_x_down = None
            self.encoder_attn_adapter_gating_large_x_up = None
            self.encoder_ff_adapter_gating_large_x_down = None
            self.encoder_ff_adapter_gating_large_x_up = None
            # self.gating_non_linear = None

        if config.use_encoder_gating_large_x_lowrank:
            self.encoder_attn_gating_large_x_down = nn.Linear(self.embed_dim, config.gating_down_dim)
            self.encoder_attn_gating_large_x_up = nn.Linear(config.gating_down_dim, self.embed_dim)
            self.encoder_ff_gating_large_x_down = nn.Linear(self.embed_dim, config.gating_down_dim)
            self.encoder_ff_gating_large_x_up = nn.Linear(config.gating_down_dim, self.embed_dim)
        else:
            self.encoder_attn_gating_large_x_down = None
            self.encoder_attn_gating_large_x_up = None
            self.encoder_ff_gating_large_x_down = None
            self.encoder_ff_gating_large_x_up = None
            # self.gating_non_linear = None
            # 补充：这里会导致上面的gating_non_linear变为None。。

        if config.no_encoder_attn_adapter: # 【补充：弃用，放弃在下面加入新的attn_adapter模块】
            self.attn_adapter = None

            self.encoder_attn_adapter_gating_large_x = None

            self.encoder_attn_adapter_gating_small_xy_cat = None

            self.encoder_attn_adapter_gating_middle_xy_add = None

            self.encoder_attn_adapter_gating_middle_ia3_add

            self.encoder_attn_adapter_gating_large_x_down = None
            self.encoder_attn_adapter_gating_large_x_up = None

            self.encoder_attn_gating_large_x_down = None
            self.encoder_attn_gating_large_x_up = None

        self.adapter_non_linear = get_activation('gelu_new')
        if config.use_encoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
            self.encoder_adapter_multihead_dim = int(config.adapter_down_dim/config.encoder_adapter_multihead_num_head) # x/h
            self.attn_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.attn_adapter_multihead_up = nn.Linear(config.adapter_down_dim, self.embed_dim)
            self.ff_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.ff_adapter_multihead_up = nn.Linear(config.adapter_down_dim, self.embed_dim)

        elif config.use_encoder_adapter_up_multihead: # 【补充 768 → x，再加上h个 x → 768/h】
            self.attn_adapter_multihead_down = nn.Linear(self.embed_dim, config.adapter_down_dim)
            self.encoder_adapter_multihead_dim = int(self.embed_dim/config.encoder_adapter_multihead_num_head) # 768/h，跟上面不同
            self.attn_adapter_multihead_up = nn.ModuleList([
                nn.Linear(config.adapter_down_dim, self.encoder_adapter_multihead_dim) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.ff_adapter_multihead_down = nn.Linear(self.embed_dim, config.adapter_down_dim)
            self.ff_adapter_multihead_up = nn.ModuleList([
                nn.Linear(config.adapter_down_dim, self.encoder_adapter_multihead_dim) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
        
        elif config.use_encoder_adapter_down_up_multihead: # 【补充：h个 768 → x/h，拼接，再加上h个 x → 768/h，拼接】
            self.encoder_adapter_multihead_dim1 = int(config.adapter_down_dim/config.encoder_adapter_multihead_num_head) # x/h
            self.encoder_adapter_multihead_dim2 = int(self.embed_dim/config.encoder_adapter_multihead_num_head) # 768/h，跟上面不同

            self.attn_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim1) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.attn_adapter_multihead_up = nn.ModuleList([
                nn.Linear(config.adapter_down_dim, self.encoder_adapter_multihead_dim2) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            
            self.ff_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim1) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.ff_adapter_multihead_up = nn.ModuleList([
                nn.Linear(config.adapter_down_dim, self.encoder_adapter_multihead_dim2) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])

        elif config.use_encoder_adapter_down_up_pair_multihead: # 【补充：h对 768 → x/h → 768/h】
            self.encoder_adapter_multihead_dim1 = int(config.adapter_down_dim/config.encoder_adapter_multihead_num_head) # x/h
            self.encoder_adapter_multihead_dim2 = int(self.embed_dim/config.encoder_adapter_multihead_num_head) # 768/h，跟上面不同

            self.attn_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim1) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.attn_adapter_multihead_up = nn.ModuleList([
                nn.Linear(self.encoder_adapter_multihead_dim1, self.encoder_adapter_multihead_dim2) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            
            self.ff_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.encoder_adapter_multihead_dim1) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])
            self.ff_adapter_multihead_up = nn.ModuleList([
                nn.Linear(self.encoder_adapter_multihead_dim1, self.encoder_adapter_multihead_dim2) 
                for _ in range(config.encoder_adapter_multihead_num_head)
                ])

        else:
            self.attn_adapter_multihead_down = None
            self.attn_adapter_multihead_up = None
            self.ff_adapter_multihead_down = None
            self.ff_adapter_multihead_up = None       


    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, past_key_value: tuple = None, block_adapters=None, task=None, output_attentions: bool = False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        if self.use_lora or self.config.use_encoder_attn_value_parallel_adapter_down_dim:
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, task=task
            )
        else:
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, output_attentions=output_attentions
            )

        # attn模块的输入是x1，输出是x2
        # attn_adapter的输入是x2，中间输出是△y，最终输出是残差连接y1=x2+△y
        if self.attn_adapter is not None:
            hidden_states = self.attn_adapter(hidden_states, task) 
        elif self.attn_adapter_multihead_down is not None: #【补充：记得做上残差连接】
            if self.config.use_encoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
                adapter_down_multihead_output = [self.attn_adapter_multihead_down[i](hidden_states) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                adapter_up_multihead_output = self.attn_adapter_multihead_up(adapter_down_multihead_output)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_up_multihead: # 【补充 768 → x，再加上h个 x → 768/h】
                adapter_down_multihead_output = self.attn_adapter_multihead_down(hidden_states) # 【补充：down和nonlinear】
                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) 

                adapter_up_multihead_output = [self.attn_adapter_multihead_up[i](adapter_down_multihead_output) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_down_up_multihead: # 【补充：h个 768 → x/h，拼接，再加上h个 x → 768/h，拼接】
                adapter_down_multihead_output = [self.attn_adapter_multihead_down[i](hidden_states) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear】

                adapter_up_multihead_output = [self.attn_adapter_multihead_up[i](adapter_down_multihead_output) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_down_up_pair_multihead: # 【补充：h对 768 → x/h → 768/h】
                adapter_up_multihead_output = [
                    self.attn_adapter_multihead_up[i](
                        self.adapter_non_linear(
                            self.attn_adapter_multihead_down[i](hidden_states)))
                    for i in range(self.config.encoder_adapter_multihead_num_head)
                    ] # 【补充：multihead】【不需要中间进行拼接因为一对对】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)
                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

        if self.encoder_attn_adapter_gating_large_x is not None:
            gate = self.encoder_attn_adapter_gating_large_x(residual) # adapter_gating的输入是x1
            gate = torch.sigmoid(gate)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate # adapter_gating的对象是y1
            # 最终在下面的attn模块残差连接，y2 = x1+Gate()*y1 = x1+Gate()*(x2+△y)
        elif self.encoder_attn_adapter_gating_large_x_down is not None:
            gate = self.encoder_attn_adapter_gating_large_x_down(residual)
            gate = self.gating_non_linear(gate)
            gate = self.encoder_attn_adapter_gating_large_x_up(gate)
            gate = torch.sigmoid(gate)
            if self.config.use_store_gate_large:
                store_gate_path = os.path.join(self.config.store_gate_path, task+'_gate_large_encoder_self_attention')
                torch.save(gate, store_gate_path)
                print("gate.shape = ", gate.shape)
                print("store_gate_path = ", store_gate_path)
                exit(0)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate # 前面的Multihead Adapter的output就是hidden_states=h+△h，也就是做了残差连接的，此处相当于(h+△h)*G
        elif self.encoder_attn_adapter_gating_small_xy_cat is not None:
            gate_input = torch.cat([residual, hidden_states], dim=2)
            gate = self.encoder_attn_adapter_gating_small_xy_cat(gate_input)
            gate = torch.sigmoid(gate)
            gate = torch.mean(gate, dim=1).unsqueeze(-1) # [batch,len,1] → torch.mean会[batch, 1]，所以要unsqueeze
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate
        elif self.encoder_attn_adapter_gating_middle_xy_add is not None:
            gate_input = residual + hidden_states
            gate = self.encoder_attn_adapter_gating_middle_xy_add(gate_input)
            gate = torch.sigmoid(gate)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate
        elif self.encoder_attn_adapter_gating_middle_ia3_add is not None:
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + torch.ones_like(hidden_states) + self.encoder_attn_adapter_gating_middle_ia3_add # 此处相当于(h+△h) + G = (h+△h) + 1 + G^
            else:
                hidden_states = hidden_states + hidden_states*self.encoder_attn_adapter_gating_middle_ia3_add # 此处相当于 (h+△h)*G，只不过G=(1+G^)
        elif self.encoder_attn_adapter_gating_layernorm:
            hidden_states = self.encoder_attn_adapter_gating_layernorm(hidden_states)
        elif self.encoder_attn_adapter_gating_l2norm:
            # hidden_states = self.encoder_attn_adapter_gating_l2norm(hidden_states)
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        if self.encoder_attn_gating_large_x_down is not None:
            # gating替代adapter
            # attn模块的输入是x1，输出是x2
            gate = self.encoder_attn_gating_large_x_down(residual) # gating的输入是x1
            gate = self.gating_non_linear(gate)
            gate = self.encoder_attn_gating_large_x_up(gate) # gating的中间输出是△y
            gate_delta_y = gate
            gate = torch.sigmoid(gate) # gating score
            if self.config.use_encoder_gating_large_x_lowrank_add_x2_deltay:
                hidden_states = (gate_delta_y + hidden_states)*gate
            else:
                hidden_states = gate_delta_y + hidden_states*gate # gating score的对象是x2
            # Gate()由△y，没必要再乘△y
            # 最终在下面的attn模块残差连接，y2 = x1+△y+Gate()*x2

        if self.adapter_hypernet is not None:
            hidden_states = self.adapter_hypernet(hidden_states, block_adapters.self_attention)

        if self.config.use_encoder_gating_scaling:
            hidden_states = hidden_states*self.config.encoder_gating_scaling_factor

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states # x + h
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)

        if self.ff_adapter is not None:
            hidden_states = self.ff_adapter(hidden_states, task)
        elif self.ff_adapter_multihead_down is not None: #【补充：记得做上残差连接】
            if self.config.use_encoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
                adapter_down_multihead_output = [self.ff_adapter_multihead_down[i](hidden_states) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                adapter_up_multihead_output = self.ff_adapter_multihead_up(adapter_down_multihead_output)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_up_multihead: # 【补充 768 → x，再加上h个 x → 768/h】
                adapter_down_multihead_output = self.ff_adapter_multihead_down(hidden_states) # 【补充：down和nonlinear】
                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) 

                adapter_up_multihead_output = [self.ff_adapter_multihead_up[i](adapter_down_multihead_output) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_down_up_multihead: # 【补充：h个 768 → x/h，拼接，再加上h个 x → 768/h，拼接】
                adapter_down_multihead_output = [self.ff_adapter_multihead_down[i](hidden_states) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear】

                adapter_up_multihead_output = [self.ff_adapter_multihead_up[i](adapter_down_multihead_output) for i in range(self.config.encoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

            elif self.config.use_encoder_adapter_down_up_pair_multihead: # 【补充：h对 768 → x/h → 768/h】
                adapter_up_multihead_output = [
                    self.ff_adapter_multihead_up[i](
                        self.adapter_non_linear(
                            self.ff_adapter_multihead_down[i](hidden_states)))
                    for i in range(self.config.encoder_adapter_multihead_num_head)
                    ] # 【补充：multihead】【不需要中间进行拼接因为一对对】
                adapter_up_multihead_output = torch.cat(adapter_up_multihead_output, dim=-1)
                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

        if self.encoder_ff_adapter_gating_large_x is not None:
            gate = self.encoder_ff_adapter_gating_large_x(residual) # adapter_gating的输入是x1
            gate = torch.sigmoid(gate)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate # adapter_gating的对象是y1
        elif self.encoder_ff_adapter_gating_large_x_down is not None:
            gate = self.encoder_ff_adapter_gating_large_x_down(residual)
            gate = self.gating_non_linear(gate)
            gate = self.encoder_ff_adapter_gating_large_x_up(gate)
            gate = torch.sigmoid(gate)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate
        elif self.encoder_ff_adapter_gating_small_xy_cat is not None:
            gate_input = torch.cat([residual, hidden_states], dim=2)
            gate = self.encoder_ff_adapter_gating_small_xy_cat(gate_input)
            gate = torch.sigmoid(gate)
            gate = torch.mean(gate, dim=1).unsqueeze(-1) # [batch,len,1] → torch.mean会[batch, 1]，所以要unsqueeze
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate
        elif self.encoder_ff_adapter_gating_middle_xy_add is not None:
            gate_input = residual + hidden_states
            gate = self.encoder_ff_adapter_gating_middle_xy_add(gate_input)
            gate = torch.sigmoid(gate)
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + gate # 此处相当于(h+△h) + G
            else:
                hidden_states = hidden_states*gate
        elif self.encoder_ff_adapter_gating_middle_ia3_add is not None:
            if self.config.use_encoder_adapter_gating_add:
                hidden_states = hidden_states + torch.ones_like(hidden_states) + self.encoder_ff_adapter_gating_middle_ia3_add # 此处相当于(h+△h) + G = (h+△h) + 1 + G^
            else:
                hidden_states = hidden_states + hidden_states*self.encoder_ff_adapter_gating_middle_ia3_add
        elif self.encoder_ff_adapter_gating_layernorm:
            hidden_states = self.encoder_ff_adapter_gating_layernorm(hidden_states)
        elif self.encoder_ff_adapter_gating_l2norm:
            # hidden_states = self.encoder_ff_adapter_gating_l2norm(hidden_states)
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        if self.encoder_ff_gating_large_x_down is not None:
            # gating替代adapter
            # ff模块的输入是x1，输出是x2
            gate = self.encoder_ff_gating_large_x_down(residual) # gating的输入是x1
            gate = self.gating_non_linear(gate)
            gate = self.encoder_ff_gating_large_x_up(gate) # gating的中间输出是△y
            gate_delta_y = gate
            gate = torch.sigmoid(gate) # gating score
            if self.config.use_encoder_gating_large_x_lowrank_add_x2_deltay:
                hidden_states = (gate_delta_y + hidden_states)*gate
            else:
                hidden_states = gate_delta_y + hidden_states*gate # gating score的对象是x2
            # Gate()由△y，没必要再乘△y
            # 最终在下面的ff模块残差连接，y2 = x1+△y+Gate()*x2

        if self.adapter_hypernet is not None:
            hidden_states = self.adapter_hypernet(hidden_states, block_adapters.feed_forward)

        if self.config.use_encoder_gating_scaling:
            hidden_states = hidden_states*self.config.encoder_gating_scaling_factor

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.use_lora = config.use_lora
        if config.use_lora:
            self.self_attn = LoraBartAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                config=config.lora_config,
            )
        elif config.use_decoder_self_attn_value_ia3:
            self.self_attn = BartAttentionWithValueAdapter(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=None,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        elif config.use_decoder_self_attn_value_parallel_adapter_down_dim:
            config.use_decoder_self_attn_value_parallel_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_self_attn_value_parallel_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_self_attn_value_parallel_adapter_down_dim_config.adapter_down_dim = config.decoder_self_attn_value_parallel_adapter_down_dim
            self.self_attn = BartAttentionWithValueAdapter(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=config.use_decoder_self_attn_value_parallel_adapter_down_dim_config,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        else:
            self.self_attn = BartAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)


        if config.use_lora:
            self.encoder_attn = LoraBartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                config=config.lora_config
            )
        elif config.use_decoder_enc_attn_value_parallel_adapter_down_dim:
            config.use_decoder_enc_attn_value_parallel_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_enc_attn_value_parallel_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_enc_attn_value_parallel_adapter_down_dim_config.adapter_down_dim = config.decoder_enc_attn_value_parallel_adapter_down_dim
            self.encoder_attn = BartAttentionWithValueAdapter(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=config.use_decoder_enc_attn_value_parallel_adapter_down_dim_config,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        elif config.use_decoder_enc_attn_key_parallel_adapter_down_dim:
            config.use_decoder_enc_attn_key_parallel_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_enc_attn_key_parallel_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_enc_attn_key_parallel_adapter_down_dim_config.adapter_down_dim = config.decoder_enc_attn_key_parallel_adapter_down_dim
            self.encoder_attn = BartAttentionWithKeyAdapter(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=config.use_decoder_enc_attn_key_parallel_adapter_down_dim_config,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        elif config.use_decoder_enc_attn_value_sequential_adapter_down_dim:
            config.use_decoder_enc_attn_value_sequential_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_enc_attn_value_sequential_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_enc_attn_value_sequential_adapter_down_dim_config.adapter_down_dim = config.decoder_enc_attn_value_sequential_adapter_down_dim
            self.encoder_attn = BartAttentionWithValueAdapter(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=config.use_decoder_enc_attn_value_sequential_adapter_down_dim_config,
                config=self.config
                # use_decoder_enc_attn_value_sequential_adapter=True
                # task=task, #【补充：应该在forward时输入】
            )
        elif config.use_decoder_enc_attn_value_ia3 or \
            config.use_decoder_enc_attn_value_parallel_adapter_down_multihead or \
            config.use_decoder_enc_attn_value_parallel_adapter_down_up_pair_multihead: # 【补充：ia3和MultiheadDownVPA不需要AdapterController那个类，重新定义的】
            self.encoder_attn = BartAttentionWithValueAdapter(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                adapter_config=None,
                config=self.config
                # task=task, #【补充：应该在forward时输入】
            )
        else:
            self.encoder_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        if self.config.use_decoder_ff_ia3:
            self.decoder_ff_ia3 = nn.Parameter(torch.zeros(config.decoder_ffn_dim)) # dim=3072，跟attn ia3不一样
            nn.init.normal_(self.decoder_ff_ia3, std=0.02) # 因为会调用init_weight，实际上就是这个，需要到multitask.py进行one init 
        else:
            self.decoder_ff_ia3 = None

        self.add_adapter_cross_attn = config.add_adapter_cross_attn

        if (config.use_adapter or config.use_compacter or config.use_lradapter) and not config.no_decoder_adapter:
            self.self_attn_adapter = AdapterController(config.adapter_config)

            if config.add_adapter_cross_attn:
                self.enc_attn_adapter = AdapterController(config.adapter_config)
            else:
                self.enc_attn_adapter = None
            self.ff_adapter = AdapterController(config.adapter_config)
        else:
            self.self_attn_adapter = None
            self.enc_attn_adapter = None
            self.ff_adapter = None

        if config.use_decoder_self_attn_adapter_down_dim:
            config.use_decoder_self_attn_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_self_attn_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_self_attn_adapter_down_dim_config.adapter_down_dim = config.decoder_self_attn_adapter_down_dim
            self.decoder_self_attn_adapter = AdapterController(config.use_decoder_self_attn_adapter_down_dim_config)
        else:
            self.decoder_self_attn_adapter = None #【补充：不用self.self_attn_adapter这个名字，就是为了避免把前面的弄成None。。】

        if config.use_decoder_enc_attn_adapter_down_dim:
            config.use_decoder_enc_attn_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_enc_attn_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_enc_attn_adapter_down_dim_config.adapter_down_dim = config.decoder_enc_attn_adapter_down_dim
            self.decoder_enc_attn_adapter = AdapterController(config.use_decoder_enc_attn_adapter_down_dim_config)
        else:
            self.decoder_enc_attn_adapter = None #【补充：不用self.enc_attn_adapter这个名字，就是为了避免把前面的弄成None。。】

        if config.use_decoder_ff_adapter_down_dim:
            config.use_decoder_ff_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_ff_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_ff_adapter_down_dim_config.adapter_down_dim = config.decoder_ff_adapter_down_dim
            self.decoder_ff_adapter = AdapterController(config.use_decoder_ff_adapter_down_dim_config)
        else:
            self.decoder_ff_adapter = None #【补充：不用self.ff_adapter这个名字，就是为了避免把前面的弄成None。。】


        self.adapter_hypernet = None
        if config.use_hyperformer:
            self.adapter_hypernet = MetaLayersAdapterController(config.adapter_config)

        if config.use_decoder_enc_attn_key_value_adapter_down_dim:
            config.use_decoder_enc_attn_key_value_adapter_down_dim_config = copy.deepcopy(config.adapter_config)
            config.use_decoder_enc_attn_key_value_adapter_down_dim_config.use_adapter_down_dim = True
            config.use_decoder_enc_attn_key_value_adapter_down_dim_config.adapter_down_dim = config.decoder_enc_attn_key_value_adapter_down_dim
            self.decoder_enc_attn_key_value_adapter = AdapterController(config.use_decoder_enc_attn_key_value_adapter_down_dim_config)
        else:
            self.decoder_enc_attn_key_value_adapter = None

        self.gating_non_linear = get_activation('gelu_new')
        if config.use_decoder_enc_attn_adapter_gating_large_x_lowrank:
            self.decoder_enc_attn_adapter_gating_large_x_down = nn.Linear(self.embed_dim, config.decoder_enc_attn_adapter_gating_large_x_lowrank_down_dim)
            self.decoder_enc_attn_adapter_gating_large_x_up = nn.Linear(config.decoder_enc_attn_adapter_gating_large_x_lowrank_down_dim, self.embed_dim)
        else:
            self.decoder_enc_attn_adapter_gating_large_x_down = None
            self.decoder_enc_attn_adapter_gating_large_x_up = None

        self.adapter_non_linear = get_activation('gelu_new')
        if config.use_decoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
            self.decoder_adapter_multihead_dim = int(config.adapter_down_dim/config.decoder_adapter_multihead_num_head) # x/h
            self.self_attn_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.decoder_adapter_multihead_dim) 
                for _ in range(config.decoder_adapter_multihead_num_head)
                ])
            self.self_attn_adapter_multihead_up = nn.Linear(config.adapter_down_dim, self.embed_dim)

            self.enc_attn_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.decoder_adapter_multihead_dim) 
                for _ in range(config.decoder_adapter_multihead_num_head)
                ])
            self.enc_attn_adapter_multihead_up = nn.Linear(config.adapter_down_dim, self.embed_dim)

            self.ff_adapter_multihead_down = nn.ModuleList([
                nn.Linear(self.embed_dim, self.decoder_adapter_multihead_dim) 
                for _ in range(config.decoder_adapter_multihead_num_head)
                ])
            self.ff_adapter_multihead_up = nn.Linear(config.adapter_down_dim, self.embed_dim)
        else:
            self.self_attn_adapter_multihead_down = None
            self.enc_attn_adapter_multihead_down = None
            self.ff_adapter_multihead_down = None
            self.self_attn_adapter_multihead_up = None
            self.enc_attn_adapter_multihead_up = None
            self.ff_adapter_multihead_up = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        block_adapters=None,
        task=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple

        if self.use_lora or \
            self.config.use_decoder_self_attn_value_parallel_adapter_down_dim:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=self_attn_past_key_value,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                task=task,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=self_attn_past_key_value,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        if self.self_attn_adapter is not None:
            hidden_states = self.self_attn_adapter(hidden_states, task)
        elif self.decoder_self_attn_adapter is not None:
            hidden_states = self.decoder_self_attn_adapter(hidden_states, task)
        elif self.self_attn_adapter_multihead_down is not None: #【补充：记得做上残差连接】
            if self.config.use_decoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
                adapter_down_multihead_output = [self.self_attn_adapter_multihead_down[i](hidden_states) for i in range(self.config.decoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                adapter_up_multihead_output = self.self_attn_adapter_multihead_up(adapter_down_multihead_output)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

        if self.adapter_hypernet is not None:
            hidden_states = self.adapter_hypernet(hidden_states, block_adapters.self_attention)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            if self.decoder_enc_attn_key_value_adapter is not None:
                encoder_hidden_states = self.decoder_enc_attn_key_value_adapter(encoder_hidden_states, task)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = None
            if past_key_value is not None and len(past_key_value) == 4: # len(past_key_value) is for prefix
                cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            if self.use_lora or \
                self.config.use_decoder_enc_attn_value_parallel_adapter_down_dim or \
                self.config.use_decoder_enc_attn_key_parallel_adapter_down_dim or \
                self.config.use_decoder_enc_attn_value_sequential_adapter_down_dim:
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                    task=task,
                )
            else:
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )

            if self.enc_attn_adapter is not None:
                hidden_states = self.enc_attn_adapter(hidden_states, task)
            elif self.decoder_enc_attn_adapter is not None:
                hidden_states = self.decoder_enc_attn_adapter(hidden_states, task)
                if self.decoder_enc_attn_adapter_gating_large_x_down is not None:
                    gate = self.decoder_enc_attn_adapter_gating_large_x_down(residual)
                    gate = self.gating_non_linear(gate)
                    gate = self.decoder_enc_attn_adapter_gating_large_x_up(gate)
                    gate = torch.sigmoid(gate)
                    hidden_states = hidden_states*gate
            elif self.enc_attn_adapter_multihead_down is not None: #【补充：记得做上残差连接】
                if self.config.use_decoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
                    adapter_down_multihead_output = [self.enc_attn_adapter_multihead_down[i](hidden_states) for i in range(self.config.decoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                    adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                    adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                    adapter_up_multihead_output = self.enc_attn_adapter_multihead_up(adapter_down_multihead_output)

                    hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】


            if self.adapter_hypernet is not None and self.add_adapter_cross_attn:
                hidden_states = self.adapter_hypernet(hidden_states, block_adapters.cross_attention)

            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if self.decoder_ff_ia3 is not None:
            if self.config.use_decoder_ff_ia3_add:
                hidden_states = hidden_states + hidden_states*self.decoder_ff_ia3
            else:
                hidden_states = hidden_states*self.decoder_ff_ia3
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)

        if self.ff_adapter is not None:
            hidden_states = self.ff_adapter(hidden_states, task)
        elif self.decoder_ff_adapter is not None:
            hidden_states = self.decoder_ff_adapter(hidden_states, task)
        elif self.ff_adapter_multihead_down is not None: #【补充：记得做上残差连接】
            if self.config.use_decoder_adapter_down_multihead: # 【补充：h个 768 → x/h，再加上x → 768】
                adapter_down_multihead_output = [self.ff_adapter_multihead_down[i](hidden_states) for i in range(self.config.decoder_adapter_multihead_num_head)] # 【补充：multihead和拼接】
                adapter_down_multihead_output = torch.cat(adapter_down_multihead_output, dim=-1)

                adapter_down_multihead_output = self.adapter_non_linear(adapter_down_multihead_output) # 【补充：nonlinear和up】
                adapter_up_multihead_output = self.ff_adapter_multihead_up(adapter_down_multihead_output)

                hidden_states = hidden_states + adapter_up_multihead_output # 【补充：残差连接】

        if self.adapter_hypernet is not None:
            hidden_states = self.adapter_hypernet(hidden_states, block_adapters.feed_forward)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::
        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    Mask filling example::
        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."
        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits
        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)
        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, task_embed=None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.task_embed = task_embed

        self.adapter_layers_hyper_net = None
        if config.use_hyperformer:
            if config.adapter_config.unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersHyperNetController(config.adapter_config, config.encoder_layers)
            if config.adapter_config.efficient_unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersOneHyperNetController(config.adapter_config, config.encoder_layers)
            
            assert self.adapter_layers_hyper_net is not None

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
        task=None
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        task_embedding = None
        if task is not None and self.task_embed is not None:
            task_embedding = self.task_embed(task)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            block_adapters = None
            if self.adapter_layers_hyper_net:
                block_adapters = self.adapter_layers_hyper_net(task_embedding, idx)

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        past_key_value,
                        block_adapters,
                        task,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, past_key_value, block_adapters, task, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, task_embed=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.task_embed = task_embed

        self.adapter_layers_hyper_net = None
        if config.use_hyperformer:
            if config.adapter_config.unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersHyperNetController(config.adapter_config, config.decoder_layers, True)
            if config.adapter_config.efficient_unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersOneHyperNetController(config.adapter_config, config.decoder_layers, True)
            
            assert self.adapter_layers_hyper_net is not None

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.
                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        task_embedding = None
        if task is not None and self.task_embed is not None:
            task_embedding = self.task_embed(task)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            block_adapters = None
            if self.adapter_layers_hyper_net:
                block_adapters = self.adapter_layers_hyper_net(task_embedding, idx)

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    block_adapters,
                    task,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    block_adapters=block_adapters,
                    task=task,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        if config.use_hyperformer:
            self.shared_task_embed = TaskEmbeddingController(config.adapter_config)
        else:
            self.shared_task_embed = None

        self.encoder = BartEncoder(config, self.shared, self.shared_task_embed)
        self.decoder = BartDecoder(config, self.shared, self.shared_task_embed)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)

        if self.output_adapter is not None:
            self.output_adapter.resize_output_dim(new_num_tokens)

        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "task" in kwargs:
            output["task"] = kwargs["task"]

        return output

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BART_START_DOCSTRING,
)
class BartForQuestionAnswering(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


if __name__ == "__main__":
    import transformers

    model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    generation_output = model.generate(**inputs)

    print(generation_output)

    print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))