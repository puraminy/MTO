# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """

import copy
import math
import os
import warnings
from regex import E

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import numpy as np

from transformers.activations import ACT2FN
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    ModelOutput
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_t5 import T5Config

#### My import
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from attempt.encoders.encoders import * 
from attempt.adapters import AdapterController
from typing import Dict, Any
import attempt.mylogs as mylogs
from attempt.callbacks import WBCallback
from torch.nn.utils.rnn import pad_sequence
import random
######### 
# My utility function
############
def normalize_scores(scores, method="soft", 
        sel_thresh=None, 
        gen_thresh_min=None, 
        gen_thresh_max=None, 
        resample=False, is_training=False):

    if method == "rb" or resample is True:
        scores = RelaxedBernoulli(temperature=gen_thresh_min or 0.0001, 
            logits=scores).rsample()  
    if method == "rbsign": 
        scores = RelaxedBernoulli(temperature=0.001, 
            logits=scores).rsample()  
        scores[scores < 0.5] = 0
        scores[scores >= 0.5] = 1

    if sel_thresh is not None:
        if method == "srelu":
            scores[scores < sel_thresh] = -100 #TODO check if it must be applied or not
        elif method == "nothing":
            scores[scores < sel_thresh] = 0

    if gen_thresh_max is not None:
        mylogs.bp("gmax")
        scores[scores > gen_thresh_max] = gen_thresh_max

    if gen_thresh_min is not None:
        if method == "srelu":
            scores[scores < gen_thresh_min] = -100
        elif method == "nothing":
            scores[scores < gen_thresh_min] = 0

    mylogs.bp("col_sums")
    col_sums = torch.sum((scores > 0)*scores, dim=0)
    if method == "importance1":
        scores = scores * col_sums
    if method == "importance2":
        scores = F.softmax(scores, -1)
        scores = scores * col_sums
    if method == "importance3":
        scores = scores * col_sums
        scores = F.softmax(scores, -1)
    if method == "max":
        max_values, _ = torch.max(scores, dim=2, keepdim=True)
        mask = torch.eq(scores, max_values)
        scores = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))
    elif method == "zero":
        scores[True] = 0
    elif method == "one": # or len(torch.nonzero(scores)) == 0:
        scores[True] = 1
    elif method == "nothing":
       pass 
    elif method == "direct" or method == "soft" or method == "srelu":
        if is_training:
            scores=scores / scores.sum(dim=-1, keepdim=True) 
            # Add a small positive constant to the scores
            epsilon = 1e-8  # Adjust the value of epsilon as needed
            scores += epsilon
        scores = F.softmax(scores, -1)
    elif method == "tanh":
        scores = torch.tanh(scores)
    elif method == "ssig":
        shifted_values = scores + 1.5
        scaled_values = 10 * (shifted_values / 3) - 3
        scores = torch.sigmoid(scaled_values)  
    elif method == "colsoft":
        scores = F.softmax(scores, 0)
    elif method == "sigmoid":
        scores = torch.sigmoid(scores)  
    elif method == "sign":
        scores[scores <= 0] = 0
        scores[scores > 0] = 1
    elif method == "relu":
        scores[scores <= 0] = 0
    elif method == "spos":
        scores = F.softmax(scores, -1)
        scores[scores > 0.3] = 1
    elif method == "sigmoid_relu":
        ret_scores = torch.sigmoid(scores)  
        ret_scores[scores <= 0] = 0
        scores = ret_scores
    elif method == "soft_relu":
        ret_scores = F.softmax(scores, -1)
        ret_scores[scores <= 0] = 0
        scores = ret_scores
    elif method == "norm":
        scores=scores / scores.sum(dim=-1, keepdim=True) 
    elif method == "normsoft":
        scores=scores / scores.sum(dim=-1, keepdim=True) 
        scores = F.softmax(scores, -1)
    elif method == "minmax":
        _min, _ = torch.min(scores, dim=-1, keepdim=True)
        _max, _ = torch.max(scores, dim=-1, keepdim=True)
        scores = (scores - _min) / (_max - _min)
    

    return scores

def batched_topk(batch_size, scores, num_attend_to, sorted=False, threshold=None):
    limit = scores.size(-1)
    if threshold is None: 
        k = num_attend_to
        k = min(k, limit)
        k = max(k, 1)
        return torch.topk(scores, k, sorted=sorted)

    # Initialize empty lists to store results
    selected_scores_list = []
    selected_indices_list = []
    for i in range(batch_size):
        k = num_attend_to
        if threshold is not None: 
            num_above = (scores[i] > threshold).sum().item()
            k = min(num_attend_to, num_above)
        k = min(k, limit)
        k = max(k, 1)
        sel_scores, sel_idx = torch.topk(scores[i], k, sorted=sorted)

        # Append results to the lists
        selected_scores_list.append(sel_scores.T)
        selected_indices_list.append(sel_idx.T)

    # Convert lists to tensors
    selected_scores = pad_sequence(selected_scores_list, 
            batch_first=True, padding_value=0.0).permute(0,2,1)
    selected_indices = pad_sequence(selected_indices_list, 
            batch_first=True, padding_value=0).permute(0,2,1)

    return selected_scores, selected_indices

def batched_index_select(inp, dim, index):
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer",
                  "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(
                f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(
        f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],

                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],

                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""

class Anneal:
    def __init__(self, temperature, anneal_dir, anneal_rate, anneal_min, anneal_type="exp"):
        self.cur_step = 0
        if anneal_dir == 1:
            self.start_val = anneal_min
            self.anneal_point = temperature
        elif anneal_dir == -1:
            self.start_val = temperature
            self.anneal_point = anneal_min

        self.cur_val = self.start_val
        self.anneal_dir = anneal_dir
        self.anneal_rate = anneal_rate
        self.anneal_type = anneal_type

    def __iter__(self):
        return self

    def anneal(self, i_step):
        if self.anneal_dir == -1 and self.cur_val <= self.anneal_point:
            return self.anneal_point
        if self.anneal_dir == 1 and self.cur_val >= self.anneal_point:
            return self.anneal_point
        delta = self.anneal_dir * self.anneal_rate * i_step
        if self.anneal_type == "exp":
           exp_rate = np.exp(delta)
           t = self.cur_val * exp_rate
        else:
           t = self.start_val + delta 
        if self.anneal_dir == -1:
            self.cur_val = max(t, self.anneal_point)
        else:
            self.cur_val = min(t, self.anneal_point)
        return self.cur_val 

    def __next__(self):
        self.cur_step += 1 
        value = self.anneal(self.cur_step)
        return value

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, adapter_config=None):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bitfit = adapter_config.bitfit if adapter_config is not None else False
        if self.bitfit:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        result = self.weight * hidden_states
        if self.bitfit:
            result = result + self.bias
        return result


class T5DenseReluDense(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.bitfit = adapter_config.bitfit if adapter_config is not None else False
        self.wi = nn.Linear(config.d_model, config.d_ff,
                            bias=False if not self.bitfit else True)
        self.wo = nn.Linear(config.d_ff, config.d_model,
                            bias=False if not self.bitfit else True)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.bitfit = adapter_config.bitfit if adapter_config is not None else False
        self.wi_0 = nn.Linear(config.d_model, config.d_ff,
                              bias=False if not self.bitfit else True)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff,
                              bias=False if not self.bitfit else True)
        self.wo = nn.Linear(config.d_ff, config.d_model,
                            bias=False if not self.bitfit else True)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(
                config, adapter_config=adapter_config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(
                config, adapter_config=adapter_config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        self.train_task_adapters = config.train_task_adapters and adapter_config.add_adapter_in_feed_forward
        if self.train_task_adapters:
            adapter_config.reduction_factor = adapter_config.task_reduction_factor
            self.adapter_controller = AdapterController(adapter_config)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, adapter_config=adapter_config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, task_block_adapters=None, task=None):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        if self.train_task_adapters:
            forwarded_states = self.adapter_controller(forwarded_states, task)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.bitfit = adapter_config.bitfit if adapter_config is not None else False
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        # mmmmmmm
        self.adapter_config = adapter_config

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim,
                           bias=False if not self.bitfit else True)
        self.k = nn.Linear(self.d_model, self.inner_dim,
                           bias=False if not self.bitfit else True)
        self.v = nn.Linear(self.d_model, self.inner_dim,
                           bias=False if not self.bitfit else True)
        self.o = nn.Linear(self.inner_dim, self.d_model,
                           bias=False if not self.bitfit else True)

        #   My code
        # self.attn_W_down = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.attn_W_up = nn.Linear(self.inner_dim, self.d_model, bias=False)
        # self.attn_non_linear = nn.SiLU()
        # self.layer_norm = nn.LayerNorm(self.d_model)
        # End my Code

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = getattr(
            config, "gradient_checkpointing", False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position >
                                 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = - \
                torch.min(relative_position,
                          torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(
                relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small,
                                        relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - \
            context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device)
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        int_seq_length = int(seq_length)

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[
            1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat(
                        [past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        #### Mycod qqqqqqqqqqqqqqqqqqqq
        # mylogs.bp("query_fwd")
        #x = self.attn_W_down(hidden_states)
        #x = self.attn_non_linear(x)
        #x = self.attn_W_up(x)
        #x = self.layer_norm(x)

        if False: #not self.is_decoder:
            prompt_masks = self.adapter_config.prompt_masks
            soft_prompts = self.adapter_config.soft_prompts
            hidden_states[prompt_masks]= soft_prompts.view(-1, self.d_model)
        #########
        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = shape(self.q(hidden_states))

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[
                0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[
                1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -int_seq_length:, :]

            if mask is not None:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # (batch_size, seq_length, dim)
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (
            self.is_decoder and use_cache) else None
        outputs = (attn_output,) + \
            (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.adapter_config = adapter_config
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, adapter_config=adapter_config)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, adapter_config=adapter_config)
        self.train_task_adapters = config.train_task_adapters and adapter_config.add_adapter_in_self_attention
        if self.train_task_adapters:
            adapter_config.reduction_factor = adapter_config.task_reduction_factor
            self.adapter_controller = AdapterController(adapter_config)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        task_block_adapters=None,
        task_ids=None,
        task=None
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        mylogs.bp("adapt")
        y = attention_output[0]
        if self.train_task_adapters:
            y = self.adapter_controller(y, task)
        if False: #not self.is_decoder: #TODO self.adapter_config.attn_tuning:
            pmask = self.adapter_config.prompt_masks
            s = pmask.size()
            y = torch.zeros((s[0],s[1], 768), device=y.device)
            y2 = self.adapter_config.soft_prompts
            y[pmask] = y2 

        hidden_states = hidden_states + self.dropout(y)
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False, adapter_config=adapter_config)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, adapter_config=adapter_config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias, adapter_config=adapter_config))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(
                config, adapter_config=adapter_config))

        self.layer.append(T5LayerFF(config, adapter_config=adapter_config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        task_block_adapters=None,
        task_ids = None,
        task=None
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention' if expected_num_past_key_values == 4 else ''}."
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            task_block_adapters=task_block_adapters,
            task_ids=task_ids,
            task=task
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        # Keep self-attention outputs and relative position weights
        attention_outputs = self_attention_outputs[2:]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + \
                    cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states,
                                       task_block_adapters=task_block_adapters,
                                       task=task)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        return outputs


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs
        

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'prefix_shared'):
                if module.prefix_shared is not None:
                    self.init_prefix_weights()
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(
                mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(
                mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5))
        ##################################################
        elif isinstance(module, nn.Linear):
            # This is for adapter layers.
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        ##################################################

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(
        ), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, adapter_config=None, prefix_emb=None, attn_tuning=False, mul_prefix_emb=None, model_dim=768, attn_method="linear", shared_attn=False, attend_target=False, temperature=2000, learned_temperature=False):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        ################# MyCode
        self.prompt_encoders = []
        self.use_private_prompts = False
        self.embedding_dim = self.config.hidden_size
        # all prompt ids for all tasks
        self.target_prompt_ids = []
        self.common_prompt_ids = [] 
        self.task_prompt_ids = []
        self.router = None
        self.init_router = None
        self.training = True
        self.pred_task = ""
        self.prompt_dim = None
        self.gen_conf = None
        self.route_method = config.route_method
        self.learn_attention = config.learn_attention
        self.normalize = config.normalize
        self.sig_coef = config.sig_coef
        self.prompt_tuning = config.prompt_tuning
        self.attn_prompt_tuning = config.attn_tuning
        self.use_source_prompts = config.use_source_prompts
        self.ignore_private = config.ignore_private
        self.attend_input = config.attend_input
        self.add_input = config.add_input
        self.add_target = config.add_target
        self.random_source = config.random_source
        self.compose_method = config.compose_method
        self.compose_target = config.compose_target
        self.select_method = config.select_method
        self.target_share_temperature = config.target_share_temperature
        self.bias = config.bias
        self.anneal_min = config.anneal_min
        self.anneal_dir = config.anneal_dir
        self.anneal_rate = config.anneal_rate
        self.anneal_type = config.anneal_type
        self.temperature = temperature

        self.anneal_router = Anneal(self.temperature, 
                anneal_dir = self.anneal_dir, 
                anneal_rate = self.anneal_rate, 
                anneal_min = self.anneal_min, 
                anneal_type=self.anneal_type)

        self.sel_thresh = config.sel_thresh
        self.thresh_anneal_dir = 1
        self.thresh_anneal_type = "linear"
        self.thresh_anneal_rate = 0.002
        self.thresh_anneal_min = 0.1
        self.do_anneal_thresh = False
        self.anneal_thresh = Anneal(self.sel_thresh, 
                anneal_dir = self.thresh_anneal_dir, 
                anneal_rate = self.thresh_anneal_rate, 
                anneal_min = self.thresh_anneal_min, 
                anneal_type=self.thresh_anneal_type)
        self.anneal_ts = Anneal(self.target_share_temperature, 
                anneal_dir = -1, anneal_rate = 0.05, 
                anneal_min = 0, anneal_type="linear")
        self.norm_method = config.norm_method
        # self.learn_source_prompts = config.learn_source_prompts
        ##############################################
        self.attend_target = attend_target
        self.use_private_prompts = config.use_private_prompts
        self.num_target_prompts = config.num_target_prompts
        self.target_share = config.target_share 
        self.attend_for = config.attend_for 
        self.attend_private = config.attend_private 
        self.prefix_emb = prefix_emb if self.attend_target is True else None
        self.prefix_tuning = config.prefix_tuning
        self.source_prompts_order = config.source_prompts_order
        self.padding_pos = config.padding_pos
        self.attn_tuning = attn_tuning
        self.mul_prefix_emb = mul_prefix_emb
        self.attn_method = attn_method
        self.model_dim = model_dim
        self.out_dim = config.prompt_out_dim if config.prompt_out_dim > 0 else model_dim
        self.shared_attn = shared_attn
        self.learned_temperature = learned_temperature
        self.target_task_id = None
        self.task_names = None
        if self.learned_temperature is True:
            # The code causes error; need to fix a bug.
            # RuntimeError: Trying to backward through the graph a second time (or directly access saved variables after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved variables after calling backward
            self.temperature = (self.model_dim * torch.exp(torch.clamp(nn.Parameter(
                torch.Tensor([1]), requires_grad=True), min=0.005, max=5))).cuda()
        self.append_prefix = self.prefix_tuning and not self.is_decoder and not self.attn_tuning
        self.append_attn_prefix = self.prefix_tuning and not self.is_decoder and self.attn_tuning
        if self.prefix_tuning:
            self.prefix_dim = adapter_config.prefix_dim
        if self.append_attn_prefix: # or self.prompt_tuning:
            if self.attn_method == "linear" or self.compose_method == "lin":
                self.attn_Wa = nn.Linear(
                    self.model_dim, self.model_dim, bias=False)
                self.layer_norm = nn.LayerNorm(self.model_dim)
            if self.attn_method == "sub" or self.compose_method == "sub":
                self.attn_W_down = nn.Linear(self.model_dim, 300, bias=False)
                self.attn_W_up = nn.Linear(300, self.model_dim, bias=False)
                self.attn_non_linear = nn.SiLU()
                self.layer_norm = nn.LayerNorm(self.model_dim)
        #######################################
        self.adapter_config = adapter_config
        self.block = nn.ModuleList(
            [T5Block(self.per_layer_config(config, i, self.adapter_config, self.is_decoder),
                     has_relative_attention_bias=bool(i == 0),
                     adapter_config=adapter_config) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_encoders(self, prompt_encoders, source_prompts, 
            src_prompt_dim, prompt_dim, tasks = None):
        self.task_names = tasks
        mylogs.bp("set")
        self.prompt_encoders = torch.nn.ModuleList(prompt_encoders)
        src_tgt_encoders = [e for e in self.prompt_encoders if e.is_source or e.is_target]
        self.prompt_dim = prompt_dim[0] if type(prompt_dim) == list else prompt_dim
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attend_num =len(src_tgt_encoders) + 1 # one for input
        self.attn_scores = torch.zeros(
            (attend_num, attend_num), device=device) 
        self.attn_mask_learned = torch.zeros(
            (attend_num, attend_num), device=device) 
        self.src_prompt_dim = src_prompt_dim
        self.prompt_names = ["input"] + [x.name for x in src_tgt_encoders]
        self.num_src_encoders = 0
        if source_prompts:
            self.num_src_encoders = len(source_prompts) + 1 # one for input 

        target_prompt_ids = []
        task_prompt_ids = []
        common_prompt_ids = []
        self.attn_mask = torch.zeros(attend_num, attend_num, device=device)
        src_list = []
        tgt_list = []
        i = 1
        for encoder in self.prompt_encoders:
            if encoder.is_common:
                common_prompt_ids.extend(encoder.prompt_ids)
            elif encoder.is_target:
                target_prompt_ids.extend(encoder.prompt_ids)
            else:
                task_prompt_ids.extend(encoder.prompt_ids)
            encoder.to(device)
            if source_prompts and encoder.name in source_prompts:
                encoder.src_idx = i
                src_list.append(i)
                i += 1
                continue
            mylogs.bp("mask")
            if encoder.is_target:
                tgt_list.append(i)
                self.attn_mask[i, :] = torch.tensor(encoder.attend_to_mask, device=device)
                i += 1

        self.attn_mask_learned[:] = self.attn_mask 
        if self.router is None:
            #router = nn.Parameter(data=torch.empty((
            #        attend_num,
            #        attend_num 
            #    ), device=device).uniform_(-1e-3, 1e-3))
            router = torch.zeros((attend_num, attend_num), device=device)
            route_method = self.route_method
            if self.bias is not None and self.bias > 0:
                i,j,k = 1,1,1
                first = True
                mylogs.bp("bias")
                if type(self.bias) == list:
                    names = [x.split("-")[0] for x in self.bias]
                    pos = [x.split("-")[1] for x in self.bias]
                    values = [x.split("-")[2] for x in self.bias]
                for encoder in self.prompt_encoders:
                    if encoder.is_private and first:
                        k = i
                        first = False
                    elif encoder.is_target:
                        if type(self.bias) == list:
                            encname = encoder.name.split("-")[1]
                            if encname in names: 
                                index = names.index(encname)
                                _pos = pos[inex]
                                if _pos == "s":
                                    router[i, j] = float(values[index])
                                    j += 1
                        else:
                            _pos, b = "s", self.bias
                            if type(self.bias) == str and "-" in self.bias:
                                _pos, b = self.bias.split("-")
                            if _pos == "x" or _pos == "s":
                                router[i, j] = float(b)
                                j += 1
                        if k > 1:
                            _pos, b = "s", self.bias
                            if type(self.bias) == str and "-" in self.bias:
                                _pos, b = self.bias.split("-")
                            if _pos == "x" or _pos == "p":
                                router[i, k] = float(b) 
                                k += 1
                    i += 1
                mylogs.bp("bias")
            self.router = nn.Parameter(data=router)

        self.attn_mask_orig = self.attn_mask.clone()
        self.source_encoders_idx = torch.tensor(src_list, device=device)
        self.target_encoders_idx = torch.tensor(tgt_list, device=device)

        self.target_prompt_ids = torch.tensor(target_prompt_ids, device=device)
        self.common_prompt_ids = torch.tensor(common_prompt_ids, device=device)
        self.task_prompt_ids = torch.tensor(task_prompt_ids, device=device)
        intrinsic_dim = 200
        self.target_router = nn.Parameter(data=torch.empty((
            attend_num
        ), device=device).uniform_(0, 0))


        if self.prompt_tuning:
            mylogs.bp("sub")
            mylogs.bp("lin")
            # inp_dim = len(source_prompts) * self.src_prompt_dim * self.model_dim 
            inp_dim = self.model_dim 
            # out_dim = self.src_prompt_dim * self.model_dim 
            out_dim = self.model_dim 
            embedding_size = self.model_dim
            num_source_prompts = len(source_prompts)
            hidden_size = num_source_prompts * self.src_prompt_dim * 200 
            # self.conv_layer = nn.Conv1d(in_channels=num_source_prompts, 
            #        out_channels=4, kernel_size=len(source_prompts)*self.model_dim)
            if self.compose_method == "lin":
                # Embedding layers for source prompts
                # self.source_embedding = nn.Embedding(num_source_prompts, embedding_size)
                # Neural network for parameterizing the combination function
                self.comp_linear = nn.Sequential(
                    nn.Linear(
                    num_source_prompts * self.src_prompt_dim * embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 
                        self.num_target_prompts * self.src_prompt_dim * embedding_size)
                )
                self.attn_Wa = nn.Linear(
                    inp_dim, out_dim, bias=False)
                self.layer_norm = nn.LayerNorm(inp_dim)
            if self.compose_method == "sub":
                self.attn_W_down = nn.Linear(inp_dim, 1000, bias=False)
                self.attn_W_up = nn.Linear(1000, inp_dim, bias=False)
                self.attn_non_linear = nn.SiLU()

            self.layer_norm = nn.LayerNorm(inp_dim)
#        self.z = nn.Parameter(data=torch.empty((
#            attend_num, 
#            intrinsic_dim
#        )).uniform_(-1e-3, 1e-3))
#
#        bound = 1 / math.sqrt(prompt_dim * self.model_dim)
#        self.A = nn.Parameter(data=torch.empty((
#            intrinsic_dim,
#            prompt_dim * self.model_dim 
#        )).uniform_(-bound, bound))
#
    def make_attn_mask(self, index=0, num_masked_prompts = 1, mask_type="rand"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attend_num = len(self.prompt_encoders) + 1 # one for input
        base = num_masked_prompts / attend_num
        nse = sum(1 for enc in self.prompt_encoders  
                  if enc.is_source and not enc.is_private and not enc.is_target)
                # if enc.is_source and not enc.is_target)
        nspe = sum(1 for enc in self.prompt_encoders if not enc.is_target)
        mylogs.bp("nrp")
        attn_mask = self.attn_mask_orig.clone() 
        k = num_masked_prompts
        targets = self.target_encoders_idx
        attn_scores = self.router #.index_select(0, targets)
        if "any" in mask_type:
            selected_indices_per_row = [torch.nonzero(torch.ones_like(row))[:, -1] 
                    for row in attn_scores]
        else:
            selected_indices_per_row = [torch.nonzero(row > 0)[:, -1] for row in attn_scores]
        for i, encoder in enumerate(self.prompt_encoders, start=1):
            if encoder.is_target:
                if mask_type == "rand" or mask_type == "random":
                    r = torch.rand((1, nse -1), device=device)
                    k_th_quant = torch.topk(r, k, largest = False)[0][:,-1:]
                    mask = r <= k_th_quant
                    attn_mask[i, 1:nse] = mask.long()
                elif (mask_type.startswith("remove")
                    or mask_type.startswith("keeponly")):
                    if mask_type.startswith("remove"):
                        attn_mask[i, 1:nse+1] = 1
                    else:
                        attn_mask[i, 1:nse+1] = 0
                    if index <=  len(selected_indices_per_row[i]):
                        to = min(nse + 1, (index -1) + num_masked_prompts)
                        to = min(to, len(selected_indices_per_row[i]))
                        idx = min(index -1, to) 
                        indices = selected_indices_per_row[i][idx:to]
                        if mask_type.startswith("remove"):
                            attn_mask[i, indices] = 0 
                        else:
                            attn_mask[i, indices] = 1 
                elif mask_type == "keep_private":
                    attn_mask[i, 1:] = 0
                    attn_mask[i, nse+1:nspe+1] = self.attn_mask_orig[i, nse+1:nspe+1]
                elif mask_type == "rem_private":
                    attn_mask[i, nse+1:nspe+1] = 0 
                elif mask_type == "keep_target":
                    attn_mask[i, 1:nspe+1] = 0
                elif mask_type == "rem_target":
                    attn_mask[i, nspe+1:] = 0
                elif mask_type == "keep_source":
                    attn_mask[i, 1:] = 0
                    attn_mask[i, 1:nse+1] = 1
                elif mask_type == "rem_source":
                    attn_mask[i, 1:nse+1] = 0
                elif mask_type == "keep_input":
                    attn_mask[i, :] = 0
                    attn_mask[i, 0] = 1
                elif mask_type == "rem_input":
                    attn_mask[i, 0] = 0
                else:
                    to = min(nse + 1, index + num_masked_prompts)
                    if mask_type == "rem":
                        # attn_mask[i, 1:nse+1] = 1
                        attn_mask[i, index:to] = 0 
                    elif mask_type == "keep":
                        attn_mask[i, 1:] = 0
                        attn_mask[i, index:to] = self.attn_mask_orig[i, index:to] 
        return attn_mask.long()

    def anneal(self, i_step):
         mylogs.bp("anneal")
         self.temperature = self.anneal_router.anneal(i_step)
         if self.do_anneal_thresh is True:
             self.sel_thresh = self.anneal_thresh.anneal(i_step)

    ################# MyCode fffffffffff
    def attend_input(self, inputs_embeds, src_prompts, 
            target_prompts, add_target, source_idx=None, 
            target_idx =None, task_ids=None, task=""):
        batch_size = inputs_embeds.shape[0]
        attend_for = target_prompts
        inp_target = target_prompts
        if self.attend_for == "target": 
            inp_target = target_prompts
        elif self.attend_for == "inp_target": 
            #pool = torch.nn.AdaptiveMaxPool1d(self.src_prompt_dim)
            target = target_prompts.squeeze(1)
            inp_target = torch.cat([inputs_embeds, target], dim=1)
            #inp_target = inp_target.permute(0,2,1)
            #inp_target = pool(inp_target).permute(0,2,1)
            inp_target = inp_target.unsqueeze(1)
        elif self.attend_for == "input": 
            inp_target = inputs_embeds 
            inp_target = inp_target.unsqueeze(1)
        avg_attend_to, _ = torch.max(attend_to, 2)
        avg_attend_for, _ = torch.max(inp_target, 2)
        if self.attn_method == "dot":
            x = torch.transpose(avg_attend_to, 1,2)
            attn_scores = avg_attend_for.bmm(x)
        elif self.attn_method == "linear":
            x = self.attn_Wa(avg_attend_to)
            x = self.layer_norm(x)
            x = torch.transpose(x, 1,2)
            attn_scores = avg_attend_for.bmm(
                x) / self.temperature

        elif self.attn_method == "sub":
            x = self.attn_W_down(avg_attend_to)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)
            #x = x.unsqueeze(-1) ###
            x = torch.transpose(x, -2, -1)
            attn_scores = avg_attend_for.bmm(
                x) / self.temperature

        # implement token level model
        elif self.attn_method == "token":
            x = self.attn_W_down(avg_attend_to)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)
            x = x.unsqueeze(-1)
            attn_scores = torch.einsum(
                "bpld,bdk->bplk", attend_for, x) / self.temperature
        elif self.attn_method == "constant":
            # FIXME: more efficient implementation
            attn_scores = (torch.ones(attend_for.size(1), 
                attend_to.size(1), device=inputs_embeds.device) / attend_to.size(1))
            attn_scores = attn_scores.repeat(batch_size, 1, 1)
        else:
            raise NotImplementedError

        return attn_scores 

    def attend_prompts(self, inputs_embeds, src_prompts, 
            source_idx=None, num_targets=1, 
            target_idx =None, task_ids=None, attn_mat=None, task=""):
        #avg_inputs_embeds, _ = torch.max(inputs_embeds, 1)
        #pool = torch.nn.AdaptiveAvgPool1d(self.promt_dim)
        mylogs.bp("att")
        if not self.training:
            mylogs.bp("all")

        if not self.training: 
           mylogs.bp("ccc")
           if "keep-source" in self.gen_conf["mask_type"]:
               mylogs.bp("keepsrc")
           elif "keep-" in self.gen_conf["mask_type"]:
               mylogs.bp("keepprompt")
           if self.gen_conf is not None and "attn_mask" in self.gen_conf:
               attn_mask = self.gen_conf["attn_mask"] 

        batch_size = inputs_embeds.shape[0]
        private_prompt = None
        avg_inputs_embeds = None
        if self.attend_input or self.add_input:
            pool2 = torch.nn.AdaptiveMaxPool1d(self.src_prompt_dim)
            avg_inputs_embeds = pool2(inputs_embeds.permute(0,2,1)).permute(0,2,1)
        if self.use_private_prompts:
            private_prompt = src_prompts[:,-1,:,:]
            if self.attend_for == "private": 
                inp_target = private_prompt.unsqueeze(1)
                attend_to = src_prompts[:,:-1,:,:]
        if self.attend_input:
            #avg_inputs_embeds = avg_inputs_embeds.unsqueeze(1)
            src_prompts[:,0,:,:] = avg_inputs_embeds
            attend_to = src_prompts
        else:
            attend_to = src_prompts[:,1:,:,:]

        device=inputs_embeds.device
        attn_scores = None
        attend_to_idx = source_idx
        if not self.attend_input:
            attend_to_idx = source_idx[:,1:]

        compose_method = self.compose_method
        if not self.training:
            if "gen_cmm" in self.gen_conf and self.gen_conf["gen_cmm"] is not None: 
                compose_method = self.gen_conf["gen_cmm"]

        if compose_method in ["wcp1","wsp1","wmp1"]: # or self.ignore_private:
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompt = attend_to[:,-1,:,:]
            private_prompt = private_prompt.unsqueeze(1)
            attend_to_idx = attend_to_idx[:,:-1] # skip private prompts
            attend_to = attend_to[:,:-1,:,:]
        if compose_method in ["cat"] and self.add_target: # or not self.attend_private:
            mylogs.bp("attcat")
            pass
            #last_prompt = attend_to[:,-1,:,:]
            #last_prompt = last_prompt.unsqueeze(1)
            #attend_to_idx = attend_to_idx[:,:-1] # skip private prompts
            #attend_to = attend_to[:,:-1,:,:]

        # Bernouli 
        route_method = self.route_method
        gen_norm_method = self.norm_method
        if self.attn_method == "const":
            route_idx = attend_to_idx
            router = torch.ones(target_idx.size()[1],
                    route_idx.size()[1], 
                    device=inputs_embeds.device)
            router = router.repeat(batch_size, 1, 1)
            attn_scores = router
        elif self.attn_method == "rb":
            mylogs.bp("rmconst")
            route_idx = attend_to_idx
            router = torch.zeros(target_idx.size(1),
                    route_idx.size(1), 
                    device=inputs_embeds.device)
            router = router.repeat(batch_size, 1, 1)
            for i in range(batch_size):
                router[i] = self.router[target_idx[i].reshape(-1,1), 
                                    route_idx[i]]
            attn_dist = torch.ones_like(router)
            if route_method == "const":
                attn_dist = 0*attn_dist
                b = 1
            else:
                attn_dist = -1*attn_dist
                b = next(self.anneal_ts)

            #end = attn_dist.size(2)
            #max_task_num = torch.max(task_ids).item()
            #if max_task_num < end:
            #    for i in range(batch_size):
            #        task_id = task_ids[i].item()
            #        attn_dist[i, :, task_id] = b 

            if self.training: # and self.learn_attention:
                logits = router
                mylogs.bp("rbsample")
                rb_scores = RelaxedBernoulli(temperature=self.temperature, 
                    logits=logits).rsample()            
                if route_method == "params":
                    attn_scores = router
                elif route_method == "const":
                    attn_scores  = attn_dist
                    self.norm_method = "nothing"
                elif route_method == "importance":
                    col_sums = torch.sum(router, dim=0)
                    attn_scores = rb_scores * col_sums
                else:
                    attn_scores = rb_scores # + attn_dist
            elif not self.training:
                mylogs.bp("route")
                #attn_scores = router
                #attn_scores = torch.sigmoid(attn_scores)  # layer * n_prompts
                if route_method == "const":
                    attn_scores  = attn_dist
                    self.norm_method = "nothing"
                elif route_method == "importance":
                    col_sums = torch.sum(router, dim=0)
                    attn_scores = router * col_sums
                else:
                    attn_scores = router

            #z = torch.mm(self.z, self.A) 
            #soft_prompts = torch.matmul(router.unsqueeze(0), z).view(-1, self.model_dim).tile(batch_size, 1, 1)

        mylogs.bp("before")
        if self.training and "before" in self.norm_method and self.attn_method != "const":
            method = self.norm_method.replace("before_","")
            attn_scores = normalize_scores(attn_scores, method, is_training=self.training) 

        #if compose_method in ["cat","concat","catw"]: #,"pool","mpool"]:
        #    num_attend_to = (num_targets * attend_for.size(2)) // self.src_prompt_dim
        #    num_attend_to = num_attend_to // num_targets
        #else:
        num_attend_to = self.num_target_prompts

        if not self.training and "gen_ntp" in self.gen_conf:
            num_attend_to = self.gen_conf["gen_ntp"]

        if False: #self.attend_target or self.attend_private: # force to select them
            attn_scores[:,:,-1] = attn_scores[:,:,-1]+ 2

        mylogs.bp("tk1")
        mylogs.bp(task + "1")
        if not self.training:
            mylogs.bp("tk2")
            mylogs.bp(task + "2")

        if not "pool" in compose_method and not "lin" in compose_method:
            num_select = num_attend_to
        else:
            num_select = attn_scores.size(-1) # select all

        mylogs.bp("negg")
        sorting_opts = ["sorted", "sorted_asc","sorted_desc"]
        attn_sel_scores, attend_to_x = attn_scores, attend_to 
        if (num_select < attn_scores.size(-1) 
            or self.source_prompts_order in sorting_opts):
            attn_sel_scores, attend_to_sel_idx = batched_topk(batch_size,
                    attn_scores, 
                    num_select, 
                    sorted=self.source_prompts_order in sorting_opts,
                    threshold=None) #  self.sel_thresh)
            if self.source_prompts_order == "rand":
                idx = torch.randperm(attend_to_sel_idx.shape[-1])
                attend_to_sel_idx = attend_to_sel_idx[:,:,idx].view(attend_to_sel_idx.size())
                attn_sel_scores = attn_sel_scores[:,:,idx].view(attn_sel_scores.size())
            elif self.source_prompts_order == "sorted_asc": #TODO it doesn't work
                attend_to_sel_idx = torch.flip(attend_to_sel_idx, dims=(-1,))
                attn_sel_scores = torch.flip(attn_sel_scores, dims=(-1,))

            if False: #self.attend_target or self.attend_private: # force to select them
                attn_sel_scores[attn_sel_scores >= 2] = attn_sel_scores[attn_sel_scores >= 2]- 2
            attend_to_idx = batched_index_select(attend_to_idx, 1, attend_to_sel_idx)

            # if not self.attend_input:
            #    attend_to_sel_idx = attend_to_sel_idx + 1

            mylogs.bp("params")
            # Create a binary mask for the top k indices
            if route_method == "params":
                # top_k_mask = torch.zeros_like(attn_scores)
                # top_k_mask.scatter_(-1, attend_to_sel_idx, 1)
                # attn_sel_scores = attn_score * top_k_mask
                pass

            # Apply the mask to select the top k prompts
            # top_k_mask = top_k_mask.squeeze(1).unsqueeze(-1).unsqueeze(-1)
            # attend_to = attend_to.view(batch_size, attn_scores.shape[-1], -1)  
            # attend_to_1 = attend_to * top_k_mask
            # attend_to_1 = attend_to_1.view(batch_size, num_targets, -1, 
            #        self.src_prompt_dim, self.model_dim)

            attend_to_x = batched_index_select(attend_to, 1, attend_to_sel_idx)

        attend_to_x = attend_to_x.view(batch_size, num_targets, -1, 
                self.src_prompt_dim, self.out_dim)
        if route_method == "params":
            # attend_to_x = attend_to
            # attend_to_x = attend_to_x.unsqueeze(1)
            pass

        if not self.training:
            gen_thresh_min = None 
            gen_thresh_max = None
            if self.gen_conf is not None and "gen_norm_method" in self.gen_conf:
                gen_norm_method = self.gen_conf["gen_norm_method"] 
            if self.gen_conf is not None and "gen_thresh_min" in self.gen_conf:
                gen_thresh_min = self.gen_conf["gen_thresh_min"] 
            if self.gen_conf is not None and "gen_thresh_max" in self.gen_conf:
                gen_thresh_max = self.gen_conf["gen_thresh_max"] 
            mylogs.bp("gn-"+ gen_norm_method)
            mylogs.bp("norm")
            if attn_mat is not None:
                mylogs.bp("amat")
                attn_idx = attend_to_idx
                for i in range(batch_size):
                    attn_sel_scores[i] = attn_mat[target_idx[i].reshape(-1,1), 
                                        attn_idx[i]]
            else:
                attn_sel_scores = normalize_scores(attn_sel_scores, 
                    gen_norm_method,
                    gen_thresh_min=gen_thresh_min,
                    gen_thresh_max=gen_thresh_max, is_training=self.training)

        mylogs.bp("norm")
        if self.training and self.attn_method != "const":
            method = self.norm_method.replace("after_","")
            attn_sel_scores = normalize_scores(attn_sel_scores, method, 
                    sel_thresh=self.sel_thresh, is_training=self.training)

        mylogs.bp("params")
        if route_method == "params":
            # attn_sel_scores = attn_sel_scores.new_ones(attn_sel_scores.shape)
            # attn_sel_scores = torch.ones_like(attn_sel_scores, requires_grad=True)
            pass

        target_shares = torch.ones(1, batch_size, device=device)

        if self.random_source > 0 and not self.training:
            num_cols = attn_sel_scores.size(-1)  
            num_selected_cols = self.random_source  # Number of random columns to select
            num_selected_cols = min(num_selected_cols, num_cols)
            selected_cols_indices = random.sample(range(num_cols), num_selected_cols)

            attn_sel_scores = attn_sel_scores[:, :, selected_cols_indices]
            attend_to_x = attend_to_x[:, :, selected_cols_indices, :, :]
            attend_to_idx = attend_to_idx[:, selected_cols_indices] 
        
        if self.norm_method == "nothing":
            if self.attn_method == "const":
                assert torch.all(attn_sel_scores == 1), "Attention scores must be all one"
        if compose_method in ["wavg","mwavg"]: 
            if True: #not self.ignore_private:
                soft_prompts = torch.einsum(
                    'bts, btsld -> btld', attn_sel_scores, attend_to_x)
            else:
                s_attn_sel_scores = attn_sel_scores[:,:,:-1]
                s_attend_to_x = attend_to_x[:,:,:-1,:,:]
                assert self.use_private_prompts is True, "use private prompts must be enabled"
                private_prompts = attend_to_x[:,:,-1,:,:]
                soft_prompts = torch.einsum(
                        'bts, btsld -> btld', s_attn_sel_scores, 
                        s_attend_to_x)
        elif compose_method == "rcat":
            soft_prompts = torch.einsum(
                'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            soft_prompts = torch.einsum(
                'bts, btsld -> btld', agg_scores, soft_prompts)
        elif compose_method in ["cat","catw","mcat","scat", "mscat"]:
            mylogs.bp("cat")
            soft_prompts = torch.einsum(
                'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            soft_prompts = soft_prompts.reshape(batch_size, num_targets,-1, self.model_dim) 
        elif compose_method == "concat":
            attn_sel_scores[True] = 1
            soft_prompts = torch.einsum(
                'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            soft_prompts = soft_prompts.reshape(batch_size, num_targets,-1, self.model_dim) 
        elif compose_method in ["wsp1","wmp1","wcp1"]:
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', attn_sel_scores, 
                    attend_to_x)
            if compose_method == "wsp1": 
               soft_prompts = avg_prompts + private_prompt 
            elif compose_method == "wmp1": 
               soft_prompts = avg_prompts * private_prompt 
            elif compose_method == "wcp1": 
               soft_prompts = torch.cat([avg_prompts,private_prompt], dim=2)
            attn_sel_scores = torch.cat(
                   [attn_sel_scores, target_shares.reshape(batch_size, 1, 1)], dim=-1)
            attend_to_idx = torch.cat([attend_to_idx, target_idx], dim=-1) 
        elif compose_method == "wmp":
            mylogs.bp("wmp")
            s_attn_sel_scores = attn_sel_scores[:,:,:-1]
            s_attend_to_x = attend_to_x[:,:,:-1,:,:]
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompts = attend_to_x[:,:,-1,:,:]
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', s_attn_sel_scores, 
                    s_attend_to_x)
            if self.target_share == 2:
               soft_prompts = avg_prompts * private_prompts 
            else:
               ts = attn_sel_scores[:,:,-1]
               ts = ts.reshape(batch_size, 1, 1, 1)
               soft_prompts = avg_prompts * (ts * private_prompts) 
        elif compose_method == "wsp":
            mylogs.bp("wsp")
            s_attn_sel_scores = attn_sel_scores[:,:,:-1]
            s_attend_to_x = attend_to_x[:,:,:-1,:,:]
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompts = attend_to_x[:,:,-1,:,:]
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', s_attn_sel_scores, 
                    s_attend_to_x)
            if self.target_share == 2:
               soft_prompts = avg_prompts + private_prompts 
            else:
               ts = attn_sel_scores[:,:,-1]
               ts = ts.reshape(batch_size, 1, 1, 1)
               soft_prompts = avg_prompts + (ts * private_prompts) 
        elif compose_method == "wcp":
            mylogs.bp("wcp")
            s_attn_sel_scores = attn_sel_scores[:,:,:-1]
            s_attend_to_x = attend_to_x[:,:,:-1,:,:]
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompts = attend_to_x[:,:,-1,:,:]
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', s_attn_sel_scores, 
                    s_attend_to_x)
            if self.target_share != 2:
               ts = attn_sel_scores[:,:,-1]
               ts = ts.reshape(batch_size, 1, 1, 1)
               private_prompts = ts * private_prompts
            soft_prompts = torch.cat(
                   [avg_prompts, private_prompts], dim=2)
        elif compose_method == "wcat":
            mylogs.bp("wcat")
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', attn_sel_scores, 
                    attend_to_x)
            ts = target_shares.reshape(batch_size, 1, 1, 1)
            if self.target_share != 2:
                private_prompt = ts * private_prompt
            private_prompt = private_prompt.unsqueeze(1)
            soft_prompts = torch.cat(
                   [avg_prompts, private_prompt], dim=2)
            attn_sel_scores = torch.cat(
                   [attn_sel_scores, target_shares.reshape(batch_size, 1, 1)], dim=-1)
            attend_to_idx = torch.cat([attend_to_idx, target_idx], dim=-1) 
        elif  "pool" in compose_method:
            mylogs.bp("pool")
            # b t s l d > b t l d
            # 12 1 4 10 768 > 12 1 10 768
            # 12 1 4 7680
            # 12 7680 4 pooling
            # 12 7680 1
            # 12 7680
            # 12 1 10 780
            if "mpool" in compose_method:
                pool = torch.nn.AdaptiveMaxPool1d(1)
            else:
                pool = torch.nn.AdaptiveAvgPool1d(1)

            inp = attend_to_x
            if compose_method in ["wpool","wmpool"]:
                inp = torch.einsum(
                    'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            x = inp.view(inp.size(0), inp.size(1), inp.size(2), -1)
            x = x.permute(0, 1, 3, 2)
            x = x.view(-1, x.size(2), x.size(3))
            x = pool(x)
            x = x.squeeze(dim=-1)
            soft_prompts = x.reshape(batch_size, num_targets,-1, self.model_dim) 
        elif compose_method == "lin":
            mylogs.bp("lin")
            # inp = attend_to_x
            inp = torch.einsum(
                'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            # Flatten the source prompts along the second and third dimensions
            x = inp.view(inp.size(0), -1)
            # Pass the flattened source prompts through the neural network
            x = self.comp_linear(x)
            soft_prompts = x.reshape(batch_size, num_targets,-1, self.model_dim) 
            # attn_sel_scores = F.softmax(soft_prompts, dim=1)
        elif compose_method == "lin2":
            mylogs.bp("lin")
            x = attend_to_x
            x = self.attn_Wa(x)
            x = torch.einsum(
                'bts, btsld -> btld', attn_sel_scores, x)
            # x = x.reshape(batch_size, num_targets,-1)
            # x = self.attn_Wa(x)
            # x = self.layer_norm(x)
            soft_prompts = x.reshape(batch_size, num_targets,-1, self.model_dim) 
        elif compose_method == "sub":
            mylogs.bp("sub")
            x = torch.einsum(
                'bts, btsld -> btsld', attn_sel_scores, attend_to_x)
            x = self.attn_W_down(x)
            # x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            # x = self.layer_norm(x)
            soft_prompts = x.reshape(batch_size, num_targets,-1, self.model_dim) 

        if self.add_input:
            mylogs.bp("adinp")
            soft_prompts = avg_inputs_embeds.unsqueeze(1) + soft_prompts 
        return soft_prompts, attn_sel_scores, attend_to_idx

    def add_target_prompts(self, target_prompts, soft_prompts, 
            attn_sel_scores, attend_to_idx, target_idx):
       batch_size = soft_prompts.shape[0]
       device = soft_prompts.device
       if self.target_share is not None:
            if self.target_share == -1 or self.target_share == -10:
                target_router = self.target_router.unsqueeze(0)
                target_router = batched_index_select(target_router, 1, target_idx)
                if self.target_share == -10:
                    target_shares = target_router
                else:
                    if self.training:
                        tst = self.target_share_temperature
                        # tst = self.temperature
                        target_shares = RelaxedBernoulli(temperature=tst, 
                            logits=target_router).rsample()            
                    else:
                        target_shares = torch.sigmoid(target_router) # * self.sig_coef) 
                        # target_shares = F.softmax(target_router, dim=-1)
                        # target_shares = RelaxedBernoulli(temperature=0.01, 
                        #    logits=target_router).rsample()            
            elif self.target_share >= 1:
                target_shares = torch.ones(1, batch_size, device=device)
            else:
                target_shares = self.target_share * torch.ones(1, batch_size, device=device)
       if self.target_share == -2:
            top, _ = torch.max(attn_sel_scores, -1) 
            target_shares = top.transpose(0,1)
       elif self.target_share == -3:
            top, _ = torch.max(attn_sel_scores, -1) 
            target_shares = 1 - top.transpose(0,1)
       elif self.target_share == -4:
            top = torch.mean(attn_sel_scores, -1) 
            target_shares = 1 - top.transpose(0,1)
       mylogs.bp("cmm")
       attn_mask = self.attn_mask
       if not self.training: 
           mylogs.bp("ccc")
           if "keep-source" in self.gen_conf["mask_type"]:
               mylogs.bp("keepsrc")
           elif "keep-" in self.gen_conf["mask_type"]:
               mylogs.bp("adtkeepprompt")
           if self.gen_conf is not None and "attn_mask" in self.gen_conf:
               attn_mask = self.gen_conf["attn_mask"] 
       mylogs.bp("adt")
       target = target_prompts
       mask = torch.zeros((batch_size,1), device=attn_mask.device)
       for i in range(batch_size):
            mask[i] = attn_mask[target_idx[i].reshape(-1,1), target_idx[i]]
       mask = mask.reshape(batch_size, 1, 1, 1)
       if self.target_share == 1:
           soft_prompts = mask * target
       elif self.target_share == 2:
           target = mask * target
       elif self.target_share != 0:
           ts = target_shares.reshape(batch_size, 1, 1, 1)
           soft_prompts = (1 - ts) * soft_prompts 
           target = mask * ts * target
       
       mylogs.bp("prod")
       if self.compose_target in ["cat","concat"]:
           soft_prompts = torch.cat([soft_prompts, target], dim=2)
       elif self.compose_target in ["prod"] or self.out_dim != self.model_dim:
           _soft_prompts = soft_prompts.view(-1, 
                   soft_prompts.size(-2), soft_prompts.size(-1))
           _target = target_prompts.view(-1, target.size(-2), target.size(-1))
           soft_prompts = soft_prompts * target 
       elif self.compose_target == "mscat":
           btsld = soft_prompts.shape
           split_index = btsld[3] // 2  # Split index for the 's' dimension
           A_split = torch.split(soft_prompts, split_index, dim=3)
           B_split = torch.split(target, split_index, dim=3)

           C_mult = A_split[0] * B_split[0]  # Multiplication
           C_add = A_split[1] + B_split[1]    # Addition

           soft_prompts = torch.cat([C_mult, C_add], dim=3)           
       else:
           soft_prompts = soft_prompts + target 

       #if self.compose_target == "mcat":
       #    soft_prompts = self.layer_norm(soft_prompts)
       # attn_sel_scores = torch.cat(
       #        [attn_sel_scores, target_shares.reshape(batch_size, 1, 1)], dim=-1)
       # attend_to_idx = torch.cat([attend_to_idx, target_idx], dim=-1) 
       return soft_prompts, attn_sel_scores, attend_to_idx

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    @property
    def get_target_prompt_ids_mask(self):
        return lambda x: self.isin(x, self.target_prompt_ids)

    @property
    def get_common_prompt_ids_mask(self):
        return lambda x: self.isin(x, self.common_prompt_ids)

    @property
    def get_task_prompt_ids_mask(self):
        return lambda x: self.isin(x, self.task_prompt_ids)

    # pppppppppp
    def prompt_encoders_forward(self, input_ids, inputs_embeds, task_ids, task, att_mask):
        if len(self.prompt_encoders) > 0:
            mylogs.bp("fwd")
            if not self.training:
                mylogs.bp("fll")
            device=input_ids.device
            batch_size = inputs_embeds.shape[0]
            tids = task_ids
            if task_ids is None and task:
                mylogs.bp("tids")
                if task in self.task_names:
                    tid = self.task_names.index(task)
                    tids = torch.full((batch_size, 1), tid, dtype=torch.long) 
            num_prompt_encoders = len(self.prompt_encoders) + 1
            #num_source_encoders = len([e for e in self.prompt_encoders if e.is_source]) + 2
            # prompt masks for all prompt tokens
            target_prompt_masks = self.get_target_prompt_ids_mask(input_ids)
            self.adapter_config.prompt_masks = target_prompt_masks
            # exteract prompt ids of tasks in the batch
            target_prompt_ids = input_ids[target_prompt_masks].view(batch_size,-1) 

            common_prompt_masks = self.get_common_prompt_ids_mask(input_ids)
            common_prompt_ids = input_ids[common_prompt_masks].view(batch_size,-1) 

            task_prompt_masks = self.get_task_prompt_ids_mask(input_ids)
            sp_prompt_ids = input_ids[task_prompt_masks].view(batch_size,-1) 
            
            mylogs.bp("fwd")
            target_prompts = torch.zeros(
                (*target_prompt_ids.size(), self.model_dim), 
                                          device=device) 
            #target_prompts = torch.zeros((
            #    (batch_size,
            #    self.out_dim, 
            #    self.model_dim)
            #), device=device)
            common_prompts = torch.zeros((*common_prompt_ids.size(), self.model_dim), 
                                          device=device) 
            task_prompts = torch.zeros((*sp_prompt_ids.size(), self.model_dim), 
                                          device=device) 
            # a list of indexes to target encoders (one encoder per task)
            target_idx = torch.zeros_like(target_prompt_ids, device=device).long() 
            source_idx_list = [0] # 0 is for input 
            target_idx_list = []
            target_prompts_list = []
            task_prompts_list = []
            common_prompts_list = []
            src_prompts = torch.zeros(
                (num_prompt_encoders, 
                 self.src_prompt_dim, self.out_dim), device=device) 
            ii = 1
            for encoder in self.prompt_encoders:
                if encoder.is_source: # and self.use_source_prompts:
                    source_idx_list.append(ii)
                    emb = encoder(encoder.net_inps)
                    src_prompts[encoder.src_idx, :] = emb
                    ii += 1
                    continue
                
                prompt_token_fn = encoder.get_prompt_token_fn()
                if encoder.is_common:
                    common_masks = prompt_token_fn(common_prompt_ids)
                    if common_masks.any():
                        prompt_input_ids = common_prompt_ids[common_masks]
                        #call forwards on prompt encoder whose outputs are prompt embeddings
                        out = encoder(prompt_input_ids, tids)
                        prompt_embeds = out.to(device)
                        common_prompts_clone = common_prompts.clone()
                        common_prompts_clone[common_masks] = prompt_embeds
                        common_prompts_list.append(common_prompts_clone)
                        continue

                target_masks = prompt_token_fn(target_prompt_ids)
                if not target_masks.any():
                    task_masks = prompt_token_fn(sp_prompt_ids)
                    if task_masks.any():
                        #find input ids for prompt tokens
                        prompt_input_ids = sp_prompt_ids[task_masks]
                        #call forwards on prompt encoder whose outputs are prompt embeddings
                        out = encoder(prompt_input_ids, tids)
                        prompt_embeds = out.to(device)
                        task_prompts_clone = task_prompts.clone()
                        task_prompts_clone[task_masks] = prompt_embeds
                        task_prompts_list.append(task_prompts_clone)
                    else:
                        ii += 1
                else: 
                    #find input ids for prompt tokens
                    prompt_input_ids = target_prompt_ids[target_masks]
                    #call forwards on prompt encoder whose outputs are prompt embeddings
                    mylogs.bp("fwdtarget")
                    out = encoder(prompt_input_ids, tids)
                    prompt_embeds = out.to(device)
                    target_prompts_clone = target_prompts.clone()
                    target_prompts_clone[target_masks] = prompt_embeds
                    target_prompts_list.append(target_prompts_clone)
                    target_idx_list.append(ii)
                    target_idx[target_masks] = ii
                    ii += 1
            if common_prompts_list:
                common_prompts = torch.stack(common_prompts_list) 
                # averaging task prompts in the case that there are shared prompts
                mask = common_prompts!=0
                common_prompts = (common_prompts*mask).sum(dim=0)/mask.sum(dim=0)
                inputs_embeds[common_prompt_masks]=common_prompts.view(-1, self.model_dim)
            if task_prompts_list:
                task_prompts = torch.stack(task_prompts_list) 
                # averaging task prompts in the case that there are shared prompts
                mask = task_prompts !=0
                task_prompts = (task_prompts*mask).sum(dim=0)/mask.sum(dim=0)
                inputs_embeds[task_prompt_masks]=task_prompts.view(-1, self.model_dim)
            if target_idx_list:
                target_prompts = torch.stack(target_prompts_list) 
                mask = target_prompts != 0
                # averaging target prompts in the case that there are shared prompt tokens
                target_prompts = (target_prompts*mask).sum(dim=0)/mask.sum(dim=0)
                if self.attn_prompt_tuning and not self.target_share == 1:
                    attn_mask = self.attn_mask
                    mylogs.bp("ccc")
                    attn_mat = None
                    if not self.training: 
                        if self.gen_conf is not None and "attn_mask" in self.gen_conf:
                            attn_mask = self.gen_conf["attn_mask"] 
                        if self.gen_conf is not None and "attn_mat" in self.gen_conf:
                            attn_mat = self.gen_conf["attn_mat"] 
                            if attn_mat is not None:
                                mylogs.bp("amat")
                    if len(source_idx_list) > 1 or self.attend_input:
                        target_idx = torch.unique_consecutive(target_idx, dim=1)  
                        source_idx_list = torch.tensor(source_idx_list, device=device).long()
                        target_idx_list = torch.tensor(target_idx_list, device=device).long()
                        #target_idx = target_idx_list.repeat(batch_size, 1)
                        mylogs.bp("fwdmask")
                        if not self.training: 
                           mylogs.bp("ccc")
                           if "keep-source" in self.gen_conf["mask_type"]:
                               mylogs.bp("keepsrc")
                           elif "keep-" in self.gen_conf["mask_type"]:
                               mylogs.bp("keepprompt")
                           if self.gen_conf is not None and "attn_mask" in self.gen_conf:
                               attn_mask = self.gen_conf["attn_mask"] 
                        source_idx = source_idx_list.repeat(batch_size, 1)
                        attn_mask = attn_mask.repeat(batch_size, 1, 1)
                        sel_attn_mask = batched_index_select(attn_mask, 2, 
                                source_idx.unsqueeze(1))
                        sel_attn_mask = batched_index_select(sel_attn_mask, 1, 
                                target_idx.unsqueeze(1))
                        s_mask = sel_attn_mask.bool().squeeze(1)
                        source_idx = source_idx[s_mask].view(batch_size, -1)
                        src_prompts = src_prompts.repeat(batch_size, 1, 1, 1) 
                        sel_prompts = batched_index_select(src_prompts, 1, 
                            source_idx.unsqueeze(1))
                        mylogs.bp("fwdatt")
                        #if (self.attend_target 
                        #    or self.add_target and self.compose_method in ["cat"]):
                        #    pool = torch.nn.AdaptiveMaxPool1d(self.src_prompt_dim)
                        #    tpv = target_prompts.view(batch_size,-1,self.model_dim)
                        #    avg_tp = pool(tpv.permute(0,2,1)).permute(0,2,1)
                        #    avg_tp = avg_tp.view(batch_size, -1, 
                        #            self.src_prompt_dim, self.model_dim)
                        #    sel_prompts = torch.cat((sel_prompts, avg_tp), dim=1)
                        #    source_idx = torch.cat([source_idx, target_idx], dim=1)
                        mylogs.bp("fwdatt")
                        if source_idx.size(1) > 1 or self.attend_input:
                            soft_prompts, attn_scores, attend_to_idx = self.attend_prompts(
                                inputs_embeds, 
                                src_prompts = sel_prompts, 
                                source_idx=source_idx, 
                                target_idx=target_idx, 
                                task_ids = tids,
                                attn_mat = attn_mat,
                                task=task)
                            if self.add_target:
                                target_prompts = target_prompts.view(batch_size,
                                    -1, self.prompt_dim, self.out_dim)
                                (soft_prompts, 
                                 attn_scores, 
                                 attend_to_idx) = self.add_target_prompts(
                                         target_prompts,
                                         soft_prompts,
                                         attn_scores,
                                         attend_to_idx,
                                         target_idx=target_idx 
                                         )
                            # self.adapter_config.soft_prompts = soft_prompts.view(-1, 
                            # self.model_dim)
                            if not self.training:
                                num_targets = target_idx.size()[-1]
                                attend_to_idx = attend_to_idx.view(batch_size, 
                                        num_targets, -1)
                                src_idx = attend_to_idx[batch_size - 1]
                                tgt_idx = target_idx[batch_size - 1]
                                mylogs.bp("pred2")
                                ascore = attn_scores[batch_size - 1]
                                self.attn_scores[tgt_idx.reshape(-1,1), src_idx] = ascore 
                                self.attn_mask_learned[tgt_idx.reshape(-1,1), src_idx] = 1 
                            ###### Pad extra prompt tokens
                            # amask = amask.squeeze(1)
                            masked_prompts = soft_prompts
                            tmask = target_prompt_masks.clone()
                            amask = torch.ones((batch_size, 
                                attn_scores.size(-1)*self.src_prompt_dim), dtype=bool)
                            ignore_zeros = False
                            if not self.training:
                                ignore_zeros = self.gen_conf.get("ignore_zeros", False)
                            if (self.compose_method in ["cat","concat","scat","mcat"] 
                                and ignore_zeros):
                                mylogs.bp("pred1")
                                if self.training: 
                                    thresh = self.sel_thresh 
                                else:
                                    thresh = self.gen_conf.get("gen_thresh_min", None)
                                if thresh is not None:
                                    amask = attn_scores > thresh 
                                    if not torch.all(amask):
                                        mylogs.bp("amask")
                                    amask = amask.repeat_interleave(self.src_prompt_dim)
                                    amask = amask.view(batch_size, -1)
                                    _amask = amask.unsqueeze(1)
                                    masked_prompts = soft_prompts[_amask]

                                number_to_keep_per_batch = torch.sum(amask, dim=-1) 
                                sequence_length = tmask.size(1)
                                if True: #self.padding_pos == "end":
                                    sequence_range = range(sequence_length)
                                else:
                                    sequence_range = range(sequence_length -1, -1, -1)
                                num_true = [0]*batch_size
                                alen = amask.size(1)
                                for i in range(batch_size):
                                    k = 0
                                    for j in sequence_range: 
                                        if (tmask[i, j] and k < alen and amask[i, k] 
                                            and num_true[i] < number_to_keep_per_batch[i]):
                                            num_true[i] += 1
                                            k += 1
                                        elif tmask[i, j]:
                                            tmask[i, j] = False
                                            att_mask[i, j] = 0
                                            input_ids[i, j] = 0
                                            k += 1

                            inputs_embeds[tmask]= masked_prompts.view(-1, self.model_dim)
                            if not self.training: # or mylogs.is_debug(): 
                                pass
                                # assert torch.all((attn_scores >= 0) 
                                # & (attn_scores <= 1)), "Not all values are between 0 and 1"
                                # assert torch.all((self.attn_scores >= 0) 
                                # & (self.attn_scores <= 1)), \ 
                                # "Not all values of self.attn_scores are between 0 and 1"
                                # targets = self.target_encoders_idx
                                #ss1 = self.attn_scores  
                                # self.attn_scores.index_select(0, targets)
                                #ss2 = self.router.index_select(0, targets)
                                #ss3 = self.attn_mask.index_select(0, targets)
                                #y_labels = [self.prompt_names[i] for i in targets]
                                #img_buf = WBCallback.save_images(scores=[ss1,ss2,ss3], 
                                #    y_labels=y_labels,
                                #    x_labels=self.prompt_names,
                                #    title= "at5:" + self.route_method + ":" \
                                #            + self.compose_method + ":" + self.attn_method, 
                                #    add_tags=False) 
                        else:
                            self.adapter_config.soft_prompts=target_prompts.view(-1, 
                                    self.model_dim)
                            inputs_embeds[target_prompt_masks]= target_prompts.view(-1, 
                                    self.model_dim)
                    else:
                        self.adapter_config.soft_prompts=target_prompts.view(-1, 
                                self.model_dim)
                        inputs_embeds[target_prompt_masks]= target_prompts.view(-1, 
                                self.model_dim)
                else:
                    self.adapter_config.soft_prompts=target_prompts.view(-1, self.model_dim)
                    inputs_embeds[target_prompt_masks]=target_prompts.view(-1, 
                            self.model_dim)
            return input_ids, att_mask 
        return input_ids, att_mask
    ######################################################
    def per_layer_config(self, config, layer_id, adapter_config, is_decoder):
        """Sets the train_task_adapter in the config, based on the information given."""
        def is_valid_layer(layer_id, adapter_config, is_decoder):
            valid_layer_ids = adapter_config.task_adapter_layers_encoder\
                if not is_decoder else adapter_config.task_adapter_layers_decoder
            if valid_layer_ids is None:
                return True
            return True if layer_id in valid_layer_ids else False
        if adapter_config is None:
            return config
        config = copy.deepcopy(config)
        valid_task_adapter_layer_id = is_valid_layer(
            layer_id, adapter_config, is_decoder)
        add_task_adapter = True if not is_decoder else adapter_config.task_adapter_in_decoder
        config.train_task_adapters = config.train_task_adapters and\
            valid_task_adapter_layer_id and\
            add_task_adapter
        return config
        ####################################################

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(
                torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + \
            str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def set_target_task_id(self, task_id):
        self.target_task_id = task_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_embedding=None,
        task_ids=None,
        task=None
    ):
        # Model parallel
        #task_ids=None #TODO remove it
        #task_ids = task_ids.long()
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            ################ MyCode mmmmmmmmmmmm
            mylogs.bp("befw")
            if self.prompt_tuning or self.attn_prompt_tuning:
                input_ids, attention_mask = self.prompt_encoders_forward(input_ids, 
                        inputs_embeds, task_ids, task, att_mask = attention_mask)
            ################ My code End

            if self.append_prefix and self.append_attn_prefix is False:
                inputs_embeds = torch.cat([self.prefix_emb.unsqueeze(0).repeat(
                    inputs_embeds.shape[0], 1, 1), inputs_embeds], dim=1)  # bsz, seqlen, dim
                input_shape = inputs_embeds.size()[:-1]

            ##################################### aaaaaaaaaaaa
            if self.append_attn_prefix:
                if self.attend_target is True:
                    if task_ids is not None:
                        target_prompts = torch.index_select(
                            self.prefix_emb, 0, task_ids)
                    else:
                        if self.shared_attn is False:
                            target_prompts = self.prefix_emb.repeat(
                                inputs_embeds.shape[0], 1, 1)
                        else:
                            target_prompts = self.prefix_emb[0].repeat(
                                inputs_embeds.shape[0], 1, 1)
                    mul_prefix_emb_added = torch.cat((self.mul_prefix_emb.repeat(
                        inputs_embeds.shape[0], 1, 1, 1), target_prompts.unsqueeze(1)), dim=1)
                else:
                    mul_prefix_emb_added = self.mul_prefix_emb.repeat(
                        inputs_embeds.shape[0], 1, 1, 1)

                soft_prompts, _,_ = self.attend_prompts(inputs_embeds, 
                    src_prompts = mul_prefix_emb_added, 
                    target_prompts = target_prompts,
                    add_target = self.add_target)
                inputs_embeds = torch.cat(
                    [soft_prompts, inputs_embeds], dim=1)  # bsz, seqlen, dim
                input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + \
            seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape) #, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                        hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    task_ids=task_ids,
                    task=task
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + \
                    (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            `What are input IDs? <../glossary.html#input-ids>`__

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
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

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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



def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.prefix_tuning = config.prefix_tuning
        self.attn_tuning = config.attn_tuning
        self.attn_method = config.attn_method
        self.attend_target = config.attend_target
        if self.prefix_tuning:
            self.prefix_dim = adapter_config.prefix_dim
            self.init_prefix_from_vocab = adapter_config.init_prefix_from_vocab
        self.shared_attn = config.shared_attn
        self.temperature = config.temperature
        self.learned_temperature = config.learned_temperature
        if self.shared_attn is True:
            self.prefix_shared = nn.Parameter(torch.zeros(
                (config.num_target, self.prefix_dim, config.d_model))) if self.prefix_tuning else None
        else:
            self.prefix_shared = nn.Parameter(torch.zeros(
                (self.prefix_dim, config.d_model))) if self.prefix_tuning else None
        self.prefix_num = config.prefix_num

        self.mul_prefix_emb = nn.Parameter(torch.zeros(
            (self.prefix_num, self.prefix_dim, config.d_model))) if self.prefix_tuning and self.attn_tuning else None

        #############################################################
        self.num_src_encoders = 0
        self.source_encoders_idx = None
        self.target_encoders_idx = None
        self.target_prompt_ids = []
        self.task_prompt_ids = []
        self.attn_scores = None
        self.attn_mask = None
        self.attn_mask_orig = None
        self.attn_mask_learned = None
        self.prompt_names = None

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, adapter_config=adapter_config, prefix_emb=self.prefix_shared, attn_tuning=self.attn_tuning, mul_prefix_emb=self.mul_prefix_emb,
                               model_dim=config.d_model, attn_method=self.attn_method, shared_attn=self.shared_attn, attend_target=self.attend_target, temperature=self.temperature, learned_temperature=self.learned_temperature)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        if config.train_task_adapters:
            decoder_config.train_task_adapters = adapter_config.task_adapter_in_decoder
        self.decoder = T5Stack(decoder_config, self.shared, adapter_config=adapter_config, prefix_emb=self.prefix_shared, attn_tuning=self.attn_tuning, mul_prefix_emb=self.mul_prefix_emb,
                               model_dim=config.d_model, attn_method=self.attn_method, shared_attn=self.shared_attn, attend_target=self.attend_target, temperature=self.temperature, learned_temperature=self.learned_temperature)

        self.bitfit = adapter_config.bitfit if adapter_config is not None else False
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False if not self.bitfit else True)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    ########## My functions
    @property
    def prompt_encoders(self):
        return self.encoder.prompt_encoders

    @property
    def prompt_encoders_num(self):
        return len(self.encoder.prompt_encoders)
    
    def load_encoders(self, load_dir = None, load_source_prompts = False, prefix=""):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prefix = "pt_" + prefix if not self.attn_tuning else "att_" + prefix
        prefix = prefix.strip("_")
        for encoder in self.prompt_encoders:
            if not load_source_prompts and encoder.is_source:
                continue
            encoder.load(load_dir, prefix=prefix)
            encoder.to(device)

    def store_encoders(self, output_dir = None, 
            prompts_only=False, 
            prompts_and_router_only=False,
            save_source_prompts = False, prompts_to_save=None, prefix="", 
            router_prefix="", save_router=False):
        prefix = prefix.strip("_")
        if prompts_to_save:
            for encoder in self.prompt_encoders:
                if not save_source_prompts and encoder.is_source:
                    continue
                if (prompts_to_save != "all" 
                    and not encoder.name in prompts_to_save):
                    continue
                encoder.save(output_dir, prefix=prefix)
        if prompts_only: return
        attn_tuning = self.attn_tuning
        mylogs.bp("router")
        if attn_tuning is True and save_router:
            router_dict = {}
            for i, n in enumerate(self.encoder.prompt_names):
                router_dict[n] = self.encoder.router[i]
            torch.save(router_dict, os.path.join(
                        output_dir, router_prefix +  "_router.pt"))

        if prompts_and_router_only: return
        for name, param in self.named_parameters():
            # Save attention and layer norm weights.
            if attn_tuning is True and "encoder.attn_Wa.weight" == name:
                attn_weights_params = param
                torch.save(attn_weights_params, os.path.join(
                    output_dir, "attn_Wa_weights.pt"))
            if attn_tuning is True and "encoder.attn_W_down.weight" == name:
                attn_weights_params = param
                torch.save(attn_weights_params, os.path.join(
                    output_dir, "attn_W_down.pt"))
            if attn_tuning is True and "encoder.attn_W_up.weight" == name:
                attn_weights_params = param
                torch.save(attn_weights_params, os.path.join(
                    output_dir, "attn_W_up.pt"))
            if attn_tuning is True and "encoder.layer_norm.weight" == name:
                attn_weights_params = param
                torch.save(attn_weights_params, os.path.join(
                    output_dir, "layer_norm_weight.pt"))
            if attn_tuning is True and "encoder.layer_norm.bias" == name:
                attn_weights_params = param
                torch.save(attn_weights_params, os.path.join(
                    output_dir, "layer_norm_bias.pt"))

    # Before attention
    ################## End my functions

    def init_prefix_weights(self):
        if self.init_prefix_from_vocab:
            indices = np.random.permutation(range(5000))[:self.prefix_dim]
            init_weight = self.get_input_embeddings().state_dict()[
                "weight"][indices]
            self.prefix_shared.data = init_weight.clone().detach()
        else:
            random_range = 0.5
            self.prefix_shared.data.uniform_(-random_range, random_range)

    def store_prefix_weights(self, prefix_embeddings):
        # need to pass them as a parameter?
        # stack or cat?
        #embeddings = torch.stack([emb.cuda() for emb in prefix_embeddings])
        embeddings = torch.stack([emb for emb in prefix_embeddings])
        # Initialize the embeddings
        self.mul_prefix_emb.data = embeddings.clone().detach()

    def update_router(self, path):
        mapl=torch.device('cpu')
        self.encoder.router = torch.load(path, map_location=mapl)
    # update attention weights
    def update_attention_weights(self, attention):
        self.encoder.attn_Wa.data = attention.cuda()

    def update_layer_norm_weights(self, layer_norm_dir):
        self.encoder.layer_norm.weight.data = torch.load(
            os.path.join(layer_norm_dir, "layer_norm_weight.pt"))
        self.encoder.layer_norm.bias.data = torch.load(
            os.path.join(layer_norm_dir, "layer_norm_bias.pt"))

    def update_attention_weights_sub(self, attention):
        assert len(attention) == 2
        assert "attn_W_down" in attention[0]
        assert "attn_W_up" in attention[1]
        self.encoder.attn_W_down.weight.data = torch.load(attention[0]).cuda()
        self.encoder.attn_W_up.weight.data = torch.load(attention[1]).cuda()

    def update_prefix_weights_single(self, prefix_embedding):
        self.prefix_shared.data = prefix_embedding

    def update_prefix_weights_multi(self, prefix_embedding, num_target):
        self.prefix_shared.data = torch.stack(
            [prefix_embedding.detach().clone() for _ in range(num_target)])

    def update_prefix_weights(self, prefix_embeddings, target_embedding=None):

        def prefix_emb_similarity(emb_a, emb_b):
            return torch.sum(F.cosine_similarity(emb_a.cuda(), emb_b.cuda()))

        if len(prefix_embeddings) == 1:
            self.prefix_shared.data = prefix_embeddings[0]
        else:
            if target_embedding is not None:
                target_embedding.cuda()
                sum_sims = torch.sum(torch.Tensor([prefix_emb_similarity(
                    emb, target_embedding) for emb in prefix_embeddings]))
                W_weighting = torch.Tensor([prefix_emb_similarity(
                    emb, target_embedding) / sum_sims for emb in prefix_embeddings]).detach()
                res = torch.einsum(
                    'mld,m->ld', torch.stack([emb.cuda() for emb in prefix_embeddings]), W_weighting.cuda())
                self.prefix_shared.data = res
            else:
                self.W_weighting = nn.Parameter(
                    torch.rand(len(prefix_embeddings)))
                res = torch.einsum(
                    'mld,m->ld', torch.stack([emb.cuda() for emb in prefix_embeddings]), self.W_weighting.cuda())
                self.prefix_shared.data = res

    ###########################################################

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    ## my code for resizing embeddings
    def resize_token_embeddings(self, new_num_tokens = None) -> torch.nn.Embedding:
        resized_embeds = super().resize_token_embeddings(new_num_tokens)
        self.set_input_embeddings(resized_embeds)
        return resized_embeds

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        task_ids=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if self.prefix_tuning:
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones((attention_mask.shape[0], self.prefix_dim)).to(
                    attention_mask.device), attention_mask], dim=1)
 
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task_embedding=None,
                task=task,
                task_ids=task_ids
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_embedding=None,
            task=task
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        mylogs.bp("loss")
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            #### my code
            if loss.isnan(): 
                eps = 1e-6
                loss = torch.tensor(eps, requires_grad=True)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs, model_name
    ) -> Dict[str, Any]:

        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            if self.prefix_tuning:
                attention_mask = encoder_kwargs['attention_mask']
                if attention_mask is not None:
                    encoder_kwargs['attention_mask'] = torch.cat([torch.ones(
                        (attention_mask.shape[0], self.prefix_dim)).to(attention_mask.device), attention_mask], dim=1)
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(
                input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "task": kwargs["task"]
            # "lang": kwargs["lang"]
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
