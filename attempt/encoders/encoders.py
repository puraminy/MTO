import wandb
import re
from pathlib import Path
import transformers
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import os, glob
import math
from os.path import expanduser
import attempt.mylogs as mylogs

from transformers import AddedToken 
from transformers.optimization import Adafactor
from torch.optim import AdamW
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from attempt.maps import *
from collections import OrderedDict

def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)

class PromptEncoder(torch.nn.Module):
    enc_type = "encoder"
    def __init__(self, name, prompt_tokens, length=None, model=None, 
            tokenizer=None, 
            is_source =False, enc_type="encoder", 
            num_tasks = None, task_emb_dim=50, 
            task_compose_method='concat'): 
        super().__init__()
        self.name = name
        self.load_name = name
        self.enc_type = enc_type
        self.prompt_tokens = prompt_tokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.length = len(prompt_tokens) if prompt_tokens else length
        self.embedding_dim = model.config.hidden_size
        self.embedding = torch.nn.Embedding(self.length, self.embedding_dim)
        self.num_tasks = num_tasks
        self.task_emb_dim = task_emb_dim
        self.task_compose_method = task_compose_method
        #self.task_embedding = nn.Embedding(num_tasks, self.task_emb_dim)
        #self.task_cat_proj = nn.Linear(self.embedding_dim + self.task_emb_dim, 
        #        self.embedding_dim).to(self.device)
        #self.task_proj = nn.Linear(self.task_emb_dim, self.embedding_dim).to(self.device)
        self.net_inps = torch.arange(self.length, device=self.device)

        self.prompt_ids = self.get_prompt_ids(prompt_tokens, model, tokenizer)
        self.input_ids = torch.tensor(self.prompt_ids, device=self.device)
        self.id_offset = min(self.prompt_ids) if self.prompt_ids else 0 
        self.is_source = is_source
        self.is_loaded = False
        self.is_shared = True 
        self.is_target = False
        self.is_private = False
        self.is_common = False
        self.src_idx = -1
        self.attend_to_mask = None
        self.attend_to = []
        if not is_source:
            self.attend_to = ["source_" + name, "source_for_" + name]

    def get_prompt_ids(self, prompt_tokens, model, tokenizer, init_emb_flag = True):
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

        if not init_emb_flag:
            return prompt_ids 

        cur_embeddings = model.get_input_embeddings()
        init_embs = {}
        mylogs.bp("encoder")

        for pid, p in enumerate(prompt_ids):
            if pid < cur_embeddings.num_embeddings:
                emb = cur_embeddings.weight[pid,:].detach().clone() 
                init_embs[pid] = emb

        # init from words
        for pid, p in enumerate(prompt_tokens):
            if "_" in p:
               q = p.strip("<").strip(">")
               w = q.split("_")[1]
               if "?" in w: w = w.split("?")[0]
               if not w.isdigit():
                   wid = tokenizer.convert_tokens_to_ids([w])[0]
                   emb = cur_embeddings.weight[wid,:].detach().clone() 
                   init_embs[pid] = emb

        self.init_embs = init_embs
        self.init_embedding(init_embs)
        return prompt_ids 

    def get_filename(self, length=None, prefix="", as_saved=False, name=""):
        if not name: name = self.load_name if self.load_name else self.name
        length = length if length is not None else self.length
        if as_saved: 
            fname= (prefix + "_" if prefix else "") + \
                    self.enc_type + "_" + name + "_" + str(length) + ".pt"
        else:
            fname=(prefix + "_" if prefix else "") + \
                    self.enc_type + "_" + name + "_" + str(length) + ".pt"
                    # name + "_" + str(length) + ".pt"
        if self.is_source:
            fname = fname.replace("source_","") 
        if self.is_target:
            fname = fname.replace("tar-","") 
        return fname

    def save(self, save_dir, prefix="pt"):
        fname = os.path.join(save_dir, self.get_filename(prefix=prefix, as_saved=True))
        state_dict = self.state_dict()
        torch.save(state_dict, fname)

    def exists(self, load_dir, prefix="pt", length = None, as_saved=False):
        fname = os.path.join(load_dir, self.get_filename(length, prefix, as_saved=as_saved))
        files = glob.glob(fname)
        if  len(files) > 0:
            fname = files[0]
            return True, fname
        return False, fname

    def load(self, load_dir, prefix="pt", length = None, as_saved=False, 
            ignore_if_prompt_not_exists=False, name=""):
        fname = os.path.join(load_dir, self.get_filename(length, prefix, 
            as_saved=as_saved, name=name))
        files = glob.glob(fname)
        if  len(files) == 0:
            if not ignore_if_prompt_not_exists: 
                assert len(files) > 0, fname + " doesn't exists to be loaded!"
                assert len(files) == 1, fname + " are multiple files!"
            return False
        fname = files[0]
        mapl=torch.device('cpu')
        self.is_loaded = True

        state = torch.load(fname, map_location=mapl)
        size = state["embedding.weight"].size()

        self.length = size[0]
        self.embedding_dim = size[1] 
        self.embedding = torch.nn.Embedding(self.length, self.embedding_dim)
        self.load_state_dict(state)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_inps = torch.arange(self.length, device=self.device)
        mylogs.tinfo("Prompt for %s was loaded ", self.name)
        return True

    def init_embedding(self, init_embs):
        if init_embs:
            with torch.no_grad():
                for _id,emb in init_embs.items():
                    if _id < len(self.embedding.weight):
                        self.embedding.weight[_id] = emb
        else:
            random_range = 0.5
            self.embedding.weight.data.uniform_(-random_range, random_range)

    def init_embs_from_ids(self, embeds):
        embs = {}
        for i, pid in enumerate(self.prompt_ids):
           if pid < len(embeds.weight):
               emb = embeds.weight[pid,:].detach().clone() 
               self.embedding.weight[i] = emb

    def init_embs_from_words(self, embeds):
        indices = np.random.permutation(range(5000))[:self.length]
        init_weight = embeds.state_dict()[
            "weight"][indices]
        self.embedding.weight.data = init_weight.clone().detach()

    def forward(self,prompt_token_ids, tids=None, training=True):
        if tids is not None:
            task_id = tids[0]
        index_list = prompt_token_ids
        if self.input_ids.size()[0] > 0:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        ret_embeds = self.forward_step(index_list, tids, training)
        return ret_embeds

    def forward_step(self, index_list, tids=None, training=True):
        raise NotImplementedError()

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)

    def dump_embeddings_into(self, weight, task_ids = None):
        if task_ids == None:
            task_ids = [0]
        with torch.no_grad():
            embs = self.forward(self.input_ids, tids=task_ids, training=False)
            detached_embeddings = embs.detach()
            weight[self.prompt_ids,:]=detached_embeddings

    def get_emb(self, task_ids = None):
        with torch.no_grad():
            embs = self.forward(self.input_ids, tids=task_ids, training=False)
            detached_embeddings = embs.detach()
            return detached_embeddings

class EmbeddingPromptEncoder(PromptEncoder):
    enc_type = "emb"
    def forward_step(self, index_list=None, tids=None, training=True):
        """
        Args:
            index_list: Optional tensor of indices to override default input (e.g., for shuffling)
            tids: task ids (shape: [B] or scalar). Required for task conditioning.
        Returns:
            Tensor of shape [B, length, embedding_dim]
        """
        B = tids.size(0) if tids is not None else 1  # batch size
        input_ids = self.net_inps if index_list is None else index_list
        token_embs = self.embedding(input_ids)  # [length, D]
        if tids is not None:
            token_embs = token_embs.unsqueeze(0).expand(B, -1, -1)
            task_embs = self.task_embedding(tids)  # [B, task_emb_dim]
            task_embs = task_embs.unsqueeze(1).expand(-1, self.length, -1) 
            if self.task_compose_method == "concat":
                # Concatenate and apply projection
                merged = torch.cat([token_embs, task_embs], dim=-1)
                ret_embeds = self.task_cat_proj(merged)  # [B, length, embedding_dim]
            elif self.task_compose_method == "add":
                # Add task vector (after projection if dims don't match)
                if task_embs.size(-1) != self.embedding_dim:
                    task_embs = self.task_proj(task_embs)
                ret_embeds = token_embs + task_embs  # [B, length, embedding_dim]
            else:
                raise ValueError(f"Unknown task_compose_method: {task_compose_method}")
        else:
            ret_embeds = token_embs  # no task awareness

        return ret_embeds
        
class MatPromptEncoder(PromptEncoder):
    enc_type = "mat"
    def __init__(self, shared_mat, n_prompts, 
            intrinsic_dim, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature 
        self.z = nn.Parameter(data=torch.empty((
            intrinsic_dim,
            self.embedding_dim
        )).uniform_(-1e-3, 1e-3))
        #self.A = shared_mat 
        #hsize = intrinsic_dim
        #self.mlp = torch.nn.Sequential(
        #    torch.nn.Linear(intrinsic_dim, hsize),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(hsize, intrinsic_dim)
        #)

    def forward_step(self, index_list, tids=None, training=True):
        # z = self.embedding(self.net_inps)
        return self.z
        z = self.mlp(self.z)
        running_weight = torch.mm(z, self.A) 
        running_weight = running_weight.view(self.length, -1)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, hsize, activation):
        super(ResidualBlock, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, hsize)
        self.linear2 = torch.nn.Linear(hsize, out_features)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = x + out  # Skip connection
        return out


class ResidualMLPPromptEncoder(PromptEncoder):
    enc_type = "rmlp" #residual_mlp

    def __init__(self, num_layers=1, hidden_size=-1,
                 nl="relu", out_dim=-1, in_dim=-1, **kwargs):
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim
        if out_dim == -1:
            out_dim = embedding_dim
        if in_dim == -1:
            in_dim = embedding_dim
        if nl is not None:
            if nl.lower() == "gelu":
                activation = torch.nn.GELU()
            elif nl.lower() == "relu":
                activation = torch.nn.ReLU()
            elif nl.lower() == "silu":
                activation = torch.nn.SiLU()
            elif nl.lower() == "elu":
                activation = torch.nn.ELU()
            else:
                activation = None
        else:
            activation = None
        hsize = hidden_size if hidden_size > 1 else embedding_dim # // 2
        layers = [ResidualBlock(in_dim, out_dim, hsize, activation)]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hsize, hsize, activation))
        # layers.append(torch.nn.Linear(hsize, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward_step(self, index_list, tids=None, training=True):
        embs = self.embedding(self.net_inps)
        running_weight = self.mlp(embs)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds


class ResMLP(PromptEncoder):
    enc_type = "MLP1"
    def __init__(self,
                 num_layers=1,
                 enc_type = "mlp",
                 hidden_size=-1,
                 in_dim=-1,
                 out_dim=-1,
                 nl='relu', # activation function
                 layer_norm=False,
                 dropout=0.0,
                 residual=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert enc_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nl in ['relu','gelu', 'tanh', 'sigm']

        self.enc_type = enc_type 
        embedding_dim = self.embedding_dim
        if out_dim == -1:
            out_dim = embedding_dim

        if in_dim == -1:
            in_dim = embedding_dim
        
        hidden_size = hidden_size if hidden_size > 1 else embedding_dim # // 2
        if enc_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(in_dim, hidden_size)]

            if nl=='relu':
                layers.append(nn.ReLU())
            elif nl=='gelu':
                layers.append(nn.GELU())
            elif nl=='tanh':
                layers.append(nn.Tanh())
            elif nl=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(hidden_size, out_dim))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))

            if enc_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif enc_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=in_dim,
                                           hidden_size=in_dim, # // 2,
                                           num_layers=1 if enc_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(in_dim, in_dim),
                                          nn.ReLU(),
                                          nn.Linear(out_dim, out_dim))

        elif enc_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward_step(self, index_list, tids=None, training=True):
        inputs = self.embedding(self.net_inps)
        if self.enc_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.enc_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            running_weight = output_embeds
        elif self.residual:
            running_weight = self.module(inputs) + inputs
        else:
            running_weight = self.module(inputs)

        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds

import torch
import torch.nn.functional as F

class MLPPromptEncoderFiLM(PromptEncoder):
    enc_type = "mlp_film"
    def __init__(
        self,
        num_layers=1,
        hidden_size=-1,
        nl="gelu",
        out_dim=-1,
        in_dim=-1,
        task_emb_dim=128,
        film_hidden=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim

        # dims
        if out_dim == -1:
            out_dim = embedding_dim
        if in_dim == -1:
            in_dim = embedding_dim
        self.hidden_size = hidden_size
        self.task_emb_dim = task_emb_dim

        # nonlinearity lookup
        self.nlf = {
            'gelu': torch.nn.GELU(),
            'relu': torch.nn.ReLU(),
            'silu': torch.nn.SiLU(),
            'elu': torch.nn.ELU(),
            'id': torch.nn.Identity(),
        }.get(nl.lower(), None)

        # base MLP layers (ModuleList for layer alignment)
        self.mlp = torch.nn.ModuleList()
        if hidden_size <= 0:
            self.mlp.append(torch.nn.Linear(in_dim, out_dim))
        else:
            hsize = hidden_size if hidden_size > 1 else embedding_dim
            self.mlp.append(torch.nn.Linear(in_dim, hsize))
            if self.nlf: self.mlp.append(self.nlf)
            if num_layers == 2:
                self.mlp.append(torch.nn.Linear(hsize, hsize))
                if self.nlf: self.mlp.append(self.nlf)
            self.mlp.append(torch.nn.Linear(hsize, out_dim))

        # FiLM projection heads aligned with mlp layers
        self.film_projs = torch.nn.ModuleList([
            (torch.nn.Sequential(
                torch.nn.Linear(self.task_emb_dim, film_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(film_hidden, layer.out_features * 2)
            ) if isinstance(layer, torch.nn.Linear) else None)
            for layer in self.mlp
        ])

        # task embedding
        self.task_embedding = torch.nn.Embedding(self.num_tasks, task_emb_dim)

    def forward_step(self, index_list, tids=None, training=True):
        B = tids.size(0) if tids is not None else 1
        # prepare token embeddings [B, L, in_dim]
        input_ids = self.net_inps if index_list is None else index_list
        token_embs = self.embedding(input_ids)
        if tids is not None:
            token_embs = token_embs.unsqueeze(0).expand(B, -1, -1)
            task_embs = self.task_embedding(tids)
            task_embs = task_embs.unsqueeze(1).expand(-1, token_embs.size(1), -1)

            h = token_embs
            h_in = h
            # apply MLP with FiLM at each linear
            for layer, proj in zip(self.mlp, self.film_projs):
                if isinstance(layer, torch.nn.Linear):
                    h = layer(h)
                    # FiLM: proj is not None
                    gamma_beta = proj(task_embs)
                    gamma, beta = gamma_beta.chunk(2, dim=-1)
                    h = gamma * h + beta
                else:
                    h = layer(h)
            running_weight = h + h_in
        else:
            # no task conditioning: plain MLP
            h = token_embs
            h_in = h
            for layer in self.mlp:
                h = layer(h)
            running_weight = h # h.unsqueeze(0) # + h_in.unsqueeze(0)

        # index and return prompt tokens
        if index_list is None:
            index_list = torch.arange(self.length, device=self.device)
        if running_weight.dim() == 3:
            idx = index_list.unsqueeze(0).expand(running_weight.size(0), -1)
            ret_embeds = torch.gather(
                running_weight,
                dim=1,
                index=idx.unsqueeze(-1).expand(-1, -1, running_weight.size(-1))
            )
        else:
            ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds

class MLPPromptEncoder(PromptEncoder):
    enc_type = "mlp"
    def __init__(self, num_layers=1, hidden_size=-1, 
                 nl="gelu", out_dim=-1, in_dim=-1, **kwargs):
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim

        if out_dim == -1:
            out_dim = embedding_dim
        if in_dim == -1:
            in_dim = embedding_dim

        self.hidden_size = hidden_size

        # Nonlinearity selection
        if nl is not None:
            nl = nl.lower()
            if nl == "gelu":
                nlf = torch.nn.GELU()
            elif nl == "id":
                nlf = torch.nn.Identity()
            elif nl == "relu":
                nlf = torch.nn.ReLU()
            elif nl == "silu":
                nlf = torch.nn.SiLU()
            elif nl == "elu":
                nlf = torch.nn.ELU()
            else:
                nlf = None
        else:
            nlf = None

        layers = []

        # If hidden_size <= 0, use a single linear layer: in_dim -> out_dim
        if hidden_size == 0:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        else:
            hsize = hidden_size if hidden_size > 1 else embedding_dim
            layers.append(torch.nn.Linear(in_dim, hsize))
            if nlf is not None:
                layers.append(nlf)

            if num_layers == 2:
                layers.append(torch.nn.Linear(hsize, hsize))
                if nlf is not None:
                    layers.append(nlf)
            layers.append(torch.nn.Linear(hsize, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward_step(self, index_list, tids=None, training=True):
        """
        Args:
            index_list: indices into the prompt (e.g., [0, 1, 2, ...])
            tids: [B] tensor of task ids
        Returns:
            Tensor [B, len(index_list), embedding_dim]
        """
        B = tids.size(0) if tids is not None else 1
        input_ids = self.net_inps if index_list is None else index_list  # [L]
        token_embs = self.embedding(input_ids)  # [L, input_dim]
        if tids is not None:
            # Expand for batch: [B, L, input_dim]
            token_embs = token_embs.unsqueeze(0).expand(B, -1, -1)
            task_embs = self.task_embedding(tids)  # [B, task_emb_dim]
            task_embs = task_embs.unsqueeze(1).expand(-1, token_embs.size(1), -1)  
            # [B, L, task_emb_dim]

            if self.task_compose_method == "concat":
                merged = torch.cat([token_embs, task_embs], dim=-1)  
                # [B, L, input_dim + task_emb_dim]
                token_embs = self.task_cat_proj(merged)  # [B, L, embedding_dim]
            elif self.task_compose_method == "add":
                task_proj = self.task_proj(task_embs) if self.task_emb_dim != self.embedding_dim else task_embs
                token_embs = token_embs + task_proj  # [B, L, embedding_dim]
            else:
                raise ValueError(f"Unknown task_compose_method: {task_compose_method}")

        # Apply MLP: [B, L, embedding_dim]
        # running_weight = self.mlp(token_embs)  # [B, L, D]
        h = token_embs
        h_in = h
        for layer in self.mlp:
            h = layer(h)
        running_weight = h + h_in
        if index_list is None:
            index_list = torch.arange(self.length, device=self.device)

        if running_weight.dim() == 3:
            # Case where running_weight is [B, L, D] and you want per-sample indexing
            index_list = index_list.unsqueeze(0).expand(B, -1)  # [B, L_idx]
            ret_embeds = torch.gather(
                running_weight,
                dim=1,
                index=index_list.unsqueeze(-1).expand(-1, -1, running_weight.size(-1))
            )
        else:
            ret_embeds = F.embedding(index_list, running_weight)

        return ret_embeds



class LSTMEmbeddingPromptEncoder(PromptEncoder):
    enc_type = "lstm"
    def __init__(self,num_layers=1, hidden_size=-1, **kwargs) -> None:
        mylogs.bp("encoder|lstm")
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim
        hsize = hidden_size if hidden_size > 1 else embedding_dim
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim, # // 2, #my code
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        activation = torch.nn.ReLU()
        in_dim = out_dim = embedding_dim
        layers = [ResidualBlock(in_dim, out_dim, hsize, activation)]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hsize, hsize, activation))
        self.mlp = torch.nn.Sequential(*layers)

 #### llllllf
    def forward_step(self, index_list, tids=None, training=True):
        net_inputs = self.net_inps
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        x = self.lstm(embeds.unsqueeze(0))
        running_weight = self.mlp(x[0]).squeeze(0)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds

def add_general_specials(tokenizer):
    num_added_toks: dict = {}
    if tokenizer.bos_token is None:
        num_added_toks['bos_token'] = "<s>"
    if tokenizer.eos_token is None:
        num_added_toks['eos_token'] = "</s>"
    if tokenizer.pad_token is None:
        num_added_toks['pad_token'] = "<pad>"
    if tokenizer.sep_token is None:
        num_added_toks['sep_token'] = "<sep>"
    if tokenizer.cls_token is None:
        num_added_toks['cls_token'] = "<cls>"
    if tokenizer.mask_token is None:
        num_added_toks['mask_token'] = "<mask>"

    num_tokens = tokenizer.add_special_tokens(num_added_toks)
    return num_tokens

def add_pt_specials(tokenizer):
    cur_list = tokenizer.additional_special_tokens
    new_tokens = list(set(REL_TO_TOKEN.values()))+ \
                 list(set(GEN_TOKENS.values())) 
    added_tokens = [ 
            AddedToken(tok,lstrip=True,
                rstrip=True, single_word=True)
            for tok in new_tokens if not tok in cur_list
    ]
    added_tokens = cur_list + added_tokens
    num_tokens = tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    return num_tokens

def extend_tokenizer(tokenizer, model, tokens = []):
    cur_list = tokenizer.additional_special_tokens
    new_tokens = []
    new_tokens += tokens
    added_tokens = [ 
            AddedToken(tok,lstrip=True,
                rstrip=False, single_word=True)
            for tok in new_tokens if not tok in cur_list
    ]
    if added_tokens:
        added_tokens = cur_list + added_tokens
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    model.resize_token_embeddings(len(tokenizer))
    for tok in tokens:
        with torch.no_grad():
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            # model.get_input_embeddings().weight[tok_id].uniform_(-0.1, 0.1)
            orig_tok = tok.strip("<").strip(">")
            if hasattr(model, "lm_head"):
                exid = tokenizer.convert_tokens_to_ids(orig_tok)
                ex_emb = model.lm_head.weight.data[exid]
                model.lm_head.weight[tok_id] = ex_emb.clone() # .uniform_(-0.1, 0.1)

def create_encoder(name, model, tokenizer, prompt_tokens, 
        length=None, encoder_type="lstm", non_linear="relu",
        hidden_size=-1, num_layers=1, out_dim=-1, in_dim=-1,
        is_source = False, shared_mat=None, 
        num_tasks =None, task_emb_dim=50, task_compose_method='concat'):
    embedding_dim = model.config.hidden_size
    cur_list = tokenizer.additional_special_tokens
    my_specials = [x for x in cur_list if not "<extra_id"  in x]
    if "@" in name:
        name, encoder_type = name.split("@") 

    if type(encoder_type) == list:
        encoder_type = "@".join([str(p) for p in encoder_type])
    prompt_encoder = None
    if "mlp" in encoder_type:
        if encoder_type in ["mlp", "mlp_res"]:
            _enc_type = encoder_type.split("@")
            if len(_enc_type) > 1:
                num_layers = int(_enc_type[1])
            if len(_enc_type) > 2:
                hidden_size = int(_enc_type[2])
            if len(_enc_type) > 3:
                non_linear = _enc_type[3]

        # assert False, str(num_layers) + "-" + str(hidden_size) + "-" + str(non_linear)
        if encoder_type == "mlp_res" or encoder_type == "rmlp":
            prompt_encoder = ResidualMLPPromptEncoder(name = name,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                nl = non_linear,
                in_dim = in_dim,
                out_dim = out_dim,
                is_source = is_source,
                num_tasks = num_tasks,
                num_layers=num_layers, 
                hidden_size=hidden_size, 
                enc_type=encoder_type,
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                )
        elif encoder_type.startswith("mlpres"):
            res_type = "MLP1"
            if len(encoder_type.split("@")) > 0:
                res_type = encoder_type.split("@")[-1]
            prompt_encoder = ResMLP(name = name,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                nl = non_linear,
                in_dim = in_dim,
                out_dim = out_dim,
                is_source = is_source,
                num_tasks = num_tasks,
                enc_type=res_type, 
                hidden_size=hidden_size,
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                )
        elif "film" in encoder_type:
            prompt_encoder = MLPPromptEncoderFiLM(name = name,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                nl = non_linear,
                in_dim = in_dim,
                out_dim = out_dim,
                is_source = is_source,
                num_tasks = num_tasks,
                num_layers=num_layers, 
                enc_type=encoder_type,
                hidden_size=hidden_size,
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                )
        else:
            prompt_encoder = MLPPromptEncoder(name = name,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                nl = non_linear,
                in_dim = in_dim,
                out_dim = out_dim,
                is_source = is_source,
                num_tasks = num_tasks,
                num_layers=num_layers, 
                enc_type=encoder_type,
                hidden_size=hidden_size,
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                )
    elif encoder_type.startswith("emb"):
        prompt_encoder = EmbeddingPromptEncoder(name = name,
                model=model, tokenizer=tokenizer,
                length = length,
                is_source = is_source,
                num_tasks = num_tasks,
                enc_type=encoder_type,
                prompt_tokens=prompt_tokens, 
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                )

    elif encoder_type.startswith("mat"):
        prompt_encoder = MatPromptEncoder(
                n_prompts= 1, #len(prompt_tokens),
                intrinsic_dim=300,
                temperature=5,
                name = name, 
                length = length,
                is_source = is_source,
                num_tasks = num_tasks,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                shared_mat=shared_mat) 
    else:
        _enc_type = encoder_type.split("@")
        num_layers = 1
        hidden_size = -1
        if len(_enc_type) > 1:
            num_layers = int(_enc_type[1])
        if len(_enc_type) > 2:
            hidden_size = int(_enc_type[2])
        prompt_encoder = LSTMEmbeddingPromptEncoder(
                name = name, 
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                is_source = is_source,
                num_tasks = num_tasks,
                num_layers=num_layers, 
                task_emb_dim=task_emb_dim,
                task_compose_method=task_compose_method,
                hidden_size=hidden_size)
    # prompt_encoder.enc_type = encoder_type
    return prompt_encoder, encoder_type


