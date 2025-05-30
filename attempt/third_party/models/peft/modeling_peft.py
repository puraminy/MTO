import torch
import math
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions import Gumbel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, PreTrainedModel
from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, PeftModel
import attempt.mylogs as mylogs
from attempt.callbacks import WBCallback
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import * 
from entmax import sparsemax

class AnnealLambda:
    def __init__(self, module, lambda_start=0.0, lambda_max=0.1, warmup_steps=5000, target_entropy=0.5, entropy_tolerance=0.1):
        self.module = module
        self.lambda_start = lambda_start
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps
        self.target_entropy = target_entropy
        self.entropy_tolerance = entropy_tolerance
        self.current_lambda = lambda_start

    def anneal(self, step):
        module = self.module
        if step < self.warmup_steps:
            self.current_lambda = self.lambda_start + (self.lambda_max - self.lambda_start) * (step / self.warmup_steps)
        else:
            # Adaptive scaling based on entropy deviation
            if hasattr(module, "router"):
                entropy_value = relaxed_bernoulli_entropy(module.router).item()

                if entropy_value < self.target_entropy - self.entropy_tolerance:
                    self.current_lambda *= 1.05  # Increase λ if entropy is too low
                elif entropy_value > self.target_entropy + self.entropy_tolerance:
                    self.current_lambda *= 0.95  # Decrease λ if entropy is too high

        # Ensure lambda stays within bounds
        self.current_lambda = min(self.lambda_max, max(0.0, self.current_lambda))
        return self.current_lambda

import numpy as np

class AnnealTemperatureDynamic:
    def __init__(self, module, temp_start=1.0, temp_min=0.1, warmup_steps=5000, 
                 target_entropy=0.5, entropy_tolerance=0.1, anneal_type="exp"):
        self.module = module
        self.temp_start = temp_start
        self.temp_min = temp_min
        self.warmup_steps = warmup_steps
        self.target_entropy = target_entropy
        self.entropy_tolerance = entropy_tolerance
        self.current_temp = temp_start
        self.anneal_type = anneal_type

    def anneal(self, step):
        module = self.module

        # Warm-up phase (gradual reduction)
        if step < self.warmup_steps:
            self.current_temp = self.temp_start - (self.temp_start - self.temp_min) * (step / self.warmup_steps)
        else:
            # Adaptive control based on entropy deviation
            if hasattr(module, "router"):
                entropy_value = relaxed_bernoulli_entropy(module.router).item()

                if entropy_value < self.target_entropy - self.entropy_tolerance:
                    self.current_temp *= 1.05  # Increase temperature (encourage smoother probabilities)
                elif entropy_value > self.target_entropy + self.entropy_tolerance:
                    self.current_temp *= 0.95  # Decrease temperature (sharpen probabilities)

        # Ensure temperature stays within bounds
        self.current_temp = max(self.temp_min, self.current_temp)
        return self.current_temp


class Anneal:
    def __init__(self, temperature, anneal_dir, anneal_rate, anneal_min, anneal_type="exp"):
        self.cur_step = 0
        if anneal_dir == 1:
            self.start_val = anneal_min
            self.anneal_point = temperature
        elif anneal_dir == -1:
            self.start_val = temperature
            self.anneal_point = anneal_min
        elif anneal_dir == 0:
            self.start_val = temperature
            self.anneal_point = temperature

        self.cur_val = self.start_val
        self.anneal_dir = anneal_dir
        self.anneal_rate = anneal_rate
        self.anneal_type = anneal_type

    def __iter__(self):
        return self

    def anneal(self, i_step):
        if self.anneal_dir == 0:
            return self.cur_val
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

def normalize_scores(scores, method="soft", 
        sel_thresh=None, 
        gen_thresh_min=None, 
        gen_thresh_max=None, 
        resample=False, is_training=False, temperature=1):

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
    elif method == "sparse":
        scores = sparsemax(scores, dim=-1)
    elif method == "direct" or method == "soft" or method == "srelu":
        if is_training:
            #scores=scores / scores.sum(dim=-1, keepdim=True) 
            # Add a small positive constant to the scores
            epsilon = 1e-8  # Adjust the value of epsilon as needed
            scores += epsilon
        scores = F.softmax(scores, -1)
    elif method == "softemp":
        scores = F.softmax(scores / temperature, -1)
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
    elif method == "sigtemp":
        scores = torch.sigmoid(scores / temperature)  
    elif method == "sign":
        scores[scores <= 0] = 0
        scores[scores > 0] = 1
    elif method == "relu":
        scores[scores <= 0] = 0
    elif method == "spos":
        scores = F.softmax(scores, -1)
        scores[scores > 0.3] = 1
    elif method == "soft_pos":
        mask = (scores > 0).float()
        masked_scores = scores.masked_fill(mask < -1, -200)
        if is_training:
            #scores=scores / scores.sum(dim=-1, keepdim=True) 
            # Add a small positive constant to the scores
            epsilon = 1e-8  # Adjust the value of epsilon as needed
            masked_scores += epsilon
        scores = F.softmax(masked_scores, -1)
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

class AttnConfig:
    pass

# === Custom Attentive Prompt Embedding === #
class AttentivePromptEncoder(torch.nn.Module):
    def __init__(self, config, adapter_config=None, embed_tokens=None, prefix_emb=None, attn_tuning=False, mul_prefix_emb=None, shared_attn=False, attend_target=False, learned_temperature=False):
        super().__init__()

        self.embed_tokens = embed_tokens
        self.adapter_config = adapter_config
        ################# MyCode
        self.prompt_encoders = []
        self.use_private_prompts = False
        self.embedding_dim = model_dim = config.d_model
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
        # all prompt ids for all tasks
        self.common_prompt_ids = [] 
        self.task_prompt_ids = []
        self.router = None
        self.init_router = None
        self.training = True
        self.pred_task = ""
        self.prompt_dim = None
        self.soft_prompts = None
        self.task_prompt = None
        self.gen_conf = None
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
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
        self.private_bias = config.private_bias
        self.anneal_min = config.anneal_min
        self.anneal_dir = config.anneal_dir
        self.anneal_rate = config.anneal_rate
        self.anneal_type = config.anneal_type
        self.temperature = config.temperature
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))    

        self.lambda_entropy = config.lambda_entropy
        self.entropy_loss = 0
        self.cont_loss = 0
        self.do_entropy_loss = config.do_entropy_loss
        self.do_cont_loss = config.do_cont_loss
        self.anneal_router_entropy = AnnealLambda(module=self, 
                lambda_start=self.lambda_entropy,
                warmup_steps=config.warmup_steps)

        self.aneal_router_temperature_dynamic = AnnealTemperatureDynamic(module=self,
                temp_start=self.temperature,
                temp_min=self.anneal_min, 
                warmup_steps=config.warmup_steps)

        self.aneal_router_temperature = Anneal(self.temperature, 
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
        self.attn_method = config.attn_method
        self.gate_method = config.gate_method
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
        if True: #self.append_attn_prefix: # or self.prompt_tuning:
            if self.gate_method == "linear" or self.compose_method == "lin":
                self.attn_Wa = nn.Linear(
                    self.model_dim, self.model_dim, bias=False)
            if self.gate_method == "sub" or self.compose_method == "sub":
                self.attn_W_down = nn.Linear(self.model_dim, 300, bias=False)
                self.attn_W_up = nn.Linear(300, self.model_dim, bias=False)
                self.attn_non_linear = nn.SiLU()
                self.layer_norm = nn.LayerNorm(self.model_dim)
        #######################################
        self.adapter_config = adapter_config

    def set_encoders(self, prompt_encoders, source_prompts, 
            src_prompt_dim, prompt_dim, pvt_prompt_dim =None, tasks = None):
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
        self.pvt_prompt_dim = pvt_prompt_dim
        self.num_tasks = len(tasks)

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
            b = self.bias
            if type(self.bias) == str and "_" in self.bias:
                _pos, b = self.bias.split("_")
            if self.bias is not None and int(b) != 0:
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
                            if type(self.bias) == str and "_" in self.bias:
                                _pos, b = self.bias.split("_")
                            if _pos == "x" or _pos == "s":
                                router[i, j] = float(b)
                                j += 1
                        if k > 1:
                            _pos, b = "s", self.bias
                            if type(self.bias) == str and "_" in self.bias:
                                _pos, b = self.bias.split("_")
                            if _pos == "x" or _pos == "p":
                                router[i, k] = float(b) 
                                k += 1
                    i += 1
                mylogs.bp("bias")
            self.router = nn.Parameter(data=router, requires_grad=True)
        comp_config = {
            "attn_method": self.attn_method,  # or "cross_attn", "rb", "const"
            "compose_method": self.compose_method,  # or "wmp", "wcp", "pool", etc.
            "model_dim": self.model_dim,
            "num_sources": self.num_src_encoders,
            "num_tasks": len(tasks) 
        }
        self.layer_norm_attn = nn.LayerNorm(normalized_shape=self.num_src_encoders) 
        comp_config = dotdict(comp_config.copy())
        self.attn_mask_orig = self.attn_mask.clone()
        self.source_encoders_idx = torch.tensor(src_list, device=device)
        self.target_encoders_idx = torch.tensor(tgt_list, device=device)

        self.target_prompt_ids = torch.tensor(target_prompt_ids, device=device)
        self.common_prompt_ids = torch.tensor(common_prompt_ids, device=device)
        self.task_prompt_ids = torch.tensor(task_prompt_ids, device=device)
        intrinsic_dim = 200
        if self.target_share is not None:
            b = 0 # self.target_share  # or any other float
            #self.target_router = nn.Linear(..., bias=True)
            #self.target_router.bias.data.fill_(-2.0)
            self.target_router = nn.Parameter(
               torch.full((attend_num,), float(b), device=device)
            )

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
        if True: # "any" in mask_type:
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
         if self.anneal_type == "dyn" and not self.anneal_dir == 0:
             self.temperature = self.aneal_router_temperature_dynamic.anneal(i_step)
         else:
             self.temperature = self.aneal_router_temperature.anneal(i_step)
         if self.do_entropy_loss is True:
             self.lambda_entropy = self.anneal_router_entropy.anneal(i_step)
         if self.do_anneal_thresh is True:
             self.sel_thresh = self.anneal_thresh.anneal(i_step)

    def update_entropy_loss(self, probs):
        loss = 0
        if self.do_entropy_loss is True:
            router_entropy_loss = entropy_loss(probs)
            lambda_entropy = self.lambda_entropy # 0.1  # Weight for entropy penalty
            loss = lambda_entropy * router_entropy_loss  # Add entropy penalty
        self.entropy_loss = loss

    ################# MyCode fffffffffff
    def attend_input_score(self, inputs_embeds, # src_prompts, 
            attend_to, att_mask=None):
        batch_size = inputs_embeds.shape[0]
        avg_attend_for = self.get_avg_inputs_embeds(inputs_embeds, att_mask)
        avg_attend_to, _ = torch.max(attend_to, 2)
        if self.gate_method == "dot":
            x = torch.transpose(avg_attend_to, 1,2)
            attn_scores = avg_attend_for.bmm(x)
        elif self.gate_method == "linear":
            x = self.attn_Wa(avg_attend_to)
            x = self.layer_norm(x)
            x = torch.transpose(x, 1,2)
            attn_scores = avg_attend_for.bmm(
                x) / self.temperature
        elif self.gate_method == "sub":
            x = self.attn_W_down(avg_attend_to)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)
            #x = x.unsqueeze(-1) ###
            x = torch.transpose(x, -2, -1)
            attn_scores = avg_attend_for.bmm(
                x) / self.temperature
        # implement token level model
        elif self.gate_method == "proj":
            query = self.query_proj(avg_attend_to)  # [B, 1, D]
            key = self.key_proj(avg_attend_for)       # [B, N, D]
              # Project query and key
            attn_scores = (torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.model_dim)) #self.temperature  # [B, 1, N]
        elif self.gate_method == "token":
            x = self.attn_W_down(avg_attend_to)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)
            x = x.unsqueeze(-1)
            attn_scores = torch.einsum(
                "bpld,bdk->bplk", attend_for, x) / self.temperature
        else:
            raise NotImplementedError

        alpha = torch.sigmoid(attn_scores).view(batch_size)  # [B]
        return alpha

    def get_avg_inputs_embeds(self, input_embeds, attention_mask):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (input_embeds * mask).sum(dim=1)    # [B, D]
            counts = mask.sum(dim=1) + 1e-6              # avoid division by zero
            avg_inputs_embeds = summed / counts          # [B, D]
        else:
            avg_inputs_embeds = input_embeds.mean(dim=1)  # [B, D]

        avg_inputs_embeds = avg_inputs_embeds.unsqueeze(1)
        return avg_inputs_embeds

    def attend_prompts(self, inputs_embeds, src_prompts, 
            pvt_prompts=None, 
            source_idx=None, num_targets=1, 
            target_idx =None, task_ids=None, attn_mat=None, att_mask=None, task=""):
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

        if compose_method in ["wcp1","wsp1","wmp1","catp"]: # or self.ignore_private:
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompt = attend_to[:,-1,:,:]
            private_idx = attend_to_idx[:,-1] 
            if pvt_prompts is not None:
                private_prompt = torch.stack(
                    [pvt_prompts[int(idx.item())] for idx in private_idx], dim=0
                )
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
        route_idx = attend_to_idx
        router = torch.zeros(target_idx.size(1),
                route_idx.size(1), 
                device=inputs_embeds.device)
        router = router.repeat(batch_size, 1, 1)
        for i in range(batch_size):
            router[i] = self.router[target_idx[i].reshape(-1,1), 
                                route_idx[i]]
        if self.attn_method in ["sub","linear","dot","token"]:
            attn_scores = self.attend_input_score(inputs_embeds, 
            attend_to, att_mask)
        elif "gated" in self.attn_method:
            avg_inputs_embeds = self.get_avg_inputs_embeds(inputs_embeds, att_mask)
            prompt_summary = attend_to.mean(dim=2)  # [B, N, D]
            # Register these once in __init__, don't redefine here every forward
            # prompt_summary = self.layer_norm(prompt_summary)
            query = self.query_proj(avg_inputs_embeds)  # [B, 1, D]
            key = self.key_proj(prompt_summary)       # [B, N, D]
              # Project query and key
            attn_scores = (torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.model_dim)) #self.temperature  # [B, 1, N]
            if "rb" in self.attn_method:
                attn_scores = RelaxedBernoulli(temperature=self.temperature, 
                    logits=attn_logits).rsample()            
            elif False:
                attn_weights = F.softmax(attn_logits, dim=-1)
                attn_scores = attn_weights
        elif self.attn_method == "const":
            router = torch.ones(target_idx.size()[1],
                    route_idx.size()[1], 
                    device=inputs_embeds.device)
            router = router.repeat(batch_size, 1, 1)
            attn_scores = router
        elif self.attn_method == "params":
            attn_scores = router
        elif self.attn_method == "rb":
            mylogs.bp("rmconst")
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
                # route_method = "rb" #TODO fix this fixed assignment
                if route_method == "rb" or route_method == "soft":
                    attn_scores = RelaxedBernoulli(temperature=self.temperature, 
                        logits=logits).rsample()            
                elif route_method == "gumb":
                    gumb_scores = Gumbel(0, 1).sample(router.shape).to(router.device)
                    attn_scores = (router + gumb_scores) / self.temperature 
                elif route_method == "const":
                    attn_scores  = attn_dist
                    self.norm_method = "nothing"
                elif route_method == "importance":
                    col_sums = torch.sum(router, dim=0)
                    attn_scores = rb_scores * col_sums
                else:
                    raise ValueError("Not recognized route method:" + route_method)
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
        if self.private_bias > 0:
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            bias_ratio = self.private_bias
            logit_bias = torch.log(torch.tensor(bias_ratio / (1 - bias_ratio))) 
            attn_scores[:, :, -1] += logit_bias

        mylogs.bp("before")
        if self.training and "before" in self.norm_method and self.attn_method != "const":
            method = self.norm_method.replace("before_","")
            attn_scores = normalize_scores(attn_scores, method, 
                    is_training=self.training, temperature=self.temperature) 

        self.update_entropy_loss(attn_scores)
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
        if task:
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
            # TODO and False
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
                    sel_thresh=self.sel_thresh, is_training=self.training, 
                    temperature=self.temperature)

        mylogs.bp("params")
        if route_method == "params":
            # attn_sel_scores = attn_sel_scores.new_ones(attn_sel_scores.shape)
            # attn_sel_scores = torch.ones_like(attn_sel_scores, requires_grad=True)
            pass

        # target_shares = torch.ones(1, batch_size, device=device)

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
                #assert torch.all(attn_sel_scores == 1), "Attention scores must be all one"
                pass
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
        elif compose_method in ["wsp1","wmp1","wcp1", "catp"]:
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', attn_sel_scores, 
                    attend_to_x)
            if self.target_share == 0:
                alpha = self.attend_input_score(inputs_embeds, private_prompt, att_mask)
                alpha = alpha.reshape(batch_size, 1, 1, 1)
                beta = self.attend_input_score(inputs_embeds, avg_prompts, att_mask)
                beta = beta.reshape(batch_size, 1, 1, 1)
                combined = torch.cat([alpha, beta], dim=1)  # shape: (batch_size, 2, 1, 1)
                weights = F.softmax(combined, dim=1)      
                alpha, beta = weights[:, 0:1], weights[:, 1:2]  
            else:
                alpha = self.get_target_shares(attn_sel_scores, private_idx, batch_size)
                alpha = alpha.reshape(batch_size, 1, 1, 1)
                beta = self.get_target_shares(attn_sel_scores, target_idx, batch_size)
                beta = beta.reshape(batch_size, 1, 1, 1)
            if compose_method == "wsp1": 
               #alpha = torch.sigmoid(self.alpha_raw)  # shape: scalar in (0, 1)
               #soft_prompts = (1-alpha) * avg_prompts + alpha * private_prompt 
               if self.target_share == -1:
                   soft_prompts = (1 - alpha) * avg_prompts + alpha * private_prompt 
               elif self.target_share < -4 or self.target_share == 0:
                   soft_prompts = beta * avg_prompts + alpha * private_prompt 
               else:
                   soft_prompts =  avg_prompts + private_prompt 
            elif compose_method == "wmp1": 
                   soft_prompts = avg_prompts * private_prompt 
            elif compose_method == "wcp1": 
               if self.target_share >= 1:
                   soft_prompts = torch.cat([avg_prompts, private_prompt], dim=2)
               elif alpha is not None:
                   soft_prompts = torch.cat([beta * avg_prompts, alpha * private_prompt], dim=2)
               else:
                   soft_prompts = torch.cat([avg_prompts, private_prompt], dim=2)
            elif compose_method == "catp": 
                soft_prompts = torch.cat([attend_to_x.squeeze(1), private_prompt], dim=2)
            attn_sel_scores = torch.cat(
                   [attn_sel_scores, alpha.reshape(batch_size, 1, 1)], dim=-1)
            #alpha_scores = alpha.expand(batch_size, 1, 1)  # [B, 1, 1]
            #attn_sel_scores = torch.cat([attn_sel_scores, alpha_scores], dim=-1)
            attend_to_idx = torch.cat([attend_to_idx, private_idx.unsqueeze(1)], dim=-1) 
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
            ts = attn_sel_scores[:,:,-1]
            ts = ts.reshape(batch_size, 1, 1, 1)
            if self.target_share == 1:
               soft_prompts = avg_prompts + private_prompts 
            elif self.target_share == 2:
               soft_prompts = avg_prompts + ts * private_prompts 
            elif self.target_share == 3:
               soft_prompts = avg_prompts + ts * (private_prompts - avg_prompts)
            elif self.target_share == 4:
               soft_prompts = private_prompts + (avg_prompts - private_prompts)
            else:
               soft_prompts = (1 - ts) * avg_prompts + (ts * private_prompts) 
        elif compose_method == "wcp":
            mylogs.bp("wcp")
            s_attn_sel_scores = attn_sel_scores[:,:,:-1]
            s_attend_to_x = attend_to_x[:,:,:-1,:,:]
            assert self.use_private_prompts is True, "use private prompts must be enabled"
            private_prompts = attend_to_x[:,:,-1,:,:]
            avg_prompts = torch.einsum(
                    'bts, btsld -> btld', s_attn_sel_scores, 
                    s_attend_to_x)
            if self.target_share == 2:
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
    
    def get_target_shares(self, attn_sel_scores, target_idx, batch_size):
        target_shares = None
        device = attn_sel_scores.device
        if self.target_share is not None:
            if self.target_share == -1 or self.target_share < -4:
                target_router = self.target_router.unsqueeze(0)
                target_router = batched_index_select(target_router, 1, target_idx)
                if self.target_share == -10:
                    target_shares = target_router
                else:
                    if self.training:
                        tst = self.target_share_temperature
                        # tst = self.temperature
                        target_shares = torch.sigmoid(target_router) # * self.sig_coef) 
                        #target_shares = RelaxedBernoulli(temperature=tst, 
                        #    logits=target_router).rsample()            
                    else:
                        target_shares = torch.sigmoid(target_router) # * self.sig_coef) 
                        # target_shares = F.softmax(target_router, dim=-1)
                        # target_shares = RelaxedBernoulli(temperature=0.01, 
                        #    logits=target_router).rsample()            
            elif self.target_share == -2:
                top, _ = torch.max(attn_sel_scores, -1) 
                target_shares = top.transpose(0,1)
            elif self.target_share == -3:
                top, _ = torch.max(attn_sel_scores, -1) 
                target_shares = 1 - top.transpose(0,1)
            elif self.target_share == -4:
                top = torch.mean(attn_sel_scores, -1) 
                target_shares = 1 - top.transpose(0,1)
            elif self.target_share >= 1:
                target_shares = torch.ones(1, batch_size, device=device)
            else:
                target_shares = self.target_share * torch.ones(1, batch_size, device=device)
        return target_shares

    def add_target_prompts(self, target_prompts, soft_prompts, 
            attn_sel_scores, attend_to_idx, target_idx):
       batch_size = soft_prompts.shape[0]
       device = soft_prompts.device
       mylogs.bp("cmm")
       target_shares = self.get_target_shares(attn_sel_scores, target_idx, batch_size)
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
    def prompt_encoders_forward(self, input_ids, inputs_embeds, 
            att_mask, task_ids=None, task=None):
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
            self.prompt_masks = target_prompt_masks
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
            pvt_prompts = {}  # Dictionary keyed by encoder.src_idx
            src_prompts_dict = {}
            ii = 1
            for encoder in self.prompt_encoders:
                if encoder.is_source: # and self.use_source_prompts:
                    source_idx_list.append(ii)
                    emb = encoder(encoder.net_inps)
                    src_prompts_dict[encoder.src_idx] = emb
                    if self.pvt_prompt_dim == self.src_prompt_dim:
                        src_prompts[encoder.src_idx, :] = emb
                    if encoder.is_private:
                        pvt_prompts[encoder.src_idx] = emb
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
                if self.attn_prompt_tuning: #TODO and not self.target_share == 1:
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
                                    pvt_prompts = pvt_prompts,
                                    source_idx=source_idx, 
                                    target_idx=target_idx, 
                                    task_ids = tids,
                                    attn_mat = attn_mat,
                                    att_mask = att_mask,
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
                                self.task_prompt = soft_prompts[0]
                            ###### Pad extra prompt tokens
                            # amask = amask.squeeze(1)
                            masked_prompts = soft_prompts
                            tmask = target_prompt_masks.clone()
                            amask = torch.ones((batch_size, 
                                attn_scores.size(-1)*self.src_prompt_dim), dtype=bool)
                            ignore_zeros = False
                            if not self.training:
                                ignore_zeros = self.gen_conf.get("ignore_zeros", True)
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

                            # inputs_embeds[tmask]= masked_prompts.view(-1, self.model_dim)
                            inputs_embeds = inputs_embeds.clone()  # Avoid in-place modification
                            inputs_embeds[tmask] = masked_prompts.view(-1, self.model_dim)

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
                            inputs_embeds = inputs_embeds.clone()  # Avoid in-place modification
                            self.soft_prompts=target_prompts.view(-1, 
                                    self.model_dim)
                            inputs_embeds[target_prompt_masks]= target_prompts.view(-1, 
                                    self.model_dim)
                    else:
                        self.soft_prompts=target_prompts.view(-1, 
                                self.model_dim)
                        inputs_embeds = inputs_embeds.clone()  # Avoid in-place modification
                        inputs_embeds[target_prompt_masks]= target_prompts.view(-1, 
                                self.model_dim)
                else:
                    self.adapter_config.soft_prompts=target_prompts.view(-1, self.model_dim)
                    inputs_embeds[target_prompt_masks]=target_prompts.view(-1, 
                            self.model_dim)
            return input_ids, att_mask , inputs_embeds
        return input_ids, att_mask, inputs_embeds



class PTModel(PeftModel):
    def __init__(self, nested_model, peft_config, attn_pt=None):
        super().__init__(nested_model, peft_config)
        self.nested_model = nested_model
        self.encoder = attn_pt 

    def forward(self, input_ids, 
        inputs_embeds=None, 
        attention_mask=None, 
        task_embedding=None,
        task_ids=None,
        task=None,
        **kwargs):
        embedding_layer = self.nested_model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        mylogs.bp("fwd")
        if self.encoder is not None and self.training:
            input_ids, attention_mask, inputs_embeds = \
                    self.encoder.prompt_encoders_forward(
                        input_ids, inputs_embeds, 
                        #task_ids, 
                        #task, 
                        att_mask = attention_mask)

        # inputs_embeds = torch.cat([attentive_embeddings, input_embeddings], dim=1)
        return self.nested_model(inputs_embeds=inputs_embeds, 
                            attention_mask=attention_mask, **kwargs)


from transformers import PreTrainedModel
from typing import Optional, Dict, Any
import torch
from transformers import Trainer
import torch

import torch

def relaxed_bernoulli_entropy(router_logits):
    """
    Compute entropy for relaxed Bernoulli probabilities.

    Args:
        router_logits (torch.Tensor): Learnable routing logits (before sigmoid).

    Returns:
        torch.Tensor: Entropy loss value.
    """
    probs = torch.sigmoid(router_logits)  # Convert logits to probabilities
    eps = 1e-8  # Avoid log(0)

    # Compute entropy: H(p) = - p log(p) - (1 - p) log(1 - p)
    entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)

    return entropy.mean()  # Return mean entropy loss

def entropy_loss(attention_weights):
    """
    Compute entropy loss to encourage diverse attention distribution.

    Args:
        attention_weights (torch.Tensor): Attention scores of shape (batch_size, num_heads, seq_len, seq_len)

    Returns:
        torch.Tensor: Entropy loss value
    """
    eps = 1e-8  # Avoid log(0)
    attention_probs = attention_weights.mean(dim=1)  # Average over attention heads
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + eps), dim=-1)  # Compute entropy per batch
    return entropy.mean()  # Return average entropy loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, model, alpha=0.9, beta=0.05, temperature=0.07, cls=False, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for router loss
        self.temperature = temperature  # Scaling factor for contrastive loss
        self.task_labels = []
        self.is_classifier = cls
        self.max_new_tokens = 10
    
    def _remove_unused_columns(self, dataset, description=None):
        # if description != 'training':
        #    return dataset
        keep_columns = {'input_ids', 'attention_mask', 'labels', 'task_id'}
        
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in keep_columns])
        
        return dataset

    def predict(
        self,
        test_dataset,
        task_names: Optional[List[str]] = None,
        task_labels: Optional[List[Union[int, float]]] = None,
        **kwargs
    ):
        """Override predict() to accept task information while maintaining original behavior"""

        # Store task info in dataset object (temporary, doesn't modify original dataset)
        if task_names is not None:
            test_dataset.task_names = task_names
        if task_labels is not None:
            test_dataset.task_labels = task_labels
        self.task_labels = test_dataset[0]["extra_fields"]["labels_list"]

        # Call original predict() with all its normal functionality
        return super().predict(test_dataset, **kwargs)

    def compute_loss_2(self, model, inputs, return_outputs=False, **kwargs):
        """
        Unified loss computation that handles:
        - Encoder-decoder (T5/BART) - gets encoder hidden states
        - Decoder-only (GPT) - gets final hidden states
        - Encoder-only (BERT) - extracts embeddings directly when needed
        With support for:
        - Prompt-based contrastive loss
        - Router loss
        """
        # Input preparation
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        task_ids = inputs.get("task_id", None)
        batch_size = input_ids.size(0)

        # Initialize losses
        gen_loss = cont_loss = router_loss = 0.0
        prompt_embeddings = None

        # Base forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True  # Request hidden states for all models
        )
        gen_loss = outputs.loss

        prompt_masks = model.encoder.get_target_prompt_ids_mask(input_ids)

        if prompt_masks is not None:
            # Handle different model architectures
            if hasattr(model.nested_model, 'encoder'):  # Encoder-decoder models
                embeddings = outputs.encoder_hidden_states[-1]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:  # Most models
                embeddings = outputs.hidden_states[-1]
            else:  # Fallback for BERT-style classifiers
                # Access the embedding layer directly
                embeddings = model.nested_model.get_input_embeddings()(input_ids)
                # Add positional embeddings if they exist
                if hasattr(model.nested_model, 'embeddings') and hasattr(model.nested_model.embeddings, 'position_embeddings'):
                    position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                    embeddings += model.nested_model.embeddings.position_embeddings(position_ids)

            # Extract only the prompt token embeddings
            prompt_embeddings = embeddings[prompt_masks].reshape(
                batch_size, -1, embeddings.size(-1))

        # Compute contrastive loss if we have prompt embeddings
        if prompt_embeddings is not None:
            cont_loss = self.compute_contrastive_loss(model, prompt_embeddings, task_ids)

        router_loss = self.compute_router_loss(model, input_ids, labels)

        # Combine losses
        total_loss = gen_loss + self.alpha * cont_loss + self.beta * router_loss

        return (total_loss, outputs) if return_outputs else total_loss


    def compute_contrastive_loss(self, model, prompt_reps, task_ids):
        """
        Args:
            prompt_reps: Tensor of shape [batch_size, num_prompts, hidden_dim]
            task_ids: Tensor of shape [batch_size] containing task IDs
        """
        if not model.encoder.do_cont_loss:
            cont_loss = 0
            model.encoder.cont_loss = cont_loss
            return cont_loss

        batch_size, num_prompts, hidden_dim = prompt_reps.shape

        # Normalize representations
        prompt_reps = F.normalize(prompt_reps, p=2, dim=-1)  # [bs, num_prompts, dim]

        # Compute similarity between all prompts
        flat_reps = prompt_reps.view(-1, hidden_dim)  # [bs*num_prompts, dim]
        sim_matrix = torch.mm(flat_reps, flat_reps.t()) / self.temperature  

        # Create task mask (1 where same task, 0 otherwise)
        expanded_task_ids = task_ids.repeat_interleave(num_prompts)  # [bs*num_prompts]
        mask = (expanded_task_ids.unsqueeze(1) == expanded_task_ids.unsqueeze(0)).float()

        # Remove self-similarity
        mask.fill_diagonal_(0)

        # NT-Xent loss calculation
        exp_sim = torch.exp(sim_matrix)
        pos_sim = exp_sim * mask
        neg_sim = exp_sim * (1 - mask)

        contrastive_loss = -torch.log(
            pos_sim.sum(dim=-1) / (pos_sim.sum(dim=-1) + neg_sim.sum(dim=-1) + 1e-9)
        )
        cont_loss = contrastive_loss.mean()
        model.encoder.cont_loss = cont_loss

        return cont_loss

    def compute_router_loss(self, model, input_ids, labels):
        """
        Compute the model loss with entropy regularization on router logits.

        Args:
            model: The transformer model.
            inputs: Input batch dictionary.
            return_outputs (bool): Whether to return model outputs.

        Returns:
            Loss value (with entropy regularization).
        """
        if model.encoder.do_entropy_loss:
            return model.encoder.entropy_loss  # Add entropy penalty
        return 0

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Perform a prediction step that supports:
        - Encoder-decoder models (e.g., T5, BART)
        - Encoder-only models (e.g., BERT for classification/regression)
        - Decoder-only models (e.g., GPT for generation)
        """
        # Ensure the model is in evaluation mode
        model.eval()

        # Extract common inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels", None)

        # Determine model type
        is_encoder_decoder = (hasattr(model.config, "is_encoder_decoder") 
                and model.config.is_encoder_decoder and not self.is_classifier)

        is_decoder_only = (hasattr(model.config, "is_decoder") 
                and model.config.is_decoder and not self.is_classifier)
        logits_processor = [TaskAwareLogitsProcessor(self.task_labels, self.processing_class)]

        with torch.no_grad():
            if is_encoder_decoder:
                # Handle encoder-decoder models (T5, BART, etc.)
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                    repetition_penalty=2.0,
                    # logits_processor=logits_processor
                )
                outputs = generated_tokens
                loss = None

                if prediction_loss_only and labels is not None:
                    # Calculate loss if requested (not typically done during generation)
                    model_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = model_outputs.loss
            elif is_decoder_only:
                # Handle decoder-only models (GPT, etc.)
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=1,
                    max_new_tokens=1 if self.model.classifier is not None else 10,
                    repetition_penalty=2.0,
                    #logits_processor=logits_processor
                )
                outputs = generated_tokens
                loss = None

                if prediction_loss_only and labels is not None:
                    model_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = model_outputs.loss

            else:
                # Handle encoder-only models (BERT, etc.)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if type(outputs) == dict:
                    logits = outputs["logits"]
                else:
                    logits = outputs.logits

                # Determine output type based on logits shape
                if logits.dim() == 2:  # Classification (batch_size, num_labels)
                    predicted_tokens = logits.argmax(dim=-1)
                elif logits.dim() == 1:  # Regression (batch_size,)
                    predicted_tokens = logits
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                outputs = predicted_tokens
                loss = None

                if prediction_loss_only and labels is not None:
                    loss_fct = torch.nn.CrossEntropyLoss() if logits.shape[-1] > 1 else torch.nn.MSELoss()
                    loss = loss_fct(
                        logits.view(-1, logits.shape[-1]) if logits.shape[-1] > 1 else logits.view(-1),
                        labels.view(-1)
                    )

        return (loss, outputs, labels)


from transformers import LogitsProcessor

class TaskAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, task_labels, tokenizer):
        """
        task_labels: List of labels like ["positive", "negative"]
        tokenizer: Tokenizer to get token IDs
        """
        # Convert label words to token IDs correctly
        token_ids = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in task_labels]
        self.allowed_tokens = list(set([tid for sublist in token_ids for tid in sublist]))  # Flatten & remove duplicates
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.append(tokenizer.eos_token_id)


    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))  # Initialize all as -inf
        mask[:, self.allowed_tokens] = 0  # Allow only task-specific tokens
        return scores + mask

def get_model_dimension(config):
    # Check for common attributes that represent the model's hidden size
    if hasattr(config, "d_model"):  # For models like T5, BART, Marian
        return config.d_model
    elif hasattr(config, "hidden_size"):  # For models like GPT-2, BERT, RoBERTa
        return config.hidden_size
    elif hasattr(config, "n_embd"):  # For GPT-2 specifically
        return config.n_embd
    elif hasattr(config, "dim"):  # For some other architectures
        return config.dim
    else:
        raise AttributeError(f"Could not determine model dimension for {model_name_or_path}. Config: {config}")

import torch
import torch.nn as nn
from transformers import PreTrainedModel

class CustomModelWrapper(PreTrainedModel):
    def __init__(self, nested_model, base_config, attn_pt=None, cls=False, num_labels=2):
        super().__init__(base_config)
        self.nested_model = nested_model
        self.encoder = attn_pt  # Optional prompt tuning module
        self.classifier = None
        if cls is True:
            self.classifier = nn.Linear(base_config.d_model, num_labels)  # Classification head

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:

        # Get the embedding layer from the base model
        embedding_layer = self.nested_model.get_input_embeddings()

        # Convert input_ids to embeddings if necessary
        if input_ids is not None:
            inputs_embeds = embedding_layer(input_ids)

        # Apply prompt tuning if enabled
        if self.encoder is not None and self.training:
            input_ids, attention_mask, inputs_embeds = \
                self.encoder.prompt_encoders_forward(
                    input_ids, inputs_embeds, att_mask=attention_mask
                )

        # Pass the inputs through the T5 encoder
        if self.classifier is None:
            # Pass the inputs to the base model
            outputs = self.nested_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            encoder_outputs = self.nested_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Take the last hidden state (first token or pooled representation)
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # CLS token representation

            # Apply classifier
            logits = self.classifier(pooled_output)

            outputs = {"logits": logits}

            # Compute loss if labels are provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                outputs["loss"] = loss

        return outputs
    def generate(self, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Generate sequences using the nested model.
        """
        self.nested_model.eval()
        # Apply prompt tuning if enabled
        if self.encoder is not None:
            self.encoder.training = False  # Ensure encoder is in eval mode

            # Get the embedding layer from the base model
            embedding_layer = self.nested_model.get_input_embeddings()

            # Convert input_ids to embeddings if provided
            if input_ids is not None:
                inputs_embeds = embedding_layer(input_ids)
            else:
                inputs_embeds = None

            # Apply prompt tuning
            input_ids, attention_mask, inputs_embeds = \
                self.encoder.prompt_encoders_forward(
                    input_ids, inputs_embeds, att_mask=attention_mask
                )

            # Pass inputs_embeds to generate
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            # If no prompt tuning, pass input_ids directly
            kwargs["input_ids"] = input_ids

        # Ensure attention_mask is passed
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        # Delegate generation to the nested model
        output = self.nested_model.generate(**kwargs)
        return output


class CustomModelWrapper_2(PreTrainedModel):
    def __init__(self, nested_model, base_config, attn_pt=None):
        super().__init__(base_config)
        self.nested_model = nested_model
        self.encoder = attn_pt
        d_model = get_model_dimension(base_config)
        self.classifier = nn.Linear(d_model, 3)  # Classification head

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                labels=None,
                **kwargs) -> Dict[str, Any]:
        embedding_layer = self.nested_model.get_input_embeddings()

        # Convert input_ids to embeddings if provided
        if input_ids is not None:
            inputs_embeds = embedding_layer(input_ids)

        # Apply prompt tuning if enabled
        if self.encoder is not None:
            input_ids, attention_mask, inputs_embeds = \
                self.encoder.prompt_encoders_forward(
                    input_ids, inputs_embeds, att_mask=attention_mask
                )

        # Forward pass
        if hasattr(self.nested_model, "encoder"):  # Likely T5
            outputs = self.nested_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]  # First token for classification

        else:  # Likely BERT
            outputs = self.nested_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            if hasattr(outputs, "last_hidden_state"):  # Some BERT variants might have it
                pooled_output = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "logits"):  # Standard BERT (SequenceClassifierOutput)
                pooled_output = outputs.logits  # Already has classification logits
            else:
                raise ValueError("Unknown model output structure.")

        logits = self.classifier(pooled_output) if pooled_output.dim() > 2 else pooled_output

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits} if labels is not None else logits

    def forward_2(self, input_ids: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                labels=None,
                **kwargs) -> Dict[str, Any]:
        
        # Get the embedding layer from the base model
        embedding_layer = self.nested_model.get_input_embeddings()
        if not self.training:
            pass
        # If input_ids are provided, convert them to embeddings
        if input_ids is not None:
            inputs_embeds = embedding_layer(input_ids)
        
        # Apply prompt tuning if enabled
        if self.encoder is not None:
            input_ids, attention_mask, inputs_embeds = \
                self.encoder.prompt_encoders_forward(
                    input_ids, inputs_embeds, att_mask=attention_mask
            )

        # Ensure decoder inputs are provided
        # Pass the inputs to the base model
        outputs = self.nested_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs


# === Setup for Training === #
def setup_training():
    # Load pre-trained model and tokenizer
    model_name_or_path = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    nested_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # PEFT configuration
    peft_config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=10,  # Define number of soft prompt tokens
        tokenizer_name_or_path=model_name_or_path
    )

    # Initialize AttentivePromptEncoder
    attn_prompt_embedding = AttentivePromptEncoder(peft_config)

    # Initialize custom model with attentive prompt embedding
    model = PTModel(nested_model, peft_config, attn_prompt_embedding)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available

    return model, tokenizer

# === Training Setup === #
def train_model(model, tokenizer):
    # Load dataset (GLUE MRPC)
    dataset = load_dataset("glue", "mrpc")
    
    # Preprocess the dataset for model input
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True, max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Setup data collator and training arguments
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model.nested_model)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    return model

# === Inference Function (After Training) === #
def generate_output(model, tokenizer, input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Ensure tensors are on the correct device

    # Forward pass with the trained model
    outputs = model.generate(input_ids=inputs["input_ids"], inputs_embeds=inputs["inputs_embeds"])

    # Decode and print output
    print("Generated Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# === Full Training and Inference Example === #
if __name__ == "__main__":
    # Setup model and tokenizer
    model, tokenizer = setup_training()

    # Train the model
    trained_model = train_model(model, tokenizer)

    # Inference with generated output
    input_text = "Translate English to French: Hello, how are you?"
    generate_output(trained_model, tokenizer, input_text)

