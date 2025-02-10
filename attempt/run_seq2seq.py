# branch main
# version vv
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import sys

#sys.path.append('/home/ahmad/ATTEMPT')

from utils import * 
import shutil
from pathlib import Path
import glob
from data import AutoPostProcessor
# from third_party.models import T5Config, T5ForConditionalGeneration
from third_party.models import PTModel, AttentivePromptEncoder 
from transformers import Trainer #, TrainingArguments, DataCollatorForSeq2Seq

from transformers import AutoModelForSeq2SeqLM
from peft import PromptTuningConfig, get_peft_model


from dataclasses import dataclass, field
from options import AdapterTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from third_party.trainers import Seq2SeqTrainer
from data import TaskDataCollatorForSeq2Seq
from data import AutoTask
import re
from rouge import Rouge
from utils import get_adapter_config
from attempt.utils.utils import combine_x,combine_y
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import (
    MT5TokenizerFast,
    T5TokenizerFast,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from transformers.optimization import Adafactor
import transformers
from datasets import concatenate_datasets
from typing import Optional, List
import subprocess
import functools
import logging
import numpy as np
from pytz import common_timezones
import torch
import os
from torch import nn
import torch.nn.functional as F

from data.tasks import TASK_MAPPING
import metrics.metrics as mets
from metrics.metrics import TASK_TO_METRICS
from metrics.metrics import build_compute_metrics_fn

###### my imports
from myds import my_interleave_datasets
from conflicts import check_conflicts
from callbacks import WBCallback, AnnealCallback, PTLearningRateCallback
import json
import pandas as pd
import glob
import mylogs 
import itertools, collections
from attempt.myutil import tag_to_image, trim_image
from metrics.metrics import do_score
from encoders.encoders import *
from optim import *
from PIL import Image
from deepdiff import DeepDiff

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1' #TODO remove it or declare it as parameter

logger = logging.getLogger(__name__)
global_scores = []
global_y_labels = []
global_x_labels = []

from scipy.stats import entropy


def task_similarity(dif_ij, final_i, final_j, alpha=1.0):
    # Compute the sigmoid transformation of the difference in scores
    sigmoid_dif = 1 / (1 + torch.exp(-alpha * dif_ij))

    # Compute the final score factor (geometric mean)
    final_factor = (final_i * final_j) / (final_i + final_j)

    # Combine the factors to calculate the similarity
    similarity = sigmoid_dif * final_factor

    return similarity

def cosine_similarity(A, B, N):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    portion_A = torch.tensor(A[:N]).to(device).detach().clone()
    portion_B = torch.tensor(B[:N]).to(device).detach().clone()

    # Normalize vectors a and b
    normalize_a = torch.nn.functional.normalize(portion_A, dim=0)
    normalize_b = torch.nn.functional.normalize(portion_B, dim=0)

    # Compute the cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_sim = F.cosine_similarity(normalize_a, normalize_b, dim=0)

    return cos_sim.item()


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def earth_movers_distance(p, q):
    return np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))

def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    numerator = torch.sum((x - mean_x) * (y - mean_y))
    denominator_x = torch.sqrt(torch.sum((x - mean_x)**2))
    denominator_y = torch.sqrt(torch.sum((y - mean_y)**2))
    pearson_correlation = numerator / (denominator_x * denominator_y)
    return pearson_correlation

def mbp(bp="all",*arg):
    print("info:",*arg)
    mylogs.bp(bp)

def run_command(command):
    output = subprocess.getoutput(command)
    return output

import click
import debugpy
import os.path as op
def map_param(param_map, x, key=False):
    k, v = x, ""
    if "=" in x:
        k, v = x.split("=")
    k = k.strip("--")
    pre = ""
    if k.startswith("@"):
        k = k.strip("@")
    pre += "@"
    if k.startswith("^"):
        k = k.strip("^")
        pre += "^"
    m = param_map[k] if k in param_map else k
    if key is True or not v: 
        return m
    else:
        return pre + m + "=" + v 

@click.group()
def cli():
    pass
@cli.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))

@click.argument('cfgpat')
@click.option(
    "--experiment",
    "-exp",
    default="exp",
    type=str,
    help="Experiment name"
)
@click.option(
    "--exp_conf",
    "-cfg",
    default="",
    type=str,
    help="A file containing configs"
)
@click.option(
    "--break_point",
    "-bp",
    default="",
    type=str,
    help="Stop on breakpoints equal to the value"
)
@click.option(
    "--preview",
    "-pv",
    type=str,
    help="Show only experiment configuraton or some data"
)
@click.option(
    "--exp_vars",
    "-ev",
    type=str,
    default="",
    help="The name of experiment multi-valued variables for which you want to check the difference of their values, if not given it runs all combinations"
)
@click.option(
    "--main_vars",
    "-mv",
    type=str,
    default="",
    help="The name of one multi-valued variable for which you want to check the difference of their values, if not given it runs all combinations e.g. var1@var2@var3"
)
@click.option(
    "--log_var",
    "-lv",
    type=str,
    default="",
    help="The name of an experiment multi-valued variables for which you want to log some data in a logfile names varied with the different values of the varibale"
)
@click.option(
    "--last_var",
    "-last",
    type=str,
    default="",
    help="The name of multi-valued variable you want to be the most nesting loop in combination of expeiments."
)
@click.option(
    "--debug",
    "-d",
    default="",
    type=str,
    help="Enable debugpy, you can specify a breakpoint too"
)
@click.option(
    "--trial",
    "-t",
    default="1",
    type=str,
    help="You can set it for repeating experiments with different identities"
)
@click.option(
    "--version",
    "-v",
    default="1",
    type=str,
    help="You can set it for continueing experiments with different versions (after some changes)"
)
@click.option(
    "--skip",
    "-skip",
    is_flag=True,
    help="Skip existing experiments"
)
@click.option(
    "--save_conf",
    "-conf",
    default="",
    type=str,
    help="Save config for using later"
)
@click.option(
    "--rem",
    "-rem",
    is_flag=True,
    help="Remove the existing experiment folder"
)
@click.option(
    "--label",
    "-l",
    default="",
    type=str,
    help="label for experiment"
)
@click.option(
    "--repeat",
    "-rep",
    is_flag=True,
    help="Repeat an experiment even if the folder already exists",
)
@click.option(
    "--deep_check",
    "-dc",
    is_flag=True,
    help="Check complete json confiturations for checking existing exps"
)
@click.option(
    "--merge",
    "-merge",
    default="",
    type=str,
    help="Merge experiments in one folder"
)
@click.option(
    "--not_copy_prev_exp",
    "-nc",
    is_flag=True,
    help="Don't copy the experiment of the source config to new experiment"
)
@click.option(
    "--reval",
    "-reval",
    is_flag=True,
    help="Evaluation without training"
)
@click.option(
    "--test",
    "-test",
    is_flag=True,
    help="Evaluation without training"
)
@click.option(
    "--use_wandb",
    "-uw",
    is_flag=True,
    help="Evaluation without training"
)
@click.option(
    "--download_model",
    "-mod",
    is_flag=True,
    help="Whether download pretrained model or load it from a directory"
)
@click.option(
    "--max_exp",
    "-max",
    default=0,
    type=int,
    help="Max number of experiments to do (0 means all)"
)
@click.option(
    "--new_exp_folder",
    "-to",
    "-new",
    default="",
    type=str,
    help="The name of a new directory for experiment when loading an existing config file"
)
@click.option(
    "--copy_to",
    "-copy",
    default="",
    type=str,
    help="The name of directory to copy done experiments."
)
@click.option(
    "--inp_log_path",
    "-lp",
    default="",
    type=str,
    help="The directory to save all experiments"
)
@click.pass_context
def run(ctx, cfgpat, experiment, exp_conf, break_point, preview, exp_vars, 
        log_var, last_var, main_vars, 
        debug, version, trial, skip, save_conf, rem, repeat, 
        label, deep_check, merge, not_copy_prev_exp, 
        reval, test, use_wandb, download_model, max_exp, 
        new_exp_folder, copy_to, inp_log_path):
   if debug:
       port = "1234"
       if not break_point: break_point = debug
       debugpy.listen(('0.0.0.0', int(port)))
       print("Waiting for client at run...port:", port)
       debugpy.wait_for_client()  # blocks execution until client is attached
   if break_point:
       mylogs.setbp(break_point)
   exclude_list = []
   exp_args = {}
   save_path = ""
   prev_exp_folder = ""
   prev_save_path = ""
   log_path = inp_log_path
   if not log_path:
       log_path = mylogs.logPath 
   if not log_path.startswith("/"):
       log_path = os.path.join(mylogs.logPath, log_path)
   if exp_conf or cfgpat:
        print("Experiment pattern:", cfgpat)
        cur_path = os.getcwd()
        print("Cur path:", cur_path)
        confs = glob.glob(f"*{cfgpat}*")
        print("Experiment matched confs:", confs)
        if not exp_conf and confs:
            exp_conf = confs[0]
        print("Experiment config:", exp_conf)
        with open(exp_conf) as f:
            exp_args = json.load(f)
        prev_exp_folder = exp_args["output_dir"]
        prev_save_path = exp_args.get("save_path","")
        not_copy_prev_exp = not_copy_prev_exp or exp_args.get("not_copy_prev_exp", False)
        exp_conf_name = Path(exp_conf).stem
        exp_args["conf"] = exp_conf_name
        exp_args["trial"] = str(trial) + "-ret-" + str(exp_args["expid"]).split("-")[-1]
        if experiment == "exp":
            experiment = exp_args["experiment"] + "_" + mylogs.now 
        if test:
            exp_args["do_train"] = False
            exp_args["do_test"] = True 
        if reval:
            exp_args["load_model_dir"] = prev_exp_folder 
            exp_args["do_train"] = False
            exp_args["do_test"] = True 
            exp_args["reval"] = True
            exp_args["trial"] = str(trial) + "-rev-" + str(exp_args["expid"].split("-")[-1])

   mylogs.bp("start")
   experiment = experiment.replace("#","-").replace("@","-").replace(":","-")
   if exp_conf and "experiment" in exp_args:
       cc = 1
       exp_name = experiment
       while exp_name == exp_args["experiment"]:
           exp_name = experiment + "-" + str(cc)
           cc += 1
       experiment = exp_name

   #if exp_conf and not new_exp_folder: 
   #   log_folder = experiment 
      #ans = input("Do you want save the results in (otherwise enter new folder) "+log_folder+ "[yes]:")
      #if ans and ans != "yes":
      #    new_exp_folder = ans
      #else:
   #   new_exp_folder = log_folder 

   mylogs.bp("start") 
   if experiment == "self":
       save_path = os.path.join(os.getcwd(), "output")
   if prev_exp_folder and not new_exp_folder:
       save_path = prev_save_path
   elif not reval or new_exp_folder:
       if new_exp_folder and save_path:
          relative_path = os.path.relpath(save_path, log_path)
          parts = relative_path.split(os.path.sep)
          parts[0] = new_exp_folder 
          new_path =  os.path.sep.join(parts)
          save_path = os.path.join(mylogs.resPath, new_path) 
          # save_path = os.path.join(str(Path(save_path).parent), experiment)
       elif new_exp_folder:
          save_path = os.path.join(log_path, new_exp_folder)
       else:
          save_path = os.path.join(log_path, experiment)
       if Path(save_path).exists():
          #if not rem:
          #     while Path(save_path).exists():
          #        ans = "u" #input("Do you want to delete '" + save_path + \
          #                  #"'? d)delete u)use  newname)")
          #        if ans == "d": 
          #            rem = True
          #        elif ans == "u":
          #            break
          #        else:
          #            experiment = ans
          #            save_path = os.path.join(log_path, experiment)
          if False: #rem:
               main_folder = save_path
               ans = "yes" #input("Do you want to remove " + main_folder + ":")
               if ans == "yes":
                   main_folder = main_folder.rstrip("/")
                   dirs = glob.glob(main_folder + '/*/')
                   for d in dirs:
                        shutil.rmtree(d)

       if Path(save_path).is_file():
           os.remove(save_path)

   if not save_path:
       save_path = prev_save_path if prev_save_path else os.getcwd()
   Path(save_path).mkdir(exist_ok=True, parents=True)
   if copy_to:
      copy_to = os.path.join(log_path, copy_to)
      Path(copy_to).mkdir(exist_ok=True, parents=True)

   args = {}
   args["conf"] = exp_conf
   args["save_path"] = save_path

   args["new_exp_folder"] = new_exp_folder
   args["not_copy_prev_exp"] = not_copy_prev_exp
   args["load_path"] = "" 
   args["label"] = label
   args["is_debug"] = debug
   if not reval:
      args["trial"] = trial
   if not download_model:
      args["load_path"] = mylogs.pretPath 
   if not experiment.startswith("%"):
       experiment = "%" + experiment # % forces to reserve the value as it is  
   args["experiment"] = experiment 
   args["version"] = version 
   args["break_point"] = break_point 
   args["preview"] = preview 
   args["repeat"] = repeat 
   args["reval"] = reval 
   args["use_wandb"] = use_wandb 
   tags = exp_args["tag"] if "tag" in exp_args else ["expid"] 
   full_tags = exp_args["full_tag"] if "full_tag" in exp_args else ["expid"] 

   mylogs.bp("start")
   _dir = Path(__file__).parent
   param_map = {}
   param_file = os.path.join(_dir, "params.json")
   if Path(param_file).is_file():
       with open(param_file) as f:
          param_map = json.load(f)

   all_vars = [map_param(param_map, x) for x in ctx.args]
   # all_vars = [x.strip("--") for x in ctx.args]
   mylogs.bp("vars")
   var_names = [x.split("=")[0] for x in all_vars] 
          # if not (x.split("=")[0].startswith("@comment") 
          #     or x.split("=")[0].startswith("@c-"))]
   values = []
   for x in all_vars:
       _vv = x.split("=")
       if len(_vv) < 2:
           assert False, "invalid argument " + str(x) + "|" + str(_vv)
       if not (_vv[0].startswith("@comment") or _vv[0].startswith("@c-")):
           _vv = _vv[1].strip("#")
           _vvv = _vv.split("#")
       else:
           _vvv = [_vv[1]]
          #  continue
       values.append(_vvv)
   var_dict = {k:n for k,n in zip(var_names, values)} 
   if last_var:
       last_var = map_param(param_map, last_var)
       last_var = "@" + last_var
       var_dict[last_var] = var_dict.pop(last_var)
   _mvars = []
   mylogs.bp("mvar")
   if main_vars and "--" in main_vars:
       main_vars = main_vars.split("--")
   if not main_vars:
       main_vars = [vv.strip("@") for vv in var_names if vv.endswith("@")]
   if not main_vars:
       main_vars = [map_param(param_map,x,key=True) for x in ctx.args if x.startswith("--")]
   for var in main_vars:
       if not var: continue
       var = map_param(param_map, var)
       if "=" in var:
           var_name = var.split("=")[0].strip("@")
           if False: #TODO temporary 
               assert var_name in exp_args, var_name +" must be in experiment variables (config)"
           var_item = var.split("=")[1]
           if not var_name.startswith("comment") or var_name.startswith("c"):
               var_item = var_item.strip("#").split("#")
           var_dict["@" + var_name] = var_item
           _mvars.append(var_name)
       else:
           _mvars.append(var)
   if _mvars: main_vars = _mvars

   
   mylogs.bp("prev")
   if prev_exp_folder and not "prompts_prefix" in main_vars:
       args["prompt_encoders_dir"] = prev_exp_folder
   if prev_exp_folder and not "task_name" in main_vars and not not_copy_prev_exp and not repeat:
       prev_folder = Path(prev_exp_folder)
       prev_exp_id = prev_folder.name
       eval_folders = glob.glob(
               os.path.join(prev_folder.parent, "Eval-" + prev_exp_id + "*no-mask*"))
       try:
           shutil.copytree(prev_exp_folder, 
                   os.path.join(save_path, Path(prev_exp_folder).name))
       except (FileNotFoundError, FileExistsError):
           pass
       for folder in eval_folders:
           try:
               shutil.copytree(folder, os.path.join(save_path, Path(folder).name))
           except (FileNotFoundError, FileExistsError):
               pass


   for key,val in var_dict.items():
       multi = [item for item in val if re.match("multi-(.*)", item)]
       members = [x.strip("@") for x in val if not x in multi and not "@" in x.strip("@")]
       if multi:
           ext = []
           for m in multi:
               _, l = m.split("-")
               l = len(members) if l == "all" else int(l)
               val.remove(m)
               comb = itertools.combinations(members, l)
               ext.extend(["@".join(c) for c in comb])
           val = ext + val
           var_dict[key] = val

   var_names = list(var_dict.keys())
   values = list(var_dict.values())
   inp_exp_vars = exp_vars
   mylogs.bp("start")
   mylogs.bp("mvar")
       # main_vars = "--".join([x.strip("@") for x in main_vars])

   if not exp_vars:
       #if main_vars:
       #    exp_vars = main_vars
       #else:
       exp_vars = [vv.strip("@") for vv in var_names if vv.startswith("@")]
   elif type(exp_vars) != list:
       exp_vars = inp_exp_vars = [exp_vars]
   full_tags.extend([x for x in exp_vars if not "^" in x])
   args["log_var"] = log_var 
   for ii, (vv, cc) in enumerate(zip(var_names, values)):
      if len(cc) > 1:
           if vv.startswith("@") or vv.endswith("@"):
               vv = vv.strip("@")
               tags.append(vv.strip("^"))
           full_tags.append(vv.strip("^"))
           values[ii] = [x for x in cc if not x.startswith("!")] 
           #if (exp_vars and not vv in exp_vars) or (main_vars and not vv in main_vars):
           #    values[ii] = [values[ii][0]] # ignore the rest of values for this item 
      if len(values[ii]) == 1:
           if not vv.startswith("@"):
               exclude_list.append(vv)
           vv = vv.strip("@")
   var_names = [vv.strip("@") for vv in var_names]

   full_tags = list(set(full_tags))
   mylogs.bp("full_tags")
   for pv in inp_exp_vars:
       assert pv in full_tags, f"Eror: {pv} must be 'all' or one of {full_tags} which have multiple values"

   existing_exps = glob.glob(op.join(save_path, "*.json"))
   not_conf = ["break_point","copy","expid", "total_exp", "full_tag", "tag", "preview", "output_dir", "experiment", "use_cache_file", "use_cache", "trial", "exp_number", "num_target_prompts", "prompt_masking", "per_device_train_batch_size","comment"] + [v for v in var_names if v.startswith("comment") or v.startswith("c-")]
   # args["full_tag"] = full_tags 
   tot_comb = [dict(zip(var_names, comb)) for comb in itertools.product(*values)]
   ii = len(existing_exps) if not reval else 0 
   exps_done = 0
   orig_args = args.copy()
   total = len(tot_comb)
   args["total_exp"] = total
   logger.info("Total experiments:%s", total)
   mylogs.bp("comb")
   old_comb = None
   ctags = []
   for comb in tot_comb:
       if old_comb is not None:
           diff_comb = DeepDiff(comb, old_comb) 
           if "values_changed" in diff_comb:
               vc = diff_comb["values_changed"]
               for item in vc:
                   val = item.replace("root['","").replace("']","")
                   if not val in ctags:
                       ctags.append(val)
       old_comb = comb.copy()

   args["tag"] = ctags 
   mylogs.bp("merge")
   args["merge"] = merge
   args["save_conf"] = save_conf 
   y_labels = []
   exp_number = 1
   for comb in tot_comb:
       _output_dir = []
       prev_name = ""
       prev_item = ""
       conflict = "" 
       mvars = {}
       for kk, (var_name,var_item) in enumerate(comb.items()):
           if var_name.startswith("^") and prev_name:
               prev_vals = values[kk-1]
               cur_vals = values[kk]
               assert len(prev_vals) == len(cur_vals), str(prev_vals) + " " + str(cur_vals) + "Pair variables must have same number"
               pairs = zip(prev_vals, cur_vals)
               if not (prev_item, var_item) in pairs:
                   conflict = prev_name + ":" + prev_item + " "+ var_name + ":" + var_item
                   break
           var_name = var_name.strip("^")
           args[var_name]=var_item
           if var_name in main_vars:
               mvars[var_name] = var_item
           if not var_name in exclude_list:
               _output_dir.append(var_name + "_" + str(var_item))
           prev_name = var_name
           prev_item = var_item
       if conflict:
           print(f"Dep var observed {conflict} ignored")
           continue
       ii += 1
       mylogs.bp("expid")
       if max_exp > 0 and exps_done > max_exp:
           print(f"Max number of exp reached {max_exp} ")
           return
       exp_dir = experiment.split("/")[-1] 
       mylogs.bp("merge")
       if merge:
           merge = map_param(param_map, merge, key=True)
       if not "expid" in exp_args or merge: 
           if merge:
               for (nn, vv) in mvars.items():
                   if nn != merge and not nn in not_conf:
                       exp_dir += "_" + nn + "-" + str(vv)
               # exp_dir = str(hash(exp_dir))
               h =  str(str2int(exp_dir)) 
               hash_dir = h[:3] + str(len(exp_dir)) + h[-2:]
               args["expid"] = hash_dir
           else:
               args["expid"] = ii 
       elif "-" in str(exp_args["expid"]):
           expid = str(exp_args["expid"]).replace("-rep","")
           expid = expid.strip("-")
           args["expid"] = expid.split("-")[-1] + "." + str(ii)
       else:
           args["expid"] = ii 

       args["main_vars"] = mvars
       args["cat"] = experiment.split("/")[-1] 
       args = {**exp_args, **args}
       #_output_dir.append(str(args["expid"]))
       output_dir = save_path 
       #if exp_conf:
       #    output_dir = exp_args["output_dir"]
       if merge:
           ee = args["expid"]
           exp_file = args[merge]
           _output_dir = label + "-" + str(ee)
           _output_dir = _output_dir.strip("-")
           output_dir = os.path.join(save_path, _output_dir)
           if glob.glob(op.join(output_dir, f"*{exp_file}{trial}*.tsv")): 
               if skip is True:
                   print("The experiment already exists, skipping!!")
                   print(exp_dir)
                   print(output_dir)
                   if copy_to:
                      print("Copying to ", copy_to)
                      shutil.copytree(output_dir, 
                           os.path.join(copy_to, Path(output_dir).name))
                   print("-----------------------------------------")
                   continue
               print("Merging to ", output_dir)
       else:
           ee = round(float(args["expid"]))
           eee = ee
           _output_dir = label + "-" + str(ee)
           _output_dir = _output_dir.strip("-")
           output_dir = os.path.join(save_path, _output_dir)
           #if Path(output_dir).exists() and not repeat:
           #    mylogs.minfo(f"The folder {output_dir} already exists....")
           #    ans = input("Do you want to skip the experiment?")
           #    if True: #ans == "y":
           #        continue
           if not reval:
               while Path(output_dir).exists():
                   ee += 1 
                   _output_dir = label + str(ee)
                   output_dir = os.path.join(save_path, _output_dir)
           if label:
               expid = experiment.split("/")[-1] + "-" + label + "-" + str(eee)
               expid = expid.strip("-")
               args["expid"] = expid
           else:
               expid = experiment.split("/")[-1] + "-" + str(eee)
               expid = expid.strip("-")
               args["expid"] = expid
       if repeat:
          args["expid"] += "-rep"
       args["output_dir"] = "%" + output_dir 
       _conf = json.dumps(args, indent=2)
       if preview == "conf":
           print(f"================ {ii}/{total} =====================")
           print(_conf)
           out_conf_file = os.path.join(save_path, "logs", "exp_" + str(ii) + ".json")
           Path(os.path.join(save_path, "logs")).mkdir(exist_ok = True, parents=True)
           with open(out_conf_file,"w") as f:
               print(_conf, file=f)
           continue
       # break point before running to check arguments (breakpoint must be check)
       mylogs.bp("check")
       tags_dict = mylogs.get_tag(tags, args)
       full_tags_dict = mylogs.get_tag(full_tags, args)
       #title = "@".join(list(tags_dict.values()))
       title =  mylogs.get_tag(tags, args, as_str=True)
       exp_exists = False
       conf_fname = os.path.join(save_path,"conf_"+str(args["expid"])+".json")
       if not exp_conf and existing_exps:
           for ee in existing_exps:
               if preview == "ex-why":
                   print("Checking existaince for ", ee)
               with open(ee) as f:
                   jj = json.load(f)
                   if reval and ee == conf_fname: 
                       args = jj.copy()
                       args["do_train"] = False
                       args["do_test"] = True 
                       args["trial"] = str(jj["trial"]) + "-re"
                       args["reval"] = True
                       break

                   if "output_dir" in jj:
                       output_dir = jj["output_dir"].strip("%")
                       if glob.glob(op.join(output_dir, "*.tsv")):
                           trial = int(jj["trial"]) + 1 if "trial" in jj else 2
                           exp_exists = True
                   are_equal = True
                   for k,v in args.items():
                       if not k in not_conf: 
                           if not k in jj or strval(v) != strval(jj[k]):
                               are_equal =False
                               if preview == "ex-why":
                                   print("It's not equal to because ", k, " is ",v, " against ", strval(jj[k]))
                               break
               if are_equal:
                  print(ii, " is equal to ", ee)
               if deep_check:
                  exp_exists = exp_exists and are_equal
                  break
       if preview == "tag":
           print(f"=#============== {ii}/{total} =====================")
           conf_str = json.dumps(full_tags_dict, indent=2)
           print(conf_str)
           if exp_exists:
               print("=============== DONE ===========")
           with open("logs/exp_" + str(ii) + ".tag","w") as f:
               print(conf_str, file=f)
           continue
       if exp_exists and not reval:
           args["output_dir"] = "%" + output_dir 
           print("Skipping experiment ", ii, ": The experiment already exists!")
           if not preview and not repeat:
              continue 
       # preview existing experiments 
       if preview in ["ex","ex-why","exists","run"]: #
           #print(f"========== Experiment {ii} pf {total},  Input Vars: ===============")
           #all_var_str = json.dumps(var_dict, indent=2)
           #print(all_var_str)
           print(f"========== Experiment {ii} pf {total},  Main Vars: ===============")
           main_var_str = json.dumps(mvars, indent=2)
           print(main_var_str)
           print("==================================================================")
           if preview != "run":
               ans = input("Continue preview? [yes]:")
               if not ans or ans == "yes":
                   continue
               else:
                   print("Stop!")
                   return
       done = "na"
       args["exp_number"] = exp_number
       exp_number += 1
       if debug:
           ctx.invoke(train, **args)
       else:
           try:
               if preview == "run":
                   ans = input("Run this? [yes/stop/next] [yes]:")
                   if not ans or ans == "yes":
                       done = ctx.invoke(train, **args)
                   elif ans == "stop":
                       print("Stop!")
                       return
                   else:
                       continue
               else:
                   done = ctx.invoke(train, **args)
               y_labels.append(args["expid"])
               if done != "has_conflict" and done != "is_repeated":
                   with open(conf_fname, "w") as f:
                       print(_conf, file=f)
                   exps_done += 1
               elif preview == "lict":
                   c = input("check for conflicts!")
           except Exception as e:
               print(f"================ {ii}/{total} =====================")
               _conf = json.dumps(args, indent=2)
               print(_conf)
               raise Exception("An error occured in the experiment")
       if preview == "one" or (preview == "data" and done == "data_preview"):
           print("return due preview:", preview, " done:",  done)
           return

   if False: #global_scores:
        score = torch.cat(global_scores, dim=0)
        img_buf = WBCallback.save_image(score=score, 
           y_labels=global_y_labels,
           x_labels=global_x_labels,
           title = "cat-" + args["cat"], 
           df=None) 
        if img_buf:
            cur_img = Image.open(img_buf)
            cat = Path(save_path).parent
            sp = op.join(cat, "images") 
            Path(sp).mkdir(exist_ok=True, parents=True)
            pic = "router_global"
            pp = sp + "/pred_" + pic + ".png"
            if Path(pp).is_file():
                _image = Image.open(pp)
                cur_img = combine_y([cur_img, _image])
            cur_img.save(pp)

# m3
@cli.command()
def train(**kwargs):
    seed = kwargs.get("seed", 123)
    set_seed(seed)
    global global_x_labels
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    config_name = kwargs.setdefault("config","base")
    use_wandb = kwargs.get("use_wandb", False)
    if use_wandb:
        import wandb
    home = mylogs.home
    config_file = ""
    if config_name == "base":
        config_file =f"base.json"
    elif config_name == "attempt":
        config_file= f"single_task.json"

    _dir = Path(__file__).parent
    param_map = {}
    param_file = os.path.join(_dir, "params.json")
    if Path(param_file).is_file():
       with open(param_file) as f:
          param_map = json.load(f)

    exp_conf = json.dumps(kwargs, indent=2)
    mylogs.clog.info(exp_conf)
    preview = kwargs.setdefault("preview","")
    repeat = kwargs.setdefault("repeat",False)
    reval = kwargs.setdefault("reval",False)
    log_var = kwargs.setdefault("log_var","")
    main_vars = kwargs.setdefault("main_vars",{})
    mylogs.set_args(kwargs.copy())
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if config_file and config_file.endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        config_file = op.join(_dir,"confs", config_file)
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=config_file)
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    #### My code: overwrite kwargs over arguments read from parser
    def overwrite_conf(kwargs):
        new_kwargs = {}
        for k,v in kwargs.items():
            logger.info("ARGS: %s=%s", k, v)
            v = strval(v)
            new_kwargs[k] = v
            if hasattr(model_args,k):
                setattr(model_args, k, v)
            elif hasattr(data_args,k):
                setattr(data_args, k, v)
            elif hasattr(training_args,k):
                setattr(training_args, k, v)
            elif hasattr(adapter_args,k):
                setattr(adapter_args, k, v)
        return new_kwargs

    # sssssssss
    torch.autograd.set_detect_anomaly(True)
    if use_wandb:
        training_args.report_to = kwargs.get("report_to", "wandb")

    new_exp_folder = kwargs.get("new_exp_folder","")
    prompts_conf = kwargs.get("prompts_conf", None)
    if prompts_conf in ["SLP","SL"]:
        kwargs["num_train_epochs"] = int(kwargs["num_train_epochs"]) + 10
    if prompts_conf in ["SIL","SL"] and kwargs["compose_method"] == "wsp1":
        kwargs["compose_method"] = "wavg" 

    kwargs = overwrite_conf(kwargs)

    def parse_prompts_conf(label):
        flags = {
            'add_target': kwargs.get("add_target", False),
            'use_source_prompts': False,
            'load_source_prompts': False,
            'learn_source_prompts': False,
            'use_private_prompts': False,
            'load_private_prompts': False
        }
        if 'A' in label:
            flags['add_target'] = True
        if 'S' in label:
            flags['use_source_prompts'] = True
            if 'SI' in label:
                flags['load_source_prompts'] = True
            if 'L' in label[label.index('S') + 1:]:
                flags['learn_source_prompts'] = True
        if 'P' in label:
            flags['use_private_prompts'] = True
            if 'PI' in label:
                flags['load_private_prompts'] = True

        return flags

    mylogs.bp("nsp")
    if prompts_conf:
        pflags = parse_prompts_conf(prompts_conf)
        kwargs = {**kwargs, **pflags}

    kwargs = dotdict(kwargs)
    _dict = kwargs.copy()
    for c in ["tag","log_var","main_vars","full_tag","new_exp_folder", 
            "total_exp", "exclude_tasks", "include_tasks", "break_point", "repeat"]:
        if c in _dict:
            del _dict[c]

    exp_conf = json.dumps(_dict, indent=2)
    print("============ CONF ===========")
    print(exp_conf)
    Path(training_args.output_dir).mkdir(exist_ok=True, parents=True)
    merge = kwargs.get("merge", False)
    if not reval or new_exp_folder:
        expid = kwargs.get("expid", 1)
        expid = str(expid)
        exp_conf_name = "exp.json" if not merge else "exp_" + expid + ".json"
        with open(op.join(training_args.output_dir, exp_conf_name), "w") as f:
            print(exp_conf, file=f)
    save_conf = kwargs.get("save_conf","")
    if save_conf:
        with open(op.join(mylogs.confPath, "conf_" + save_conf + ".json"), "w") as f:
            print(exp_conf, file=f)
    mylogs.bp("conf")

    #  dataset configs
    ds_confs = kwargs.setdefault("ds_config", ["conf"])
    if type(ds_confs) != list:
        ds_confs = [ds_confs]
    if type(data_args.task_name) != list:
        data_args.task_name = [data_args.task_name]

    ds_combs = itertools.product(data_args.task_name, ds_confs)
    _tasks = []
    _confs = []
    for comb in ds_combs:
        _tasks.append(comb[0])
        _confs.append(comb[1])

    data_args.task_name = _tasks
    data_args.dataset_config_name = _confs
    data_args.eval_dataset_config_name = _confs
    data_args.test_dataset_config_name = _confs

    trainer_shuffle = kwargs.setdefault("trainer_shuffle", False)
    bp = kwargs.setdefault("break_point","")
    # set other options
    if type(data_args.task_name) != list:
        data_args.task_name = [data_args.task_name]

    exclude_tasks = kwargs.setdefault("exclude_tasks", []) 
    if exclude_tasks is None:
        exclude_tasks = []
    if type(exclude_tasks) != list:
        exclude_tasks = [exclude_tasks]
    include_tasks = kwargs.setdefault("include_tasks", []) 
    if include_tasks is None:
        include_tasks = []
    if type(include_tasks) != list:
        include_tasks = [include_tasks]

    include_exclude_tasks = kwargs.setdefault("include_tasks_exclude_from_test", []) 
    if include_exclude_tasks is None:
        include_exclude_tasks = []
    if type(include_exclude_tasks) != list:
        include_exclude_tasks = [include_exclude_tasks]

    exclude_from_test_tasks = kwargs.setdefault("exclude_from_test_tasks", []) 
    for t in include_exclude_tasks:
        if not t in include_tasks:
            include_tasks.append(t)
        if not t in exclude_from_test_tasks:
            exclude_from_test_tasks.append(t)

    if exclude_from_test_tasks is None:
        exclude_from_test_tasks = []
    if type(exclude_from_test_tasks) != list:
        exclude_from_test_tasks = [exclude_from_test_tasks]

    if exclude_tasks or include_tasks:
        tasks = []
        for t in data_args.task_name + include_tasks:
            if not t in exclude_tasks and not t in tasks:
                tasks.append(t)
        data_args.task_name = tasks

    
    num_prompts = kwargs.setdefault("num_prompts", 1) 
    target_prompt_length = adapter_args.num_prompt_tokens
    source_prompt_length = adapter_args.num_prompt_tokens
    use_source_prompts = kwargs.setdefault("use_source_prompts", True)
    load_source_prompts = kwargs.setdefault("load_source_prompts", False) 
    learn_source_prompts = kwargs.setdefault("learn_source_prompts", False) 
    use_private_prompts = kwargs.setdefault("use_private_prompts", False)
    load_private_prompts = kwargs.setdefault("load_private_prompts", False)
    add_target_prompt = kwargs.setdefault("add_target", False)
    use_source_set = kwargs.setdefault("use_source_set", False)

    #if not use_source_prompts:
    #    model_args.compose_method = "pt"

    tasks = data_args.task_name
    train_prefix = {}
    test_prefix = {}
    task_names = []
    mylogs.bp("test_prefix")
    for task_name in tasks:
        tname = train_px = test_px = task_name #prefix for test and train sets
        test_prefix_list = task_name.split("--")
        if len(test_prefix_list) > 1:
            tname = test_prefix_list[0] 
        task_names.append(tname)
        train_prefix[tname] = [tname]
        test_prefix[tname] = test_prefix_list 

    cross_prefix = kwargs.get("cross_prefix", None)
    if cross_prefix:
        ctasks = cross_prefix
        if cross_prefix == "all": 
            ctasks = task_names
        if type(ctasks) != list:
            ctasks = [cross_prefix]
        for task in ctasks:
            for other in task_names:
                if other != task:
                    if not task in test_prefix:
                        test_prefix[task] = []
                    if not other in test_prefix[task]:
                        test_prefix[task].append(other)

    data_args.task_name = task_names
    data_args.eval_dataset_name=data_args.task_name
    data_args.test_dataset_name=data_args.task_name

    task_source_prompts_set ={}
    tasks = data_args.task_name
    for task_name in tasks:
        tid = task_name
        if not tid in task_source_prompts_set:
           task_source_prompts_set[tid] = []
        rel_sh = REL_TO_SHARED_TOKENS[task_name] if task_name in REL_TO_SHARED_TOKENS else task_name
        task_source_prompts_set[tid].extend(rel_sh.split())

    add_or_attend_input = model_args.attend_input or kwargs.get("add_input", False)
    nsp = 0
    inp_nsp = kwargs.setdefault("num_source_prompts", nsp) 
    source_per_task = kwargs.setdefault("source_per_task", False) 
    if source_per_task: # and inp_nsp == 0:
        nsp = len(tasks)
        data_args.source_prompts = tasks # source are the same target tasks
    elif use_source_set:
        nsp = max([len(s) for s in task_source_prompts_set.values()])
    elif data_args.source_prompts is not None:
        if type(data_args.source_prompts) != list:
            data_args.source_prompts = [data_args.source_prompts]
        if len(data_args.source_prompts) > 0:
            nsp = len(data_args.source_prompts) 

    nsp += inp_nsp 
    num_source_prompts = nsp 
    num_target_prompts = 1
    if model_args.attn_tuning is True:
        num_target_prompts = kwargs.setdefault("num_target_prompts",num_source_prompts) 
        if num_target_prompts == "auto":
            num_target_prompts = (num_source_prompts // 2) + 1
        ntp = num_target_prompts
        if ntp == 0: 
            num_target_prompts = num_source_prompts
        if num_source_prompts > 0:
            num_target_prompts = min(num_target_prompts, num_source_prompts)
        else:
            num_target_prompts = 1
        if model_args.attend_target and ntp == 0:
            num_target_prompts += 1
        if add_or_attend_input and ntp == 0:
            num_target_prompts += 1
        if use_private_prompts and ntp == 0:
            num_target_prompts += 1
        num_target_prompts = max(num_target_prompts, 1)
        if model_args.compose_method in ["cat", "concat"]: #, "pool", "mpool","lin"]:
            target_prompt_length = num_target_prompts * adapter_args.num_prompt_tokens
        #    if add_target_prompt:
        #        target_prompt_length += adapter_args.num_prompt_tokens
        elif model_args.compose_method in ["catw","mcat","scat","mscat"]:
            target_prompt_length = num_target_prompts * adapter_args.num_prompt_tokens
        elif model_args.compose_method in ["wcat", "wcp", "wcp1"]:
            target_prompt_length = 2 * adapter_args.num_prompt_tokens
        #    if add_target_prompt:
        #        target_prompt_length += adapter_args.num_prompt_tokens
        elif model_args.compose_method == "tcat":
            target_prompt_length = 2 * adapter_args.num_prompt_tokens
        elif model_args.compose_method == "wavg":
            pass
            #target_prompt_length = num_target_prompts * adapter_args.num_prompt_tokens
            #adapter_args.num_prompt_tokens = target_prompt_length

        kwargs["num_target_prompts"] = num_target_prompts
        mylogs.main_args["num_target_prompts"] = num_target_prompts

    kwargs["num_prompt_tokens"] = target_prompt_length 
    kwargs["source_prompt_length"] = source_prompt_length 
    kwargs["target_prompt_length"] = target_prompt_length 
    
    #TODO to make it compatible with older config files
    if data_args.data_path == "atomic2020":
        data_args.data_path = "datasets"

    task_args = {}
    task_args["data_seed"] = data_args.d_seed
    task_args["map_labels"] = kwargs.setdefault("map_labels", True)
    task_args["samples_per_head"] = kwargs.setdefault("samples_per_head", 3)
    task_args["start_row"] = kwargs.setdefault("start_row", 0)
    task_args["mapping"] = kwargs.setdefault("mapping", "map")
    task_args["use_cache_file"] = kwargs.setdefault("use_cache_file", True)
    task_args["use_config"] = kwargs.setdefault("use_config", True)
    task_args["equal_labels"] = kwargs.setdefault("equal_labels", False)
    task_args["map_style"] = kwargs.setdefault("map_style", "map")
    task_args["multi_choice"] = kwargs.setdefault("multi_choice", False)
    task_args["train_samples"] = data_args.max_train_samples
    task_args["val_samples"] = data_args.max_val_samples
    task_args["test_samples"] = data_args.max_test_samples
    task_args["omit_part"] = kwargs.get("omit_part","")
    task_args["qpos"] = kwargs.get("qpos","end") # position of question
    task_args["chpos"] = kwargs.get("chpos","start") # position of question
    task_args["len_thresh"] = kwargs.get("len_thresh", None) # position of question
    task_args["num_prompts"] = num_prompts 
    task_args["target_prompt_length"] = target_prompt_length 
    task_args["prompt_length"] = kwargs.setdefault("prompt_length", 
                                    adapter_args.num_prompt_tokens)
    task_args["fixed_length_prompt"] = adapter_args.fixed_length_prompt
    input_template = data_args.template
    if adapter_args.prompt_tuning and not "ptar" in input_template:
        if input_template in ["unsup-nat", "sup-nat"]:
            ptemp = "ptar"
        else:
            ptemp = "0-ptar"
        prompt_template = kwargs.get("prompt_template", ptemp)
        input_template = prompt_template + "-" + input_template 
        kwargs["template"] = input_template
    task_args["template"] = input_template 
    task_args["add_prefix"] = data_args.add_prefix
    task_args["data_path"] = data_args.data_path
    task_args["rels"] = data_args.task_name if kwargs.rels == "tasks" else kwargs.rels
    task_args["task_comb"] = kwargs.task_comb
    task_args["id"] = kwargs["expid"]

    # an option to explicitly specify the method of training 
    # (pt: prompt-tuning, ft:fine-tuning, px:prefix-tuning etc.)
    method = kwargs.setdefault("method", "")
    if kwargs.setdefault("adjust_epochs", True) and data_args.max_train_samples <= 10:
        num_epochs = training_args.num_train_epochs
        num_epochs *= 2
        training_args.num_train_epochs = num_epochs
    

    #if type(data_args.task_name) == list:
    #    model_args.multi_task = True

    # tags are variables that are varied among experiments. 
    tag = kwargs.setdefault("tag",[]) # the selected tags
    full_tag = kwargs.setdefault("full_tag",[]) # the full list of tags
    # check conflicts of options
    check_cfls = kwargs.setdefault("check_conflicts",True)
    if check_cfls: #check conflicts
        resolved, msg = check_conflicts(model_args, data_args, 
                training_args, adapter_args, kwargs)
        print(msg)
        title = mylogs.get_tag(full_tag)
        title = json.dumps(title, indent=4)
        mylogs.dlog.info(title)
        mylogs.dlog.info("%s", msg)
        mylogs.dlog.info("-------------------------------------")
        if not resolved:
            shutil.rmtree(training_args.output_dir)
            return "has_conflict"

    if False: #main_vars: #TODO it must be checked in run not here
        x = main_vars
        y = mylogs.prev_main_vars
        repeated_items = {k: x[k] for k in x if k in y and x[k] in y[k]}
        if len(repeated_items) == len(main_vars):
            shutil.rmtree(training_args.output_dir)
            print("return, is repeated!")
            return "is_repeated"
        for k,v in main_vars.items():
            if not k in mylogs.prev_main_vars:
                mylogs.prev_main_vars[k] = []
            mylogs.prev_main_vars[k].append(v)
        if preview == "mavar":
            print("preview is mvar")
            return 

    if log_var:
       mylogs.plog.handlers.clear()
       mylogs.add_handler(mylogs.plog, log_var + "_" + str(kwargs[log_var]), 
               base_folder=kwargs.save_path)
       mylogs.plog.info(exp_conf)
    ###### Collect experiment infos
    exp_info = {}
    exp_info["attn_learning_rate"] = model_args.attn_learning_rate
    multi_tasking = False
    if len(data_args.task_name) > 1:
        exp_info["multi_single"] = "multi"
        multi_tasking = True
    else:
        exp_info["multi_single"] = "single"

    wandb_dir = kwargs.save_path #op.join("logs", experiment)
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    experiment = kwargs.experiment
    tags_dict = mylogs.get_tag(tag, kwargs)
    if use_wandb:
        import wandb
        wandb.init(
          # Set the project where this run will be logged
          project= experiment.replace("#","-").replace(":","-").replace("/","-")[:100], 
          name=title,
          dir=wandb_dir,
          settings=wandb.Settings(symlink=False),
          # Track hyperparameters and run metadata
          config=tags_dict
        )
        if wandb.run is not None:
          exp_info["runid"] = wandb.run.id

    _tag = mylogs.get_tag(tag)  
    exp_info["tag"] = list(_tag.values())
    exp_info["taginfo"] = list(_tag.keys())
    _ftag = mylogs.get_tag(full_tag)  
    exp_info["ftag"] = _ftag 
    ######
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if training_args.resume_from_checkpoint is None or (last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0):
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            #existing_results = glob.glob(op.join(training_args.output_dir, "*.tsv"))
            #if existing_results and not preview and not repeat:
            #    print("Skipping experiment:", training_args.output_dir)
            #    return "skipped" 
            #last_checkpoint = None
            #out = training_args.output_dir
            #out += "_" + mylogs.now
            #Path(out).mkdir(parents = True, exist_ok=True)
            #training_args.output_dir = out
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        #handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_name_or_path =  model_args.config_name if model_args.config_name else model_args.model_name_or_path
    load_path = kwargs.setdefault("load_path", "")
    if not model_name_or_path.startswith("/") and load_path:
        model_name_or_path = op.join(load_path, model_name_or_path)
    if "mt5" in model_name_or_path:
        tokenizer = MT5TokenizerFast.from_pretrained(model_name_or_path)
    elif "pars" in model_name_or_path:
        tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Set seed before initializing model.
    tasks = data_args.task_name
    mylogs.bp("steps")
    total_samples = 0
    warmup_steps = 0
    total_steps = 1
    steps = 0
    if training_args.do_train:
        for ti, (task_name, config_name) in enumerate(zip(tasks, data_args.dataset_config_name), start=1):
             t_args = dotdict(task_args.copy())
             task = AutoTask.get(task_name, config_name, task_args=t_args, tokenizer= None)
             assert task.get_records_num("train", data_args.max_train_samples) != 0, "The number of records are zero"
             total_samples += task.get_records_num("train", data_args.max_train_samples)

        training_args.per_device_train_batch_size = min(total_samples, training_args.per_device_train_batch_size)

        steps = total_samples * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
        if training_args.warmup_steps is not None:
            warmup_steps = training_args.warmup_steps
        else:
            warmup_steps = 0.2 * steps
        total_steps = steps + warmup_steps + 5
    
    mylogs.bp("steps")
    anneal_steps = 0.6*total_steps
    ftemp = kwargs.get("fixed_temperature", -1)
    if ftemp > 0 and "temperature" not in main_vars:
        model_args.temperature = ftemp
        model_args.anneal_min = ftemp
        kwargs["anneal_min"] = ftemp 
    elif kwargs.get("adjust_temperature", True) and "temperature" not in main_vars:
        if data_args.max_train_samples < 10:
            model_args.temperature = 5
        elif data_args.max_train_samples < 20:
            model_args.temperature = 3
        else:
            model_args.temperature = 1
        #elif model_args.compose_method in ["mwavg","mcat"]:
        #    model_args.temperature = 0.001
        #else:
        #    model_args.temperature = 2
        kwargs["temperature"] = model_args.temperature
    if model_args.anneal_rate is None: 
        anneal_rate = (model_args.temperature - model_args.anneal_min)/(anneal_steps) 
    else:
        anneal_rate = model_args.anneal_rate
    # Load a model config
    config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=1,  # Define number of soft prompt tokens
        tokenizer_name_or_path=model_name_or_path
    )

    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    config.dropout_rate = kwargs.get("dropout", 0.1)
    config.prompt_tuning = adapter_args.prompt_tuning #my option
    config.attn_tuning = model_args.attn_tuning
    config.attn_method = model_args.attn_method
    compose_method = model_args.compose_method #my option
    config.compose_method = compose_method 
    compose_target = kwargs.get("compse_target", None)
    if compose_target is None:
        compose_target = "prod" if compose_method in ["mwavg","mcat"] else "sum"
    config.compose_target = compose_target 
    config.select_method = model_args.select_method #my option
    config.target_share_temperature = model_args.target_share_temperature
    config.anneal_min = model_args.anneal_min # my option
    config.anneal_type = model_args.anneal_type # my option
    config.anneal_dir = model_args.anneal_dir # my option
    config.anneal_rate = anneal_rate # my option
    config.attend_target = model_args.attend_target
    config.prompt_out_dim = kwargs.get("out_dim", -1)
    config.num_target_prompts = num_target_prompts
    config.attend_private = use_private_prompts 
    config.use_private_prompts = use_private_prompts
    config.ignore_private = kwargs.setdefault("ignore_private", False)
    config.source_prompts_order = kwargs.setdefault("source_prompts_order", "desc")
    config.padding_pos = kwargs.setdefault("padding_pos", "start")
    config.attend_for = kwargs.setdefault("attend_for", "inp_target")
    config.use_source_prompts = kwargs.setdefault("use_source_prompts", True)
    config.attend_input = model_args.attend_input #my option
    config.add_input = kwargs.setdefault("add_input", False)
    config.route_method = model_args.route_method #my option
    config.normalize = kwargs.setdefault("normalize", True)
    config.bias = kwargs.setdefault("bias", None)
    config.add_target = add_target_prompt #my option
    config.random_source = kwargs.setdefault("random_source", 0)
    config.target_share = model_args.target_share #my option
    config.sig_coef = model_args.sig_coef #my option
    norm_method = kwargs.setdefault("norm_method", "after_sigmoid") #my option
    if "-" in norm_method:
        norm_method, sel_thresh = norm_method.split("-")
        sel_thresh = float(sel_thresh) if sel_thresh != 'none' else None
    else:
        sel_thresh = kwargs.setdefault("sel_thresh", None)
    config.norm_method = norm_method
    config.sel_thresh = sel_thresh
    config.shared_attn = model_args.shared_attn
    if model_args.prompt_embedding_path:
        config.prefix_num = len(model_args.prompt_embedding_path) 
    else:
        config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.temperature = model_args.temperature
    config.learned_temperature = model_args.learned_temperature
    sign_router = kwargs.setdefault("sign_router", False) 
    if sign_router:
       model_args.learn_attention = False
    config.learn_attention = model_args.learn_attention 
    config.learn_source_prompts = learn_source_prompts
    config.learn_target_prompts = model_args.learn_target_prompts

    
    # Load the base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    config.d_model = base_model.config.d_model

    adapter_args.freeze_model = kwargs.get("freeze_model", True)
    adapter_config = get_adapter_config(
        adapter_args, data_args, training_args, config)

    # Wrap model with PEFT

    # Initialize custom model with attentive prompt embedding
    model = PTModel(base_model, config, adapter_config)
    attn_pt = model.attentive_prompt_encoder

    #model = T5ForConditionalGeneration.from_pretrained(
    #    model_name_or_path,
    #    from_tf=bool(".ckpt" in model_name_or_path),
    #    config=config,
        #cache_dir=model_args.cache_dir,
    #    revision=model_args.model_revision,
    #    use_auth_token=True if model_args.use_auth_token else None,
    #    adapter_config=adapter_config
    #)
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    mapl=torch.device('cpu')
    if model_args.load_prefix_embeddings is True:
        if model_args.prompt_embedding_path is None:
            for name, param in model.named_parameters():
                if "prefix_shared" in name or "prefix" in name:
                    shared_params = [param]
        else:
            shared_params = []
            for rel_path in model_args.prompt_embedding_path:
                path = op.join(mylogs.pretPath, "prefixes", rel_path) 
                shared_param = torch.load(path, map_location=mapl)
                shared_params.append(shared_param)
            if model_args.target_prompt_embedding_path is not None:
                target_prompt_embedding = torch.load(
                    model_args.target_prompt_embedding_path, map_location=mapl)

        if model_args.attn_tuning is True:
            if training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is False:
                # Initialize the prompt embeddings using the first prompts
                # Load all of the target prompts
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[0])
            elif training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is True:
                # initialize the embeddings
                # initialize multiple shared embeddings
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_multi(
                    shared_params[0], num_target=config.num_target)
            else:
                # Load prompt embeddings except for the last one
                # Load last prompt embeddings to initialize the target prompt embeddings.
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[-1])

        else:
            if model_args.target_prompt_embedding_path is None:
                model.update_prefix_weights(shared_params)
            else:
                model.update_prefix_weights(
                    shared_params, target_prompt_embedding)

    if model_args.load_attention is True and model_args.attn_path is not None:
        model.update_attention_weights(torch.load(model_args.attn_path, map_location=mapl))

    if model_args.load_attention is True and model_args.attn_path_sub is not None:
        model.update_attention_weights_sub(model_args.attn_path_sub)

    if model_args.load_layer_norm is True and model_args.layer_norm_dir is not None:
        model.update_layer_norm_weights(model_args.layer_norm_dir)

    ######################## My code pppppp
    
    mylogs.bp("router")
    prompts_dir = model_args.prompt_encoders_dir
    if "prompts_prefix" in main_vars or "save_to_prompts_dir" in main_vars:
        prompts_dir = op.join(mylogs.pretPath, "prompts") 
    elif prompts_dir == "save_path":
        base_folder = Path(kwargs.save_path)
        base_folder_stem = base_folder.stem
        base_folder_name = base_folder.name
        prompts_dir = training_args.output_dir.replace(base_folder_name, base_folder_stem)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_saved_router = kwargs.setdefault("use_saved_router", False) 
    router_prefix = kwargs.setdefault("router_prefix", None) 
    use_saved_router = use_saved_router # or (router_prefix and router_prefix in main_vars)
    if router_prefix is None or router_prefix == "1":
        router_prefix = str(data_args.max_train_samples)

    router_prefix = str(router_prefix) + "_" + \
            "-".join(sorted(data_args.task_name)) + "-" + str(num_source_prompts)
    mylogs.bp("router")
    router_dict = None
    init_router = None
    if model_args.attn_tuning is True and use_saved_router:
       dpath = os.path.join(prompts_dir, router_prefix + "_router.pt")
       if not Path(dpath).is_file():
           kwargs["use_saved_router"] = "NA" # Not found
           raise ValueError(dpath + " not found!")
       else:
          router_dict = torch.load(dpath, map_location=device)
          attend_num = len(router_dict)
          model.encoder.router = torch.nn.Parameter(data=torch.empty((
                attend_num,
                attend_num 
          ), device=device).uniform_(0, 0)) #-1e-3, 1e-3
          for i,(k,v) in enumerate(router_dict.items()):
               if sign_router:
                   with torch.no_grad():
                       v[v > 0] = 1.
                       v[v <= 0] = 0.
               model.encoder.router[i].data.copy_(v.data)
          model.encoder.router.to(device=device)
          init_router = model.encoder.router.detach().clone()

    mylogs.bp("penc")
    prompts_prefix = kwargs.setdefault("prompts_prefix", None) 
    if prompts_prefix is not None:
        prompts_prefix = str(prompts_prefix)
    #prompts_prefix = prompts_prefix + "_" + str(data_args.template)
    if prompts_prefix is None or prompts_prefix == "1":
        prompts_prefix = str(training_args.num_train_epochs) + \
                str(data_args.max_train_samples)

    exp_info["load_private_prompts"] = load_private_prompts
    if not load_source_prompts and model_args.attn_tuning:
        prompts_prefix = prompts_prefix 
                # + "_" \
                # + kwargs.experiment.split("/")[0] 
                # + "_" + kwargs.expid

    if not router_prefix:
        router_prefix = prompts_prefix

    shared_mat = None
    if adapter_args.prompt_tuning:
        added = add_specials(tokenizer)
        logger.info("%s tokens was addded", added)
        model.resize_token_embeddings(len(tokenizer))
        # mmmmmmmmmmmmm Add target prompts
        mylogs.bp("encoders")
        prompts = {}
        prompt_sharing = kwargs.setdefault("prompt_sharing", "shared_encoders") 
        tasks = data_args.task_name
        n_tasks = len(tasks)
        task_prompts = {}
        task_source_prompts_set ={}
        label_tokens = []
        #  tttttttttttt
        for ti, (task_name, config_name) in enumerate(zip(tasks, data_args.dataset_config_name), start=1):
             task_args["id"] = ti
             t_args = dotdict(task_args.copy())
             task = AutoTask.get(task_name, config_name, task_args=t_args, tokenizer=tokenizer)
             p = task.get_prompts()
             # label_tokens.extend(task.get_label_list())
             prompts = {**prompts, **p}
             tid = task_name #get_id()
             if not tid in task_prompts:
                 task_prompts[tid] = []
                 task_source_prompts_set[tid] = []
             for k,v in p.items():
                 task_prompts[tid].extend(v)
             rel_sh = REL_TO_SHARED_TOKENS[task_name] if task_name in REL_TO_SHARED_TOKENS else task_name
             task_source_prompts_set[tid].extend(rel_sh.split())

        # extend_tokenizer(tokenizer, label_tokens)
        for name, prompt_tokens in prompts.items():
            extend_tokenizer(tokenizer, prompt_tokens)

        # mmmmmmmmmmmmm Add source prompts
        prompt_encoders = []
        source_prompts = []
        mylogs.bp("nsp")
        nsp = kwargs.get("num_source_prompts", 0)
        if data_args.source_prompts:
            source_prompts = ["source_" + sp for sp in data_args.source_prompts]
        if nsp > 0:
            source_prompts.extend(
                    ["source_com" + str(sp) for sp in range(nsp)])
        if use_private_prompts:
            source_prompts.extend(["source_for_" + t for t in data_args.task_name])
        if use_source_set:
            pset = []
            for t in data_args.task_name:
                pset.extend(task_source_prompts_set[t])
            pset = set(pset)
            source_prompts.extend(["source_" + t for t in pset]) 

        kwargs["num_source_prompts"] = len(source_prompts)
        mylogs.main_args["num_source_prompts"] = len(source_prompts)
        intrinsic_dim = 300
        mylogs.bp("mat")
        if adapter_args.prompt_encoder_type == "mat":
            bound = 1 / math.sqrt(adapter_args.num_prompt_tokens * config.d_model)
            shared_mat = torch.nn.Parameter(data=torch.empty((
                intrinsic_dim,
                adapter_args.num_prompt_tokens * config.d_model
            )).uniform_(-bound, bound), requires_grad=False)

        prompt_num_layers = kwargs.get("num_layers",1)
        prompt_hidden_size = kwargs.get("hidden_size", -1)
        prompt_non_linear = kwargs.get("non_linear", "relu")
        prompt_out_dim = kwargs.get("out_dim", -1)
        for prompt in source_prompts: 
            encoder_name = prompt
            encoder_type = adapter_args.prompt_encoder_type
            if "_for" in encoder_name:
                encoder_type = kwargs.get("private_prompt_encoder_type", encoder_type)
                encoder_type = encoder_type 
            encoder, enc_type = create_encoder(encoder_name, model, tokenizer, 
                    prompt_tokens=[],
                    non_linear = prompt_non_linear,
                    hidden_size = prompt_hidden_size,
                    num_layers = prompt_num_layers,
                    is_source = True,
                    out_dim = prompt_out_dim,
                    length = adapter_args.num_prompt_tokens,
                    encoder_type = encoder_type,
                    shared_mat= shared_mat) 
            if "_for" in encoder_name:
                encoder.is_shared = False
                encoder.is_private = True
            if kwargs.setdefault("init_from_words", False):
                encoder.init_embs_from_words(model.get_input_embeddings())

            if load_source_prompts or (load_private_prompts and encoder.is_private): 
                ignore_if_prompt_not_exists = kwargs.setdefault("ignore_if_prompt_not_exists", False)
                mylogs.bp("load")
                load_prompt = False
                if encoder.is_private:
                    if load_private_prompts: 
                        encoder_name = encoder.name.replace("_for","")
                        load_prompt = True
                elif encoder.is_source:
                    load_prompt = True
                    if "_com" in encoder.name and not reval:
                        #pattern = re.compile(r"com\d+")
                        #enc_name = re.sub(pattern, "com", encoder.name)
                        encoder_name = encoder.name.replace("source_", "")
                        load_prompt = False
                if load_prompt: 
                    is_loaded = encoder.load(os.getcwd(), 
                        prefix=prompts_prefix,
                        ignore_if_prompt_not_exists=True,
                        length = adapter_args.num_prompt_tokens,
                        name=encoder_name)
                    if not is_loaded:
                        is_loaded = encoder.load(prompts_dir, 
                            prefix=prompts_prefix,
                            ignore_if_prompt_not_exists=ignore_if_prompt_not_exists,
                            length = adapter_args.num_prompt_tokens,
                            name=encoder_name)
                    if is_loaded:
                        logger.info("%s was loaded", encoder.name)
                    else:
                        logger.info("% doesn't exist and wasn't loaded", encoder.name)
                    if bp == "load":
                        breakpoint()
                    exp_info["load_" + prompt] = is_loaded
            prompt_encoders.append(encoder)

        ############################ Create Target Prompt Encoders #############
        mylogs.bp("mask")
        encoders_prompts = prompts
        # task prompts has one encoder per task where they could have shared tokens
        # shared encoders has one encoder per prompt ids. 
        # If two tasks use similar prompts they recieve the output of same encoders
        if prompt_sharing == "shared_prompts":
            encoders_prompts = task_prompts
        model.resize_token_embeddings(len(tokenizer))
        load_prompts = kwargs.setdefault("load_prompts", False) 
        if training_args.do_train:
            load_prompts = False
        attend_to_all = kwargs.setdefault("attend_to_all", True) 
        attend_to_all = attend_to_all and use_source_prompts
        target_prompts=[n for n,p in encoders_prompts.items() if p[0].startswith("<tar-")]  
        # create and load target prompts
        mylogs.bp("usp")
        num_attend_to = len(source_prompts) + len(target_prompts) + 1 # one for input 
        for name, prompt_tokens in encoders_prompts.items():
            encoder_type = adapter_args.prompt_encoder_type 
            if prompt_tokens[0].startswith("<tar-"):
                encoder_type = kwargs.get("target_encoder_type", encoder_type)
                prompt_non_linear = kwargs.get("target_non_linear", prompt_non_linear)
            encoder, enc_type = create_encoder(name, model, tokenizer, 
                    prompt_tokens, 
                    non_linear = prompt_non_linear,
                    hidden_size = prompt_hidden_size,
                    num_layers = prompt_num_layers,
                    in_dim = prompt_out_dim,
                    out_dim = -1,
                    encoder_type=encoder_type, 
                    shared_mat= shared_mat) 

            opp = kwargs.setdefault("output_prompts_prefix", prompts_prefix) 
            if opp is None:
                opp = str(training_args.num_train_epochs) + \
                    str(data_args.max_train_samples)
            skip_if_prompt_exists = kwargs.setdefault("skip_if_prompt_exists", True) 
            prompt_exists, prompt_fname = encoder.exists(prompts_dir, 
                prefix=str(opp),
                as_saved=True,
                length = target_prompt_length)
            if not model_args.attn_tuning and prompt_exists and skip_if_prompt_exists: 
                print("prompt exists: ", prompt_fname)
                return

            if name in task_source_prompts_set:
                encoder.attend_to.extend(
                        ["source_" + x for x in task_source_prompts_set[name]])
            if prompt_tokens[0].startswith("<tar-"):
                encoder.is_target = True
                nn = name.replace("tar-","")
                encoder.attend_to.extend(["source_for_" +  nn])
            elif prompt_tokens[0].startswith("<com_"):
                encoder.is_common = True
            if False: #TODO router_dict and name in router_dict:
                encoder.attend_to_mask = [1 if r > 0.1 else 0 for r in router_dict[name]] 
            else: 
                mylogs.bp("mask")
                if use_source_prompts:
                    encoder.attend_to_mask = [1]*num_attend_to  
                else:
                    encoder.attend_to_mask = [0]*num_attend_to  
                    encoder.attend_to_mask[0] = 1 # for input
                attn_flag = False
                all_prompts = source_prompts + target_prompts
                for i, n in enumerate(all_prompts, start=1):
                    #encoder.attend_to_mask[i] = 0 
                    #if (n in encoder.attend_to or "_com" in n) and use_source_prompts:
                    #    encoder.attend_to_mask[i] = 1 
                    #    attn_flag = True
                    if "_for" in n: 
                        if n in encoder.attend_to:
                            encoder.attend_to_mask[i] = 1 
                        else:
                            encoder.attend_to_mask[i] = 0 
                        attn_flag = True
                    if "tar-" in n: 
                        if "source_" + n in encoder.attend_to and add_target_prompt:
                            encoder.attend_to_mask[i] = 1 
                        else:
                            encoder.attend_to_mask[i] = 0 
                        attn_flag = True
                # TODO it seems unnecessary
                if not attn_flag or (not use_private_prompts and not use_source_set): 
                    if use_source_prompts:
                        encoder.attend_to_mask = [1]*num_attend_to # attend to all 
            if kwargs.setdefault("init_from_words", False):
                encoder.init_embs_from_words(model.get_input_embeddings())
            if not model_args.attn_tuning and load_prompts: 
                ignore_if_prompt_not_exists = kwargs.setdefault("ignore_if_prompt_not_exists", False)
                # if not model_args.attn_tuning or encoder.is_source:
                is_loaded = encoder.load(prompts_dir, 
                        prefix=prompts_prefix,
                        ignore_if_prompt_not_exists=ignore_if_prompt_not_exists,
                        as_saved=True,
                        length = target_prompt_length)
                ignore_train_if_exist = kwargs.setdefault("ignore_train_if_exist", False)
                if is_loaded and ignore_train_if_exist:
                    training_args.do_train = False
                    logger.info("%s training was ignored", encoder.name)
                if bp == "load":
                    breakpoint()
            prompt_encoders.append(encoder)

        exp_info["num_encoders"] = len(prompt_encoders)
        exp_info["len_encoders"] = ",".join([str(e.length) for e in prompt_encoders])
        exp_info["taginfo"].append("len_encoders")
        tasks = data_args.task_name
        mylogs.bp("setenc")
        attn_pt.set_encoders(prompt_encoders, 
            source_prompts, 
            source_prompt_length,
            target_prompt_length, tasks = tasks) 
        model.resize_token_embeddings(len(tokenizer))

    if log_var and preview == "encoders":
        mylogs.plog.info("======== Number of encoders: %s", len(prompt_encoders))
        for ii, e in enumerate(prompt_encoders):
            mylogs.plog.info("%s) Name:%s, length: %s", ii, e.name, e.length)
            mylogs.plog.info("Tokens:%s", e.prompt_tokens)
            mylogs.plog.info("Ids:%s ", e.prompt_ids)
            mylogs.plog.info(e)
        print("preview is encoders")
        return 

    mylogs.bp("freeze")
    mylogs.bp("rgrad")
    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    mylogs.plog.info("Before freeze: requires grad: %s   Not requires grad: %s", rgrad, nrgrad)
    model = modify_model_after_init(
        model, training_args, adapter_args, adapter_config)
   
    learn_loaded_prompts = kwargs.setdefault("learn_loaded_prompts", True) 
    learn_private_prompts = kwargs.setdefault("learn_private_prompts", True) 
    requires_grad_encoders = []
    if adapter_args.prompt_tuning:
        for encoder in prompt_encoders: 
            if encoder.is_private and learn_private_prompts:
                for n,p in encoder.named_parameters():
                    p.requires_grad = True
                    requires_grad_encoders.append(encoder.name)
                continue
            elif encoder.is_source:
                mylogs.bp("learn")
                if learn_source_prompts:
                    if encoder.is_private and not learn_private_prompts:
                        continue
                    if encoder.is_loaded and not learn_loaded_prompts:
                        continue
                    for n,p in encoder.named_parameters():
                        p.requires_grad = True
                        requires_grad_encoders.append(encoder.name)
            else:
                if model_args.learn_target_prompts:
                    for n,p in encoder.named_parameters():
                        p.requires_grad = True
                        requires_grad_encoders.append(encoder.name)

    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    exp_info["rgrad-nrgrad"] = str(rgrad) + "|" + str(nrgrad)
    mylogs.minfo("After freeze: requires grad: %s   Not requires grad: %s", rgrad, nrgrad)
    # mylogs.minfo("Encoders require grad: %s",requires_grad_encoders)
    mylogs.bp("freeze")

    # Load training set
    data_args.dataset_name = data_args.task_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    ########### rrrrrr
    hit_count = kwargs.setdefault("hc", 3)
    def preprocess_function(examples, max_target_length, task_id=None):
        mylogs.bp("data")
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        if preview == "data":
            print("sourece: %s", examples["source"][:hit_count])
            print("target: %s", examples["target"][:hit_count])

        if bp and bp in "data|examples":
            logger.info("sourece: %s", examples["source"][:5])
            logger.info("target: %s", examples["target"][:5])
            if "extra_fields" in examples:
                logger.info("extra: %s", examples["extra_fields"][:5])
            breakpoint()
        # Setup the tokenizer for targets
        mylogs.bp("encode")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        #if preview == "data":
        #    logger.info("target encoded: %s", labels)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        #if preview == "data":
        #    logger.info("target encoded input ids: %s", labels["input_ids"])
        # Check for <extra_id_x> in the source and reconstruct full sentence
        full_sentences = []
        for source, target in zip(examples['source'], examples['target']):
            if "<extra_id_" in source:
                for i, segment in enumerate(target.split("<extra_id_")):
                    if i > 0:  # Skip the first part before <extra_id_0>
                        placeholder = f"<extra_id_{i-1}>"
                        source = source.replace(placeholder, segment.split(">")[1], 1)
                full_sentence = source
            else:
                full_sentence = source + " " + target

            full_sentences.append(full_sentence)

        # Tokenize the full sentence and add to model_inputs
        full_tokenized = tokenizer(full_sentences, max_length=data_args.max_source_length + max_target_length,
                                   padding=padding, truncation=True)
        model_inputs["full_ids"] = full_tokenized["input_ids"]
        ##################
        if "task_ids" in examples["extra_fields"]:
            model_inputs["task_ids"] = examples["extra_fields"]["task_ids"]
        mylogs.bp("train_test_data")
        model_inputs["extra_fields"] = examples['extra_fields']  
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    task_args = dotdict(task_args.copy())
    inex_training_samples = kwargs.get("inex_training_samples", data_args.max_train_samples)
    if training_args.do_train:
        # Load datasets from files if your target datasets are not in huggingface datasets.
        train_datasets = []
        max_target_lengths = []
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, 
                data_args.dataset_config_name):
            n_obs = data_args.max_train_samples 
            if dataset_name in include_exclude_tasks:
                n_obs = inex_training_samples
            for prefix in train_prefix[dataset_name]:
                auto_task = AutoTask.get(dataset_name,
                                         dataset_config_name,
                                         task_args=task_args, tokenizer=tokenizer)
                print("loading train dataset for " + prefix)
                train_ds = auto_task.get(
                        split="train",
                        split_validation_test=training_args.split_validation_test,
                        prefix=prefix,
                        n_obs=n_obs,
                        lang=data_args.lang_name, file_name=data_args.train_file)
                train_datasets.append(train_ds)

                mtl = auto_task.get_max_target_length(
                                tokenizer=tokenizer, 
                                default_max_length=data_args.max_target_length)
                max_target_lengths.append(mtl)
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function,
                                  max_target_length=max_target_lengths[i]
                                  #mycode adding task ids
                                  ,task_id=i
                                  ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # if train_dataset != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        if trainer_shuffle:
            train_dataset = concatenate_datasets(train_datasets)
        else:
            mylogs.bp("myint")
            train_dataset = my_interleave_datasets(train_datasets, 
                batch_size=training_args.per_device_train_batch_size)
    if preview == "data":
       print("preview is data")
       return "data_preview" 

    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                    task_args=task_args, tokenizer=tokenizer).get(
            split="validation",
            split_validation_test=training_args.split_validation_test,
            prefix=train_prefix[dataset_name],
            n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, 
            dataset_config_name,
            task_args=task_args, tokenizer=tokenizer).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]

        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                functools.partial(preprocess_function,
                                  max_target_length=max_target_lengths[k]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # if name != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if preview == "template":
        print("preview is template")
        return
    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    eval_metrics = [AutoTask.get(dataset_name, 
                    dataset_config_name, task_args=task_args, tokenizer=tokenizer).metric
                    for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    print(data_args.eval_dataset_name)
    compute_metrics_fn = build_compute_metrics_fn(
        data_args.eval_dataset_name, tokenizer, data_args.ignore_pad_token_for_loss) if training_args.predict_with_generate else None
    print(compute_metrics_fn)

    data_info = {}
    has_extra = kwargs.setdefault("has_extra", True)
    if has_extra:
        data_info["eval"] = eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'] if training_args.do_eval else None
        data_info["train"] = train_dataset['extra_fields'] if training_args.do_train else None

    def compute_metrics(eval_preds):
        preds, labels, data_info, task = eval_preds
        post_processor = AutoPostProcessor.get(task, tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        task = AutoTask.get(task, None, task_args=task_args, tokenizer=tokenizer)
        mylogs.bp("compute")
        decoded_preds, decoded_labels = task.post_process(decoded_preds, decoded_labels)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # If you want to use a different learning rate for attention layer, initialize an optimizer using the learning rate here.
    grouped_params = []
    all_parameters = set([p for p in model.parameters() if p.requires_grad])
    attn_params = []
    prompt_params = []
    mylogs.bp("lr")
    if model_args.attn_learning_rate is not None and model_args.learn_attention:
        for name, param in model.named_parameters():
            if (name == "encoder.attn_W_up.weight" 
                or name == "encoder.attn_W_down.weight" 
                or name == "encoder.layer_norm.weight"
                or name == "encoder.router" 
                or name == "encoder.target_router"):
                   attn_params.append(param)

        attn_params = set(attn_params)
        grouped_params.append({'params': list(attn_params), 
            'lr': model_args.attn_learning_rate})
        

    ########### My Code
    if "learning_rate" in main_vars:
        model_args.prompt_learning_rate = training_args.learning_rate 
        model_args.target_learning_rate = training_args.learning_rate 

    prompt_learning_rate = model_args.prompt_learning_rate 
    target_prompt_learning_rate = model_args.target_prompt_learning_rate 
    source_prompt_learning_rate = model_args.source_prompt_learning_rate 
    private_prompt_learning_rate = model_args.private_prompt_learning_rate 
    if source_prompt_learning_rate is None:
        source_prompt_learning_rate = prompt_learning_rate 
    if target_prompt_learning_rate is None:
        target_prompt_learning_rate = prompt_learning_rate 
    if private_prompt_learning_rate is None:
        private_prompt_learning_rate = prompt_learning_rate 
    shr_prompt_params = []
    src_prompt_params = []
    tgt_prompt_params = []
    pvt_prompt_params = []
    mylogs.bp("opt")
    learning_rate = training_args.learning_rate
    if adapter_args.prompt_tuning:
        if "learning_rate" in main_vars:
            target_prompt_learning_rate = learning_rate
        learning_rate = target_prompt_learning_rate
        for encoder in attn_pt.prompt_encoders:
           para_list =[
                   p for n, p in encoder.named_parameters() if p.requires_grad and n != "A"]
           if para_list: 
               if encoder.is_source and not encoder.is_private:
                   src_prompt_params.extend(para_list)
                   #if encoder.name in source_prompts:
                   #    src_prompt_params.extend(para_list)
                   #else:
                   #    shr_prompt_params.extend(para_list)
               elif encoder.is_private:
                   pvt_prompt_params.extend(para_list)
               else:
                   tname = encoder.name.replace("tar-","")
                   if tname in exclude_from_test_tasks:
                       src_prompt_params.extend(para_list)
                   else:
                       tgt_prompt_params.extend(para_list)

        src_prompt_params = set(src_prompt_params)
        shr_prompt_params = set(shr_prompt_params)
        tgt_prompt_params = set(tgt_prompt_params)
        pvt_prompt_params = set(pvt_prompt_params)
        grouped_params.append({'params': list(shr_prompt_params), 
            'lr': prompt_learning_rate})
        grouped_params.append({'params': list(src_prompt_params), 
            'lr': source_prompt_learning_rate})
        grouped_params.append({'params': list(tgt_prompt_params), 
            'lr': target_prompt_learning_rate})
        grouped_params.append({'params': list(pvt_prompt_params), 
            'lr': private_prompt_learning_rate})
        prompt_params = list(src_prompt_params) \
                + list(tgt_prompt_params) + list(pvt_prompt_params)

    other_params = all_parameters - set(attn_params) - set(prompt_params)
    other_params = list(other_params)
    if shared_mat is not None:
        other_params.append(shared_mat)
    if other_params:
        grouped_params.append({'params': other_params, 'lr': training_args.learning_rate})
    #### ooooo 
    mylogs.bp("opt")
    mylogs.bp("optim")
    opt_type = kwargs.get("opt_type","adam")
    scheduler = None
    if opt_type == "sep":
        optim, scheduler = get_optimizer(model, steps,
                source_prompt_learning_rate, 
                model_args.attn_learning_rate, 0.01)
    elif opt_type in ["adam", "regular"]:
        optim = AdamW(grouped_params, lr=learning_rate)
    elif opt_type == "ada":
        optim = Adafactor(
            grouped_params,
            lr=learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        ) 
    if scheduler is None:
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps)
    name = data_args.dataset_name[0] 
    task_metric = TASK_TO_METRICS[name] if name in TASK_TO_METRICS else ["rouge"]
    if training_args.do_eval: 
        eval_ds = my_interleave_datasets(list(eval_datasets.values()), batch_size=2)
    else: 
        eval_ds = None
    image_folder = op.join(training_args.output_dir,"router_images")
    Path(image_folder).mkdir(parents=True, exist_ok = True)
    save_router_image = kwargs.get("save_router_image", False)
    wb_callback = WBCallback(save_path = image_folder, 
            save_router_image=save_router_image)
    anneal_callback = AnnealCallback() 
    ptlr_callback = PTLearningRateCallback()
    callbacks = []
    # optimizer = AdamW(model.prompt_encoder.parameters(), lr=1e-4)  # Optimize only soft prompts
    if model_args.attn_tuning:
       callbacks = [ptlr_callback, wb_callback, anneal_callback]
    if kwargs.use_optimizer: #TODO remove condition and the else part 
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset= eval_ds,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            multi_task_compute_metrics=compute_metrics_fn,
            evaluation_metrics=task_metric,
            save_checkpoint = kwargs.setdefault("save_checkpoint", False),
            shared=model_args.shared_attn,
            callbacks = callbacks, 
            shuffle = trainer_shuffle,
            optimizers=(optim, scheduler)
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_ds,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks = callbacks, 
            shuffle = trainer_shuffle,
            save_checkpoint = kwargs.setdefault("save_checkpoint", False),
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            evaluation_metrics=task_metric,
            multi_task_compute_metrics=compute_metrics_fn,
            shared=model_args.shared_attn)

    # Exit program if user wants to check some settings 
    if preview and preview != "one" and preview != "run" and preview != "test":
        print("preview is ", preview)
        return
    # Saves training config.
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_training_config(config_file, training_args.output_dir)

    def load_model(load_path, lsp=False):
        #model.load_encoders(load_path, load_source_prompts=lsp)
        mylogs.bp("load_model")
        dpath = os.path.join(load_path, "attn_W_down.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        saved_prompts_prefix = kwargs.get("saved_prompts_prefix", "*")
        saved_prompts_prefix = str(saved_prompts_prefix)
        attention_paths = [dpath, 
                os.path.join(load_path, "attn_W_up.pt")]
        if model_args.attn_tuning is True and Path(dpath).is_file():
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm and "layer_norm_bias.pt" in load_path: 
                trainer.model.update_layer_norm_weights(load_path)
        if lsp:
            for encoder in model.prompt_encoders:
                plen = encoder.length
                encoder.load(load_path, 
                        prefix = saved_prompts_prefix,
                        ignore_if_prompt_not_exists=False,
                        length = plen) 
                encoder.to(device)
        dpath = os.path.join(load_path, router_prefix + "_router.pt")
        mylogs.bp("router")
        if model_args.attn_tuning is True:
              assert Path(dpath).is_file(), dpath + " doesn't exist to load model"
              router_dict = torch.load(dpath, map_location=device)
              attend_num = len(router_dict)
              model.encoder.router = torch.nn.Parameter(data=torch.empty((
                    attend_num,
                    attend_num 
              ), device=device).uniform_(0, 0)) #-1e-3, 1e-3
              for i,(k,v) in enumerate(router_dict.items()):
                   if sign_router:
                       with torch.no_grad():
                           v[v > 0] = 1.
                           v[v <= 0] = 0.
                   model.encoder.router[i].data.copy_(v.data)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if reval:
            load_model_dir = kwargs.get("load_model_dir", training_args.output_dir)
            load_model(load_model_dir, lsp=True) 

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        print("=================== Training ====================")
        print("Experiment: ", mylogs.args("expid"), "/", mylogs.args("total_exp"))
        print("Tags: ", mylogs.get_tag(as_str=True))
        print("=================================================")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        # Load best model
        if trainer.best_prompt_checkpoint is not None:
            best_chk_path = trainer.best_prompt_checkpoint
            lsp = kwargs.setdefault("load_source_prompts", False)
            load_model(best_chk_path, lsp)

        # Save prompts
        if adapter_args.prompt_tuning:
            #if not model_args.attn_tuning: 
            #    prompts_prefix = "pt_" + prompts_prefix 
            #else: 
            #    prompts_prefix = "att_" + prompts_prefix 
            #prompts_prefix = prompts_prefix.strip("_")
            mylogs.bp("save_prompts")
            prompts_to_save = kwargs.setdefault("prompts_to_save", None) 
            save_all_prompts = kwargs.setdefault("save_all_prompts", True) 
            ssp = kwargs.setdefault("save_source_prompts", save_all_prompts) 
            opp = kwargs.setdefault("output_prompts_prefix", prompts_prefix) 
            if opp is None:
                opp = str(training_args.num_train_epochs) + \
                    str(data_args.max_train_samples)
            if not prompts_to_save:
                prompts_to_save = "all" if save_all_prompts else None
            model.store_encoders(output_dir = training_args.output_dir,
                                 prompts_and_router_only=model_args.attn_tuning, 
                                 save_source_prompts = ssp, 
                                 prompts_to_save = prompts_to_save, 
                                 save_router=True,
                                 prefix=str(opp),
                                 router_prefix=router_prefix)

            save_router = kwargs.setdefault("save_router", False) 
            save_to_prompts_dir = kwargs.get("save_to_prompts_dir", False) 
            mylogs.bp("store")
            if save_to_prompts_dir or save_router:
                Path(prompts_dir).mkdir(parents = True, exist_ok=True)
                model.store_encoders(output_dir = prompts_dir, 
                        prompts_and_router_only=model_args.attn_tuning, 
                        prompts_to_save = prompts_to_save or "all", 
                        save_source_prompts = ssp,
                        save_router = save_router,
                        prefix=str(opp),
                        router_prefix=router_prefix)

        save_model_default = data_args.max_train_samples > 2000
        if save_model_default:
            save_model_default = "template"
        save_model = kwargs.setdefault("save_model", save_model_default)
        if save_model:
            # save all model parameters and tokenizers 
            # regardless of whether they are updated or not.
            model_name = Path(model_name_or_path).stem
            parts = save_model.split("-")
            save_model = model_name
            for part in parts:
                if part in param_map:
                    part = param_map[part]
                if part in kwargs:
                    part = kwargs[part]
                    if type(part) == list:
                        part="@".join([str(s) for s in part])
                    save_model += "-" + part 
                else:
                    save_model += "-" + part
            save_model += "-" + str(data_args.max_train_samples)
            pret_dir = op.join(mylogs.pretPath, save_model) 
            trainer.save_model(pret_dir)

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        train_metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        if not model_args.save_prefix_only:
            trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics)

    # Validation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if model_args.attn_tuning is True:
            load_model_dir = kwargs.get("load_model_dir", training_args.output_dir)
            load_model(load_model_dir, lsp=True) 

        if  model_args.shared_attn is False:
            for task, eval_dataset in eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                metric_to_check = training_args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]
                if use_wandb:
                    wandb.run.summary[f"evaluation_{metric_to_check}"] = metric_value 

    # Test
    mylogs.bp("do_test")
    reval = not training_args.do_train 
    slen = len([e for e in model.prompt_encoders if e.is_source and not e.is_private]) 
    exp_info["slen"] = slen
    load_model_dir = kwargs.get("load_model_dir", training_args.output_dir)
    use_test_config = kwargs.get("use_test_config", False)
    if reval: 
        load_model(load_model_dir, 
                lsp=model_args.attn_tuning)
    for k,v in kwargs.items():
        if not k in exp_info and not k.startswith("comment"):
            exp_info[k] = v
    if training_args.do_test:
        mylogs.bp("test_prefix")
        test_datasets = {}
        max_target_lengths = []
        first_ds = ""
        auto_tasks = {}
        for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, 
                data_args.test_dataset_config_name): 
            if test_dataset in exclude_from_test_tasks:
                continue
            if not use_test_config and not test_dataset_config in ["en", "def"]:
                test_dataset_config = "def"
            auto_task = AutoTask.get(
                test_dataset, test_dataset_config,
                task_args=task_args, tokenizer=tokenizer)
            for prefix in test_prefix[test_dataset]:
                if use_test_config:
                    ds_key = test_dataset + "_" + test_dataset_config + "_" + prefix
                else:
                    ds_key = test_dataset  + "_" + prefix
                if first_ds == "": first_ds = ds_key
                auto_tasks[ds_key] = auto_task
                test_datasets[ds_key]= auto_task.get(
                        split="test",
                        split_validation_test=training_args.split_validation_test,
                        prefix=prefix,
                        n_obs=data_args.max_test_samples, 
                        lang=data_args.lang_name, 
                        file_name=data_args.test_file)

                mtl = auto_task.get_max_target_length(
                    tokenizer=tokenizer, 
                    default_max_length=data_args.max_target_length)
                max_target_lengths.append(mtl)
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                functools.partial(preprocess_function,
                                  max_target_length=max_target_lengths[k]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        if has_extra and first_ds:
            data_info["test"] = test_datasets[first_ds]['extra_fields'] if training_args.do_test else None
        logger.info("*** Test ***")
        
        no_mask_preds = {}
        # multi-task evaluations

        def compute_depth_rank_and_perplexity(model, tokenizer, input_ids, attention_mask, labels, prediction):
            # Ensure model is in evaluation mode
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            labels = torch.tensor(labels).to(device)

            # Ensure input is treated as a batch of size 1
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)

            model.eval()

            # Initialize accumulators for perplexity and depth rank
            total_log_likelihood = 0.0
            depth_ranks = []
            pred_tokens = []

            # Initialize decoder with start token for T5 (using <pad> token in this case)
            decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0).to(device)

            # Loop through the target sequence (label tokens) token-by-token
            for i in range(len(labels[0])):  # Start from the first token in labels
                with torch.no_grad():
                    # Forward pass through the model (T5 requires both encoder and decoder inputs)
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids)

                # Get logits for the next token prediction
                logits = outputs.logits[:, -1, :]  # Logits for the last token in the sequence

                # Compute softmax probabilities
                prob_dist = torch.softmax(logits, dim=-1)

                # Get the true next token
                true_next_token = labels[0, i].item()
                next_token = tokenizer.decode(true_next_token) 
                if next_token.startswith("<"):
                    decoder_input_ids = torch.cat([decoder_input_ids, labels[0, i].unsqueeze(0).unsqueeze(0)], dim=1).to(device)
                    continue
                pred_tokens.append(next_token)

                # Compute log likelihood (negative log-probability of the true token)
                log_prob = torch.log(prob_dist[0, true_next_token])
                total_log_likelihood += log_prob.item()

                # Compute DepthRank
                sorted_indices = torch.argsort(prob_dist, descending=True).squeeze(0).tolist()

                # Ensure correct token ranking
                if true_next_token in sorted_indices:
                    rank = sorted_indices.index(true_next_token) + 1  # Find the index and add 1 (1-based rank)
                else:
                    rank = len(sorted_indices)  # If not found, assign the worst possible rank

                depth_ranks.append(rank)

                # Update decoder input by appending the true token
                decoder_input_ids = torch.cat([decoder_input_ids, labels[0, i].unsqueeze(0).unsqueeze(0)], dim=1).to(device)

            # Compute average log likelihood and perplexity
            avg_log_likelihood = total_log_likelihood / len(labels[0])
            perplexity = torch.exp(-torch.tensor(avg_log_likelihood)).item()

            # Compute DepthRank for the entire sentence
            depth_rank = sum(depth_ranks) / len(depth_ranks)

            return perplexity, depth_rank, depth_ranks, pred_tokens


        def compute_depth_rank_and_perplexity2(model, input_ids, attention_mask, labels):
            # Ensure model is in evaluation mode
           # Convert numpy arrays to PyTorch tensors
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            labels = torch.tensor(labels).to(device)

            # Ensure input is treated as a batch of size 1
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)

            # Get model outputs without computing gradients
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)


            # Get model outputs without computing gradients
            loss = outputs.loss
            logits = outputs.logits

            # Compute perplexity
            perplexity = torch.exp(loss)

            # Compute DepthRank
            depth_ranks = []
            batch_size, seq_len, vocab_size = logits.size()
            for i in range(seq_len):
                logits_i = logits[0, i]  # shape: (vocab_size)
                token_id = labels[0, i]
                prob_dist = torch.softmax(logits_i, dim=-1)
                sorted_indices = torch.argsort(prob_dist, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
                depth_ranks.append(rank)

            depth_rank = sum(depth_ranks) / len(depth_ranks)

            return perplexity.item(), depth_rank

        def compute_sentence_perplexity(model, full_ids, tokenizer):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.eval()

            # Convert inputs to tensors and move to device
            full_ids = torch.tensor(full_ids).to(device)

            # Initialize accumulators for log likelihood and depth rank
            total_log_likelihood = 0.0
            depth_ranks = []

            # Start decoder with <pad> token
            decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0).to(device)

            # Loop through each token and compute the log likelihood
            for i in range(len(full_ids)):  # Now starting from the first token of full_ids
                # Get the current input sequence up to the ith token (encoder input)
                current_input_ids = full_ids.unsqueeze(0)  # Treat the input as a batch of size 1
                current_attention_mask = torch.ones(current_input_ids.size(), dtype=torch.long).to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,  # Dummy input for the encoder
                        attention_mask=current_attention_mask,
                        decoder_input_ids=decoder_input_ids
                    )

                # Get logits for the next token prediction
                logits = outputs.logits[:, -1, :]  # Get logits for the last token

                # Compute softmax probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get the log probability of the true next token
                true_next_token = full_ids[i]
                log_prob = torch.log(probs[0, true_next_token])

                # Accumulate the log probability
                total_log_likelihood += log_prob.item()

                # Compute Depth Rank
                sorted_indices = torch.argsort(probs, descending=True).squeeze(0).tolist()
                if true_next_token in sorted_indices:
                    rank = sorted_indices.index(true_next_token) + 1  # Find the 1-based index
                else:
                    rank = len(sorted_indices)  # If not found, assign the worst rank

                # Add rank to depth ranks list
                depth_ranks.append(rank)

                # Update the decoder input ids by appending the true next token
                decoder_input_ids = torch.cat([decoder_input_ids, full_ids[i].unsqueeze(0).unsqueeze(0)], dim=1).to(device)

            # Compute average log likelihood and perplexity
            avg_log_likelihood = total_log_likelihood / len(full_ids)
            avg_log_likelihood_tensor = torch.tensor(avg_log_likelihood)
            perplexity = torch.exp(-avg_log_likelihood_tensor).item()

            # Compute Depth Rank for the entire sentence
            depth_rank = sum(depth_ranks) / len(depth_ranks)

            return perplexity, depth_rank

        def evaluate_test(task, test_dataset, save_to, ds_name, auto_task, 
                gen_conf = {}, use_cache=False):
            mylogs.bp("ttt")
            if use_cache and task in no_mask_preds:
                predictions, labels, metrics = no_mask_preds[task] 
            else:
                predictions, labels, metrics = trainer.predict(
                    gen_conf = gen_conf,
                    test_dataset=test_dataset,
                    max_length=data_args.test_max_target_length, 
                    num_beams=data_args.num_beams,
                    metric_key_prefix="test", task=task)

            if adapter_args.prompt_tuning and gen_conf["mask_type"].startswith("no-mask"):
                no_mask_preds[task] = (predictions, labels, metrics)
            
            mylogs.bp("gen")
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            skip_specials=kwargs.setdefault("skip_specials", True) 

            # sssssssssss
            #predictions = np.argmax(predictions, axis=1)
            #predictions = tokenizer.batch_decode(predictions)
            df = test_dataset.to_pandas()
            if bp == "test": breakpoint()
            df["pred_text1"] = ""
            df["prefix"] = ds_name
            df["template"] = data_args.template
            df["resp"] = ""
            df["time"] = mylogs.now 
            df["date"] = mylogs.today 
            df["query"] = ""
            df["vtarget"] = ""
            df["langs"] = "en2en"
            for k,v in metrics.items():
                df[k] = v
            #df["rouge_score"] = 0.0
            #df["bert_score"] = 0.0
            for key, info in exp_info.items():
                if type(info) == list:
                    info = "@".join([str(inf) for inf in info])
                if type(info) == dict:
                    info = json.dumps(info)
                    info = info.replace("\n", "@")
                df[key] = info
            rouge_scorer = Rouge()
            golds, preds = [], []
            for i, row in df.iterrows():
                mylogs.bp("=testloop") 
                extra = row["extra_fields"]
                if "event" in extra:
                    inp = extra["event"]
                else:
                    inp = tokenizer.decode(row["input_ids"], 
                        skip_special_tokens=skip_specials) 
                inp = re.sub(r'<.*?>','', inp)
                inp = inp.strip()
                df.at[i, "input_text"] = inp #extra["event"] 
                label = extra["tail"] if "tail" in extra else "na"
                #label = tokenizer.decode(row["labels"], 
                if skip_specials:
                    label = re.sub(r'<.*?>','', label)
                    label = label.strip()
                else:
                    label = extra["target_text"].strip()
                if "vtarget" in extra:
                    vtarget = extra["vtarget"]
                    df.at[i, "vtarget"] = vtarget 
                golds.append(label)
                df.at[i, "target_text"] = label 
                #sel = False
                #if "sel" in extra:
                #    sel = extra["sel"] 
                #df.at[i, "sel"] = sel 
                df.at[i, "query"] = extra["query"]  
                df.at[i, "resp"] =  extra["resp"]  
                mylogs.bp("decode")
                pred = tokenizer.decode(predictions[i], 
                        skip_special_tokens=skip_specials) 
                if skip_specials:
                    pred = re.sub(r'<.*?>','',pred)
                else:
                    pred = pred.replace("<pad>","").replace("</s>","")
                pred = pred.strip()
                preds.append(pred)
                df.at[i, "pred_text1"] = pred

                # Compute perplexity and DepthRank
                # Extract input_ids, attention_mask, and labels from the row
                input_ids = row["input_ids"]
                attention_mask = row["attention_mask"]
                labels = row["labels"]
                # Compute perplexity and DepthRank using the precomputed values
                if kwargs.get("depth", False):
                    perplexity, depth_rank, depth_ranks, pred_tokens = compute_depth_rank_and_perplexity(model, tokenizer, input_ids, attention_mask, labels, predictions[i])

                    df.at[i, 'perp_score'] = perplexity
                    df.at[i, 'depth_score'] = depth_rank
                    df.at[i, 'depth_ranks'] = ";".join([str(d) for d in depth_ranks])
                    df.at[i, 'pred_tokens'] = ";".join([str(d) for d in pred_tokens])
                
                #full_ids = row["full_ids"]
                #full_perplexity, full_depth_rank = compute_sentence_perplexity(model, full_ids, tokenizer)
                #df.at[i, 'full_score'] = full_perplexity 
                #df.at[i, 'full_depth_rank'] = full_depth_rank 

            df = df.drop(columns=["input_ids","labels","attention_mask"])
            # assert task in TASK_TO_METRICS, "There is no metric for task " + task
            if task in TASK_TO_METRICS:
                task_metric = TASK_TO_METRICS[task] 
            else:
                task_metric = TASK_TO_METRICS["default"] 
            metrics_list = []
            mylogs.bp("metrics")
            for mstr in task_metric:
                metric = getattr(mets, mstr)
                try:
                    met = metric(preds, golds)
                except Exception as e:
                    met = {mstr:0}
            mm = 0
            for k,v in met.items():
                df[k] = v
                df["metric_"+ str(ii)] = v
                if mm == 0:
                    df["m_score"] = round(float(v),1) 
                mm += 1
            df = auto_task.before_scoring(df)
            scores = do_score(df, "rouge", save_to, use_wandb=use_wandb)
            auto_task.after_scoring(df, golds, preds)
            return df, scores, golds, preds

        ################ Draw image
        def save_image(folder, model, score_dict, spec, p_labels=None, 
                square=False, annot=True, vmin=None, vmax=None, mask_zeros=False):
            if not model_args.attn_tuning:
                return
            targets = model.encoder.target_encoders_idx
            mylogs.bp("save_image")
            y_labels = [model.encoder.prompt_names[i] for i in targets]
            y_labels = [y.replace("tar-","") for y in y_labels]
            y_labels = [p.split("-")[-1] for p in y_labels]
            if not p_labels:
                p_labels = []
                for pl in model.encoder.prompt_names:
                    if not "tar" in pl and not "input" in pl:
                        pl = pl.replace("source_for_","") 
                        pl = pl.replace("source_","") 
                        pl = pl.replace("superglue-","") 
                        pl = pl.replace("com","src") 
                        p_labels.append(pl)

            tasks = data_args.task_name
            folder = os.path.join(folder, "img_logs")
            Path(folder).mkdir(parents=True, exist_ok=True)
            title = "-".join(list(score_dict.keys()))
            title = title.strip("-")
            fname = "pred@" + title + "@_" + str(exp_info["expid"]) + ".png"
            fpath = os.path.join(folder, fname)
            
            x_labels = y_labels
            if not square:
                if p_labels: 
                    x_labels = p_labels 
            if add_or_attend_input:
                x_labels.insert(0, "inp")
            img_buf = WBCallback.save_image(
                scores=list(score_dict.values()), 
                cbar=False,
                vmin = vmin,
                vmax = vmax,
                annot=annot,
                y_labels=y_labels,
                x_labels=x_labels,
                mask_zeros = mask_zeros,
                #title = title,
                title = prompts_conf if prompts_conf else spec + " (" + str(num_source_prompts) + ")" # + ("| remove" if "rem" in title else "")
                        #title  
                        #+ " | " + str(kwargs.gen_norm_method) \
                        #+ " | " + str(kwargs.gen_thresh_min) \
                        #+ " | " + spec
            )
            if img_buf:
                im = Image.open(img_buf)
                # im = trim_image(im) 
                im.save(fpath)
                #img_list.append(im)

            #new_im = combine_y(img_list)
            #new_im.save(fpath)

        ##################
        results = {}
        ds_backup = None
        mylogs.bp("gen_conf")
        gnm = ["soft"]
        if "gen_norm_method" in main_vars:
            gnm = kwargs.setdefault("gen_norm_method",["soft"])
            if type(gnm) != list: gnm = [gnm] 
        gcmm = [None]
        if "gen_compose_method" in main_vars:
            gcmm = kwargs.setdefault("gen_compose_method",[None])
            if type(gcmm) != list: gcmm = [gcmm] 
        gen_thresh_min = kwargs.get("gen_thresh_min", [None])
        gen_thresh_max = kwargs.get("gen_thresh_max", [None])
        if type(gen_thresh_min) != list: gen_thresh_min = [gen_thresh_min]
        if type(gen_thresh_max) != list: gen_thresh_max = [gen_thresh_max]
        gen_thresh_min = [float(gg) if gg is not None else gg for gg in gen_thresh_min]
        gen_thresh_max = [float(gg) if gg is not None else gg for gg in gen_thresh_max]
        gen_ntp = kwargs.setdefault("gen_ntp",[num_target_prompts])
        if type(gen_ntp) != list: gen_ntp = [gen_ntp] 
        gen_ntp = [gg for gg in gen_ntp if gg <= num_target_prompts]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        masking_list = []
        mylogs.bp("genm")
        masking_scores = kwargs.setdefault("masking_scores",["m_score"])
        if type(masking_scores) != list: masking_scores = [masking_scores] 
        if "prompt_masking" in main_vars:
            masking_list = kwargs.setdefault("prompt_masking",[])
        if type(masking_list) != list:
            masking_list = [masking_list]
        gen_masks_list = [{"no-mask":None}]
        mask_num_start = 0
        if masking_list:
            gen_masks_list = []
        if adapter_args.prompt_tuning:
            prompt_names = model.encoder.prompt_names
        for masking in masking_list:
            gen_masks = {}
            if not "-" in masking:
                masking = "0-" + masking + "-1"
            nn, mask_type, mm = masking.split("-") 
            ss = 0
            if "_" in nn:
                ss, nn = nn.split("_")
            num_masks, num_masked_prompts, mask_num_start =int(nn), int(mm), int(ss)
            mylogs.bp("nrp")
            gen_masks["no-mask_"+mask_type] = None
            if num_masks == 0: 
                if mask_type == "remove" or mask_type == "keeponly":
                    router = model.encoder.router
                    positive_indices_per_row = [torch.nonzero(row > 0)[:, -1] 
                            for row in router]
                    max_length_index = max(range(len(positive_indices_per_row)), 
                            key=lambda i: len(positive_indices_per_row[i]))
                    # Access the maximum length and indices
                    max_length = len(positive_indices_per_row[max_length_index])
                    num_masks = max_length
                else:
                    num_masks = num_source_prompts
            col = 0
            if add_or_attend_input:
                mylogs.bp("keepinp")
                mask = model.encoder.make_attn_mask(col, 1, mask_type + "_input")
                mkey = str(col) + "-" + mask_type + "-input"
                gen_masks[mkey] = mask
            if num_masked_prompts > 0 and use_source_prompts:
                for rm in range(mask_num_start, mask_num_start + num_masks):
                    col = rm + 1
                    mask = model.encoder.make_attn_mask(col, num_masked_prompts, mask_type)
                    mkey = str(col) + "-" + mask_type + "-" \
                            + prompt_names[col].replace("source_","").replace("-","_")
                    gen_masks[mkey] = mask
            if mask_type: 
                if (use_source_prompts 
                    and not model_args.compose_method in ["mcat","mwavg"]
                    and num_source_prompts > 1):
                    col += 1
                    mask = model.encoder.make_attn_mask(col, 1, mask_type + "_source")
                    mylogs.bp("keepsrc")
                    mkey = str(col) + "-" + mask_type + "-source"
                    gen_masks[mkey] = mask
                if use_private_prompts:
                    col += 1
                    mask = model.encoder.make_attn_mask(col, 1, mask_type + "_private")
                    mkey = str(col) + "-" + mask_type + "-private"
                    gen_masks[mkey] = mask
                if (add_target_prompt 
                    and not model_args.compose_method in ["mcat","mwavg"]):
                    col += 1
                    mask = model.encoder.make_attn_mask(col, 1, mask_type + "_target")
                    mkey = str(col) + "-" + mask_type + "-target" 
                    gen_masks[mkey] = mask
            gen_masks_list.append(gen_masks)

        ii = 0
        kk = 0
        sdf_rows = []
        img_list = []
        sep_eval = kwargs.get("separate_eval", False)
        if sep_eval: 
            exp_folder = Path(training_args.output_dir).parent.parent
            exp_folder = os.path.join(str(exp_folder), exp_folder.stem + "_mask")
        else:
            exp_folder = Path(training_args.output_dir).parent
        exp_folder_name = Path(training_args.output_dir).stem
        exp_folder = str(exp_folder) 
        if not adapter_args.prompt_tuning:
            eval_folder = training_args.output_dir
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                auto_task = auto_tasks[task]
                task = task.split("_")[-1]
                ds_conf = data_args.test_dataset_config_name[idx]
                ds_name = data_args.test_dataset_name[idx]
                ds_name = "none" if not ds_name else ds_name
                ds_conf = "none" if not ds_conf else ds_conf
                is_train = "train" if training_args.do_train else "eval"
                save_to = os.path.join(eval_folder,
                     ds_conf + "_results_" + is_train + "_" + ds_name + \
                     str(kwargs.trial) + "_" + mylogs.now + "_1.tsv")
                df, scores, golds, preds = evaluate_test(task, test_dataset, save_to, 
                        ds_name, auto_task)
        else:
            attend_num =len(model.encoder.prompt_encoders) + 1 # one for input
            gen_combs = itertools.product(gnm, gcmm, 
                    gen_thresh_min, gen_thresh_max, gen_ntp)
            mylogs.bp("genm")
            full_attn_mat = None
            use_masked_attn_scores = -1
            ignore_zeros = False
            if model_args.compose_method == "mcat":
                use_masked_attn_scores = 1
                ignore_zeros = True
            if "use_masked_attn" in main_vars:
                use_masked_attn_scores = kwargs.get("use_masked_attn", use_masked_attn_scores)
            git_def = False 
            if model_args.compose_method == "wavg":
               git_def = True
            gen_ignore_target=kwargs.get("gen_ignore_target", git_def)
            for norm_method, gcmm, gmin, gmax, gntp in gen_combs:
                gen_mask_counter = 0
                for gen_masks in gen_masks_list:
                    task_scores = {}
                    effect_scores = {}
                    eval_folders = {}
                    no_mask_test_files = {}
                    mask_labels = []
                    for rm, mask in gen_masks.items():
                        if "-" in norm_method:
                            norm_method, tmin, tmax = norm_method.split("-")
                            gmin = float(tmin) if tmin != 'none' else None
                            gmax = float(tmax) if tmax != 'none' else None

                        mylogs.bp("pic")
                        rv = "Eval" if not reval else "Reval"
                        test_num = str(data_args.max_test_samples) 
                        if test_num == "-1":
                            test_num = "all"
                        eval_folder_name = rv + "-" + exp_folder_name + "-" + rm \
                                + "-" + kwargs.get("compose_method","cmm") \
                                + "_" + str(gmin) + "-" + str(gmax) + "_" + norm_method \
                                + "_" + str(gcmm) + "-" + str(use_masked_attn_scores) \
                                + "_" + str(kwargs.trial) + "-" + mylogs.now \
                                + "_num-" + test_num + "_" + str(ii) 
                        eval_folder = os.path.join(exp_folder, eval_folder_name)
                        Path(eval_folder).mkdir(parents=True, exist_ok=True)

                        model.encoder.attn_scores = torch.zeros(
                            (attend_num, attend_num), device=device) 
                        model.encoder.attn_mask_learned = torch.zeros(
                            (attend_num, attend_num), device=device) 
                        counter = 0
                        total_score = 0
                        if not rm in task_scores:
                            task_scores[rm] = {}
                        gen_conf = {"rep_penalty":2.0}
                        gen_conf["ignore_zeros"] = kwargs.get("gen_ignore_zeros", ignore_zeros)
                        gen_conf["gen_norm_method"] = norm_method
                        gen_conf["mask_type"] = rm # if mask is not None else "no-mask"
                        gen_conf["gen_thresh_min"] = gmin
                        gen_conf["gen_thresh_max"] = gmax
                        gen_conf["gen_ntp"] = gntp
                        gen_conf["gen_cmm"] = gcmm
                        exp_info["cur_masking"] = rm.split("-")[1]
                        if "keep-source" in gen_conf["mask_type"]:
                           mylogs.bp("keepsrc")
                        elif "keep-" in gen_conf["mask_type"]:
                           mylogs.bp("keepprompt")
                        test_key = ""
                        for kk, vv in gen_conf.items():
                            if kk not in ["attn_mask"]:
                                exp_info[kk] = vv
                                if kk != "mask_type":
                                    test_key += str(vv) + "-"
                        test_key = test_key.strip("-")
                        if not test_key in eval_folders:
                            eval_folders[test_key] = []
                        eval_folders[test_key].append(eval_folder_name)
                        if not test_key in task_scores[rm]:
                            task_scores[rm][test_key] = {}
                        if adapter_args.prompt_tuning:
                            targets = model.encoder.target_encoders_idx
                            y_labels = [model.encoder.prompt_names[i] for i in targets]
                            y_labels = [y.replace("tar-","") for y in y_labels]
                            if mask is not None:
                                mask_matrix = mask.index_select(0, targets)
                            orig_mask = model.encoder.attn_mask_orig 
                            orig_mask_matrix = orig_mask.index_select(0, targets)
                        for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                            auto_task = auto_tasks[task]
                            mylogs.bp("attn_mask")
                            gen_conf["attn_mask"] = model.encoder.attn_mask_orig 
                            if mask is not None: 
                               mylogs.bp("testmask")
                               if not gen_ignore_target: 
                                   tmask = model.encoder.make_attn_mask(1,1,"keep_target")
                                   mask = mask | tmask 
                               if use_masked_attn_scores < 0:
                                   gen_conf["attn_mask"] = mask 
                                   gen_conf["attn_mat"] = None
                               else:
                                   gen_conf["attn_mat"] = full_attn_mat
                                   if use_masked_attn_scores == 0:
                                       gen_conf["attn_mask"] = mask 
                                   else:
                                       masked_attn_scores = mask * full_attn_mat
                                       if use_masked_attn_scores == 2:
                                           masked_attn_scores[masked_attn_scores != 0] = 1
                                       gen_conf["attn_mat"] = masked_attn_scores 

                            ds_conf = task #data_args.test_dataset_config_name[idx]
                            ds_name = task #data_args.test_dataset_name[idx]
                            task = task.split("_")[0]
                            
                            ds_name = "none" if not ds_name else ds_name
                            ds_conf = "none" if not ds_conf else ds_conf
                            is_train = "train" if training_args.do_train else "eval"
                            save_to = os.path.join(eval_folder,
                                ds_conf + "_results_" + is_train + "_" + ds_name + ".tsv")
                            if preview == "test":
                                print(save_to)
                                ii += 1
                                counter += 1
                                continue
                            use_cache = False
                            if adapter_args.prompt_tuning: 
                                if mask is not None: 
                                    mylogs.bp("cache")
                                    task_index = y_labels.index(task)
                                    task_mask = mask_matrix[task_index]
                                    orig_task_mask = orig_mask_matrix[task_index]
                                    if torch.equal(task_mask,orig_task_mask):
                                        use_cache = True
                                        mylogs.minfo("Using cached predictions for " + task)

                            df, scores, golds, preds = evaluate_test(task, test_dataset, 
                                    save_to, ds_name, auto_task, gen_conf, use_cache = use_cache)

                            if mask is None:
                                no_mask_test_files[task] = save_to

                            df["src_path"] = op.join(mylogs.home, data_args.data_path, 
                                                    ds_conf,"test.tsv")
                            mylogs.bp("rouge")
                            # TODO make it general not according to task names
                            if True: #"xAttr" in data_args.task_name: 
                              mscore = masking_scores[0]
                              if masking_scores[0]=="m_score": 
                                 mscore = "mean_rouge" 
                              task_score = scores[mscore]
                            else:
                              task_score = df[masking_scores[0]].mean() 
                            task_scores[rm][test_key][task] = task_score
                            total_score += task_score 
                            da = {}
                            if use_wandb:
                                test_rouge = wandb.run.summary["test_rouge"]
                                #test_bert = wandb.run.summary["test_bert"]
                                num_preds = wandb.run.summary["num_preds"]
                                da["test_rouge"] = test_rouge
                                #da["test_bert"] = test_bert
                                da["num_preds"] = num_preds
                            da["task"] = task
                            da["norm_method"] = norm_method
                            sdf_rows.append(da)
                            ii += 1
                            counter += 1
#################
                        mylogs.bp("pic")
                        if mask is None:
                            full_attn_mat = model.encoder.attn_scores
                        mean_score = total_score / counter
                        if adapter_args.prompt_tuning:
                            targets = model.encoder.target_encoders_idx
                            router_scores = model.encoder.router.index_select(0, targets)
                            tlen = router_scores.size(0)
                            rsim = torch.eye(tlen)
                            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                            for i in range(tlen):
                                for j in range(tlen):
                                    if i != j:
                                        rsim[i][j] = cos(router_scores[i][:], 
                                                router_scores[j][:])
                            vmin = 0 if tlen <=3 else None
                            save_image(eval_folder, model, {"rsim":rsim}, 
                                        annot=True, square=True, vmin=vmin, vmax=1,
                                        spec = norm_method + "-" + str(gmin) \
                                                + "-" + str(gmax) \
                                                + " | {:.2f}".format(mean_score))
                            tlen = router_scores.size(0)
                            if multi_tasking:
                                start = 0 if add_or_attend_input else 1 
                                router_scores = router_scores[:,start:slen+tlen + 1]
                            save_image(eval_folder, model, {"router":router_scores}, 
                                        spec = norm_method + "-" + str(gmin) \
                                                + "-" + str(gmax) \
                                                + " | {:.2f}".format(mean_score))
                            if init_router is not None:
                                init_router_scores= init_router.index_select(0, targets)
                                init_router_scores= init_router_scores[:,start:slen+tlen + 1]
                                save_image(eval_folder, model, 
                                        {"init_router":init_router_scores}, spec=rm)

                            mylogs.bp("nusp")
                            attn_mat = model.encoder.attn_scores
                            if full_attn_mat is None:
                                attn_mat = full_attn_mat
                            scores_matrix = attn_mat.index_select(0,targets)
                            tlen = scores_matrix.size(0)
                            all_len = scores_matrix.size(1)
                            sim = torch.eye(tlen)
                            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                            for i in range(tlen):
                                for j in range(tlen):
                                    if i != j:
                                        sim[i][j] = cos(scores_matrix[i][:], 
                                                scores_matrix[j][:])

                            scores_matrix = scores_matrix.round(decimals=2)
                            square = False
                            if multi_tasking:
                                start = 0 if add_or_attend_input else 1 
                                if model_args.compose_method == "wsp1":
                                    scores_matrix = scores_matrix[:,start:slen + 1]
                                    square = True
                                else:
                                    if add_target_prompt is True:
                                        scores_matrix = scores_matrix[:,start:]
                                    else:
                                        scores_matrix = scores_matrix[:,start:slen+tlen + 1]

                            #if len(torch.nonzero(scores_matrix)) < 1:
                            #    start = slen if add_or_attend_input else slen + 1 
                            #    scores_matrix[:tlen, start: start + tlen +1] = torch.eye(tlen)
                            save_image(eval_folder, model, {"score":scores_matrix}, 
                                        square = square,
                                        spec = norm_method + "-" + str(gmin) \
                                                + "-" + str(gmax) \
                                                + " | {:.2f}".format(mean_score))
                            score_dict= {"sim":sim.round(decimals=2)} #, "rsim": rsim}
                            save_image(eval_folder, model, score_dict, 
                                    annot=True,
                                    square=True,
                                    spec = norm_method + " | {:.2f}".format(mean_score))

                            if mask is not None:
                                mask_matrix = mask.index_select(0, targets)
                            else:
                                learned_mask = model.encoder.attn_mask_learned 
                                if len(torch.nonzero(learned_mask)) < 1:
                                    learned_mask = model.encoder.attn_mask_orig
                                mask_matrix = learned_mask.index_select(0, targets)
                            if rm != "no-mask":
                                mylogs.bp("keepinp")
                            if multi_tasking:
                                start = 0 if add_or_attend_input else 1 
                                if add_target_prompt is True:
                                    mask_matrix = mask_matrix[:,start:]
                                else:
                                    mask_matrix = mask_matrix[:,start:slen+tlen + 1]
                            save_image(eval_folder, model, {"mask": mask_matrix}, spec=rm)
                            mylogs.bp("effect")
                            if "-" in rm  and mask is not None:
                                if not test_key in effect_scores:
                                    effect_scores[test_key] = torch.zeros(
                                        (tlen +1, slen + len(gen_masks) + 1), device=device) 

                                col,mtype,mlabel = rm.split("-")
                                if mlabel in ["source","private","target"]:
                                    mylogs.bp("mlabel")
                                map_label = {"source":"all_src", "target":"private"}
                                mlabel =map_label[mlabel] if mlabel in map_label else mlabel
                                mlabel = mlabel.replace("com","src")
                                mask_labels.append(mlabel)
                                col = int(col)
                                if col > 0 and not add_or_attend_input:
                                    col = col - 1
                                if col == 1:
                                    mylogs.bp("effect")
                                selected_cols_idx = [
                                        torch.nonzero(torch.ones_like(row))[:, -1] 
                                        for row in router_scores]
                                if "pos" in rm:
                                    selected_cols_idx = [torch.nonzero(row > 0)[:, -1] 
                                        for row in router_scores]
                                base_scores = task_scores["no-mask_"+mtype][test_key]
                                mask_scores = task_scores[rm][test_key]
                                for _task, _score in mask_scores.items():
                                    base_score = float(base_scores[_task])
                                    score = float(_score) 
                                    if base_score == 0:
                                        effect = 0
                                    else:
                                        effect = (score- base_score) # / base_score) #*100
                                    if True: #"keeponly" in rm:
                                        if score < 1 and base_score < 1:
                                            score = score*100
                                            base_score = base_score*100
                                        effect = score # if score > 0 else -10
                                    else:
                                        effect = -1*effect
                                        effect = min(effect, 50)
                                    #effect = max(effect, 0)
                                    if effect <= 0:
                                        effect = -10
                                    task_index = y_labels.index(_task) 
                                    if col < len(selected_cols_idx[task_index]):  
                                        col_index = selected_cols_idx[task_index][col]
                                        effect_scores[test_key][task_index, col_index]=effect
                                    # elif not "keeponly" in rm:
                                    #    effect_scores[test_key][task_index, col] = score 
                                    if False: #"target" in rm:
                                        dif = base_score - effect
                                        effect_scores[test_key][task_index, -2] = dif 

                                    effect_scores[test_key][task_index, -1] = base_score 
                                    torch.set_printoptions(threshold=10_000)
                                    print(effect_scores[test_key]) 
                    #### end of for
                    # eeeeeeeeeeeeeeeeee
                    mylogs.bp("effect")
                    spec = str(gen_mask_counter)
                    if gen_mask_counter < len(masking_list):
                        spec = masking_list[gen_mask_counter]
                    map_methods = {"mwavg":"MSUM","wavg":"SSUM","mcat":"MCAT", "pt":"PT"}
                    spec = model_args.compose_method
                    if spec in map_methods:
                        spec = map_methods[spec]
                    mask_labels.append("")
                    mask_labels.append("total")
                    for test_key, effect_score in effect_scores.items(): 
                        scores = effect_scores[test_key]
                        column_means = torch.mean(scores[:-1], dim=0)
                        scores[-1, :] = column_means
                        for eval_folder_name in eval_folders[test_key]:
                            eval_folder = os.path.join(exp_folder, eval_folder_name)
                            save_image(eval_folder, model, 
                            {"effect_" + spec : scores.round(decimals=2)}, 
                            spec= spec,
                            mask_zeros = True,
                            vmin = kwargs.get("vmin", None),
                            p_labels = mask_labels)
                    gen_mask_counter += 1
                
        ########
        sdf = pd.DataFrame(data=sdf_rows)
        if False: #img_list:
            new_im = combine_y(img_list)
            fname = "pred_" + str(exp_info["expid"]) + ".png" 
            if use_wandb:
                wandb.log({fname:wandb.Image(new_im)})
            else:
                new_im.save(os.path.join(training_args.output_dir, "images", fname))


        if model_args.attn_tuning:
            targets = model.encoder.target_encoders_idx
            scores_matrix = model.encoder.attn_scores.index_select(0, targets)
            router_scores = model.encoder.router.index_select(0, targets)
            _tag = kwargs.setdefault("tag",[])
            #if diff_args:
            #    for k,v in diff_args["values_changed"].items():
            #        if not "output_dir" in k and not "expid" in k:

            #           da[k] = v
            _main_vars = main_vars.copy()
            if "task_name" in _main_vars:
                del _main_vars["task_name"]

            global_scores.append(scores_matrix)
            targets = model.encoder.target_encoders_idx
            y_labels = [model.encoder.prompt_names[i] for i in targets]
            y_labels = [y.replace("tar-","") for y in y_labels]
            global_y_labels.extend(y_labels)
            global_x_labels = model.encoder.prompt_names 
            for score in [scores_matrix]: #[router_scores]
                img_buf = WBCallback.save_image(scores=[score], 
                   y_labels=y_labels,
                   x_labels=model.encoder.prompt_names, 
                   title = str(kwargs.expid) + str(_main_vars) \
                            + "_" + model_args.attn_method,
                    img_h=6.5 if multi_tasking else 2.5,
                    df=None) 
                if img_buf and False:
                    cur_img = Image.open(img_buf)
                    #tags_img = tag_to_image(da, get_image=True)
                    #cur_img = combine_x([tags_img, cur_img])
                    cat = Path(kwargs.save_path).parent
                    sp = op.join(cat, "images") 
                    Path(sp).mkdir(exist_ok=True, parents=True)
                    pic = "router_" + str(exp_info["expid"])
                    pp = sp + "/pred_" + pic + ".png"
                    existing_images = glob.glob(op.join(sp, "pred_*.png"))
                    merge_plots = kwargs.setdefault("merge_plots", False)
                    if existing_images and merge_plots:
                        pp = existing_images[0]
                    if Path(pp).is_file():
                        _image = Image.open(pp)
                        cur_img = combine_y([cur_img, _image])
                    cur_img.save(pp)

    if kwargs.setdefault("eval_test", False):
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,
                                       max_length=data_args.test_max_target_length, 
                                       num_beams=data_args.num_beams,
                                       metric_key_prefix="test"
                                       )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

    if model_args.save_prefix_only:
        checkpoints = glob.glob(os.path.join(
            training_args.output_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoints:
            # save models
            if not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                continue
            checkpoint_model = torch.load(os.path.join(
                os.path.join(checkpoint_dir, "pytorch_model.bin")))
            model.load_state_dict(checkpoint_model)
            new_dir = "{}_prompt_only".format(checkpoint_dir)
            os.mkdir(new_dir)
            if adapter_args.prefix_tuning:
                save_prompts(model, output_dir=new_dir, 
                     prefix_dir = prefix_dir,
                     attn_tuning=model_args.attn_tuning,
                     shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)
            if adapter_args.prompt_tuning:
                save_to_prompts_dir = kwargs.setdefault("save_to_prompts_dir", False) 
                mylogs.bp("store")
                if save_to_prompts_dir:
                    Path(op.join(new_dir, "prompts")).mkdir(parents = True, exist_ok=True)
                    model.store_encoders(output_dir = prompts_dir, prompts_only=True)

            # after saving prompts, we will remove unnecessary checkpoint dir.
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print("Error: %s : %s" % (checkpoint_dir, e.strerror))

    # Evaluate all checkpoints on all tasks if training_args.eval_all_at_last==True
    results = {}
    if training_args.eval_all_at_last:
        mylogs.bp("eval")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            print(checkpoint_dir)
            mylogs.bp("eval")
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)

            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)
            dev_metrics_all = {}
            dev_avg = []
            logger.info("*** Evaluate ***")
            for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
                if idx > 0:
                    print(task)
                    print(eval_metrics)
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                dev_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                dev_avg.append(main_metric)

            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["dev_avg"] = np.mean(dev_avg)
            results[checkpoint_dir]["dev_each"] = dev_metrics_all

        # Test
        logger.info("*** Test ***")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            # load models here
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if (model_args.load_layer_norm is True 
                and "layer_norm_bias.pt" in checkpoint_dir):
                trainer.model.update_layer_norm_weights(checkpoint_dir)
            dpath = os.path.join(checkpoint_dir, router_prefix + "_router.pt")
            if model_args.attn_tuning is True and Path(dpath).is_file():
                trainer.model.update_router(dpath)
            else:
                dpath = os.path.join(prompts_dir, router_prefix + "_router.pt")
                if Path(dpath).is_file():
                    trainer.model.update_router(dpath)

            test_metrics_all = {}
            test_avg = []
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           metric_key_prefix="test"
                                           )
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)
                test_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                test_avg.append(main_metric)
            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["test_avg"] = np.mean(test_avg)
            results[checkpoint_dir]["test_each"] = test_metrics_all
    print(results)
    if use_wandb:
        wandb.finish()
    return results

if __name__ == "__main__":
   cli()
