from attempt.third_party.models.t5 import T5LayerNorm
from attempt.adapters import (AutoAdapterConfig, AdapterController, Adapter)
import os
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json
import torch 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import hashlib
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str2int(s: str) -> int:
    # Encode the string to bytes
    encoded_string = s.encode()
    
    # Create an MD5 hash object
    md5_hash = hashlib.md5()
    
    # Update the hash object with the encoded string
    md5_hash.update(encoded_string)
    
    # Get the hexadecimal representation of the hash
    hex_digest = md5_hash.hexdigest()
    
    # Convert the hexadecimal digest to an integer
    unique_integer = int(hex_digest, 16)
    
    return unique_integer

##### My utils
def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def isfloat(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def convert(val):
   if type(val) != str:
       return val 
   ret = val
   if val.lower() == "none": 
       ret= None 
   elif val.lower() == "false":
       ret = False
   elif val.lower() == "true":
       ret= True
   elif isfloat(val):
       if "." in val or "e" in val:
           ret = float(val)
       else:
           ret = int(val)
   elif val.isdigit():
       ret= int(val)
   return ret

def combine_x(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    return new_im

def combine_y(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255) )

    y_offset = 0
    for im in images:
      new_im.paste(im, (0, y_offset))
      y_offset += im.size[1]

    return new_im

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def strval(inp):
   if type(inp) != str:
      return inp
   if inp.startswith("%"): 
      return inp[1:]
   arr = []
   inp = str(inp)
   vals = inp.split("@")
   for val in vals:
       if not val:
           continue
       ret = convert(val)
       arr.append(ret)
   if len(arr) == 1 and not "@" in inp:
       return arr[0]
   return arr

##### My utils end

def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_adapter_config(adapter_args, data_args, training_args, config):
    if (adapter_args.train_task_adapters 
            or adapter_args.prefix_tuning 
            or adapter_args.prompt_tuning 
            or adapter_args.bitfit):
        adapter_config = AutoAdapterConfig.get(
            adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model

        if adapter_args.train_task_adapters:
            data_args.tasks = [data_args.task_name]
            adapter_config.tasks = data_args.tasks
        adapter_params = [field.name for field in fields(adapter_args)]
        for p in adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p) and\
                    getattr(adapter_args, p) is not None:
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(
                    f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
        adapter_config.output_dir = training_args.output_dir
        adapter_config.attn_method = config.attn_method
        adapter_config.attend_target = config.attend_target
        adapter_config.attn_prompt = config.attn_tuning
        adapter_config.learn_attention = config.learn_attention
    else:
        adapter_config = None
    return adapter_config


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_attn_params(model, adapter_args, adapter_config):
    # update attention related weights.
    if adapter_config.attn_method == "dot":
        for n, m in model.named_parameters():
            if "mul_prefix_emb" == n:
                m.requires_grad = True

    elif adapter_config.attn_method == "linear":
        for n, m in model.named_parameters():
            if "encoder.attn_Wa.weight" == n:
                m.requires_grad = True
            if "prefix_shared" == n and adapter_config.attend_target is True:
                m.requires_grad = True

    elif adapter_config.attn_method == "rb":
        for n, m in model.named_parameters():
            if "encoder.router" == n and adapter_config.learn_attention is True:
                m.requires_grad = True

    elif adapter_config.attn_method == "sub":
        for n, m in model.named_parameters():
            if "encoder.attn_W_down.weight" == n and adapter_config.learn_attention is True:
                m.requires_grad = True
            if "encoder.attn_W_up.weight" == n and adapter_config.learn_attention is True:
                m.requires_grad = True
            if "prefix_shared" == n and adapter_config.attend_target is True:
                m.requires_grad = True
    elif adapter_config.attn_method == "constant":
        for n, m in model.named_parameters():
            if "prefix_shared" == n and adapter_config.attend_target is True:
                m.requires_grad = True
    elif adapter_config.attn_method == "concat":
        for n, m in model.named_parameters():
            if "encoder.attn_Wa.weight" == n or "attn_va" == n:
                m.requires_grad = True


def freeze_model_params(model, adapter_args, adapter_config):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # adapter parameters like adapter controllers.
    if adapter_args.train_task_adapters:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

    # Unfreezes last linear layer of decoder.
    if adapter_args.unfreeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreezes layer norms.
    if adapter_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                # this will not consider layer norms inside adapters then.
                if len(name.split(".")) < 7:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    if adapter_args.prefix_tuning:
        freeze_params(model)
        if adapter_config.attn_prompt is False:
            for n, m in model.named_parameters():
                if "prefix_shared" == n:
                    m.requires_grad = True
                # update grad
                if "W_weighting" == n:
                    m.requires_grad = True
        else:
            unfreeze_attn_params(model, adapter_args, adapter_config)

    if adapter_args.prompt_tuning:
        # freeze_params(model)
        if adapter_args.freeze_model is True: 
            for n, m in model.named_parameters():
                if True: #not "prompt_encoders" in n: 
                    m.requires_grad = False
        if adapter_config.attn_prompt is True: 
            unfreeze_attn_params(model, adapter_args, adapter_config)

    ## For bitfit we freeze the whole model except for the biases and the final classifier layer.
    if adapter_args.bitfit:
        freeze_params(model)
        # unfreeze bias terms.
        for n, p in model.named_parameters():
            if ".bias" in n:
                p.requires_grad = True

        # unfreeze the classifier.
        for param in model.lm_head.parameters():
            param.requires_grad = True
        if adapter_args.freeze_bitfit_lm_head:
            for n, param in model.lm_head.named_parameters():
                if "bias" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if adapter_args.freeze_bitfit_lm_head_all:
            for n, param in model.lm_head.named_parameters():
                param.requires_grad = False


def get_adapter_params_names(model):
    """
    Returns adapter related parameters names.
    Args:
      model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, Adapter)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_layer_norm_params_names(model):
    """Returns the layer norms parameters.
    Args:
        model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module,  (T5LayerNorm, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None


def pad_punctuation(text):
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the 
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = re.sub(r'\s+', ' ', text)
    return text


def modify_model_after_init(model, training_args, adapter_args, adapter_config):
    # Freezes model parameters.
    freeze_model_params(model, adapter_args, adapter_config)

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(
        "***** Model Trainable Parameters {} *****".format(trainable_params))
    if training_args.print_num_parameters:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("##### Parameter name %s", name)
        total_lm_head_params = sum(p.numel()
                                   for p in model.lm_head.parameters())
        total_trainable_params = sum(p.numel()
                                     for p in model.parameters() if p.requires_grad)
        total_trainable_bias_params = sum(p.numel(
        ) for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
        total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters(
        ) if p.requires_grad and ".layer_norm.weight" in n)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total trainable bias parameters %s",
                    total_trainable_bias_params)
        logger.info("Total trainable layer norm parameters %s",
                    total_trainable_layernorm_params)
        logger.info("Total parameters %s", total_params)
        t5_base_params = 222882048
        # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
        total_params_ratio = ((total_params-t5_base_params)
                              * 8+t5_base_params)/t5_base_params
        total_trainable_params_percent = (
            total_trainable_params/t5_base_params)*100
        total_trainable_bias_params_percent = (
            total_trainable_bias_params/total_trainable_params)*100
        total_trainable_layernorm_params_percent = (
            total_trainable_layernorm_params/total_trainable_params)*100
        total_trainable_lm_head_params_percent = (
            total_lm_head_params/t5_base_params)*100
        logger.info("For adapters/prompt-tuning, total params %s",
                    total_params_ratio)
        logger.info("For intrinsic, total params %s",
                    total_params/t5_base_params)
        logger.info("Total trainable params %s",
                    total_trainable_params_percent)
        logger.info("Total trainable bias params %s",
                    total_trainable_bias_params_percent)
        logger.info("Total trainable layer norm params %s",
                    total_trainable_layernorm_params_percent)
        logger.info("Total lm_head params %s",
                    total_trainable_lm_head_params_percent)
    return model


def save_json(filepath, dictionary):
    with open(filepath, "w") as outfile:
        json.dump(dictionary, outfile)


def read_json(filepath):
    f = open(filepath,)
    return json.load(f)


def save_training_config(config_file, output_dir):
    json_data = read_json(config_file)
    save_json(os.path.join(output_dir, "training_config.json"), json_data)

def save_prompts(model, output_dir, prefix_dir, 
                 attn_tuning, shared_attn, num_target, task_name):
    for name, param in model.named_parameters():
        # Save prompt weights.
        if attn_tuning is False and ("prefix_shared" in name or "prefix" in name):
            shared_params = param
            torch.save(shared_params, os.path.join(
                output_dir, "prefix_embeddings.pt"))
            if prefix_dir:
                torch.save(shared_params, os.path.join(
                    prefix_dir, "-".join(task_name) + ".pt"))
        elif attn_tuning is True and name == "prefix_shared":
            shared_params = param
            if shared_attn is True:
                for i in range(num_target):
                    torch.save(shared_params[i], os.path.join(
                        output_dir, "prefix_embeddings_{}.pt".format(task_name[i])))
            else:
                torch.save(shared_params, os.path.join(
                    output_dir, "prefix_embeddings.pt"))

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
