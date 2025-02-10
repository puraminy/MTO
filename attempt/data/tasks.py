from collections import OrderedDict
from datasets import Dataset
from datasets import load_from_disk
import collections
import abc
import os
import os.path as op
import pandas as pd
import functools
from pathlib import Path
from typing import Callable, List, Mapping
from utils import pad_punctuation
from metrics import metrics
from .utils import round_stsb_target, defdict
import datasets
import logging
import numpy as np
import torch
import re
import json
import ast
from attempt.maps import *
import attempt.mylogs as mylogs
from itertools import cycle, islice
from random import shuffle
from collections import defaultdict
import random
from tqdm import tqdm
#import nltk
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

super_glue = mylogs.home + "/datasets/super_glue.py"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_numpy(obj):
    """Recursively convert numpy arrays to lists within dictionaries."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def serialize_column(value):
    try:
        # Convert numpy arrays within the value to lists
        value = convert_numpy(value)
        # Try to convert the value to JSON
        json_value = json.dumps(value)
        return json_value
    except (TypeError, OverflowError):
        # If it fails, return the original value
        return value

def deserialize_column(value):
    try:
        # Try to load the value from JSON
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # If it fails, return the original value
        raise ValueError("Type Error")
        return value

class AbstractTask(abc.ABC):
    name = None
    ds_name = None
    do_shuffle = True  # My code
    config = NotImplemented
    prefix = None
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    generation = False
    split_map = None
    use_df = False
    use_processed_df = False
    load_df = False
    cur_example = None
    df_format = ".csv"
    do_split = False
    labels_list = None
    pcounter = 0
    start_row = 0
    post_subsample = False
    rel_nat = None
    rel_nats = {}
    rel_nat_key = None
    rel_vnats = {} # verbalizer version
    rel_vnat = None
    map_labels = True
    labels_list = None
    cache_file = True
    labels_map = {"map": {}}  
    verbalizer = {}
    use_gen_map = False
    general_map = {
            "entailment": "estelzam",
            "not_entailment": "adam",
            "contradiction": "tazad",
            "neutral": "khonsa",
            "duplicate": "tekrar",
            "not_duplicate": "natekrar",
            "equivalent": "barabar",
            "not_equivalent": "namosavi",
            "acceptable": "paziresh",
            "unacceptable": "napazir",
            "positive": "mosbat",
            "negative": "manfi"
        }
    split_folder = {}
    split_prefix = {}
    target_pos = -1
    qpos = "start"
    chpos = "start"
    omit = ""
    template_func = ""
    len_thresh = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = []
    large_data_without_all_splits = []
    records_num = {"train":0, "test":0, "validation":0}
    files = {"train":"", "test":"", "validation":""}
    # small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
    #                                     "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
    #                                     "superglue-boolq", "xsum", "scitail"]
    #large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
    #                                 "amazon-polarity", "yelp-polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "nq", "hotpotqa"]

    def __init__(self, config, task_args, task="", tokenizer=None):
        self.config = config
        mylogs.bp("tinit")
        if config is not None:
            cpath = op.join(mylogs.home, "datasets", self.name, config + ".json")
            if Path(cpath).is_file() and task_args.use_config:
                with open(cpath) as f:
                    targs = json.load(f)
                task_args = {**task_args, **targs}
                task_args = dotdict(task_args)
        self.data_path = task_args.data_path
        self.seed = task_args.data_seed
        self.template = task_args.template
        self.start_row = task_args.get("start_row", 0)
        self.tokenizer = tokenizer
        self.omit = task_args.get("omit_part",self.omit)
        self.qpos = task_args.get("qpos",self.qpos)
        self.chpos = task_args.get("chpos",self.chpos)
        self.len_thresh = task_args.get("len_thresh", self.len_thresh)
        prefix = self.prefix if self.prefix else self.name
        self.prefix = task_args.get("prefix", prefix)
        self.use_cache_file = self.cache_file 
        if self.cache_file:
            self.use_cache_file = task_args.get("use_cache_file", True)
        self.equal_labels = task_args.get("equal_labels", True)
        # list of prompts
        if task:
            self.task_name = task
        if not self.rel_nat:
            self.rel_nat = task
        self.rel_tok = "<" + task + ">"
        self.rel_word = task
        self.prompt_set = {}
        prompt_config = {}
        self.mapping = task_args.mapping
        if self.labels_list is not None and self.map_labels is True:
            self.labels_map["distinct"] = {}
            for i, label in enumerate(self.labels_list):
                self.labels_map["distinct"][label] = self.name + str(i)

        if not self.mapping in self.labels_map and self.map_labels:
            self.mapping = "map"
        prompt_config["length"] = task_args.prompt_length
        prompt_config["target_length"] = task_args.target_prompt_length
        prompt_config["fixed_length"] = task_args.fixed_lenght_prompt
        self.map_labels = self.map_labels and task_args.map_labels
        self.multi_choice = task_args.multi_choice
        self.map_style = task_args.map_style
        if self.map_style == "gen" and self.use_gen_map is True:
            if self.mapping in self.labels_map:
                for kk, vv in self.labels_map[self.mapping].items():
                    if vv in self.general_map:
                        self.labels_map[self.mapping][kk] = self.general_map[vv]

        self.prompt_config = prompt_config
        self.task_args = task_args
        self.counter = {}  # counter for logging items

    def get_id(self):
        return self.prefix

    def after_scoring(self, df, preds, golds):
        print("After Prediction")
        return

    def get_max_target_length(self, tokenizer, default_max_length):
        ll = []
        if self.labels_list is not None:
            for label in self.labels_list:
                if self.mapping in self.labels_map and self.labels_map[self.mapping]:
                    label = self.labels_map[self.mapping][label]
                ll.append(len(tokenizer.encode(label)))
            return max(ll) + 5
        return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs < 0 or (n_obs is not None and n_obs > total_size):
            n_obs = min(total_size, 5000)
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        if not self.do_shuffle:
            num_samples = len(dataset)
            return range(num_samples)
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def sample_equally_from_labels(self, dataset, n_obs):
        # Step 1: Count label distribution in the dataset
        label_counts = defaultdict(list)
        all_indices = self.shuffled_indices(dataset) 
        ii = 0
        for idx in all_indices:
            sample = dataset[idx]
            # Replace 'label' with your actual label field
            label = sample['label']
            if type(label) == float: 
                label = round(label)
            label_counts[label].append(idx)
            ii += 1
            if ii > 500 and n_obs < 100:
                break

        # Step 2: Determine target number of samples per label
        num_labels = len(label_counts)
        samples_per_label = n_obs // num_labels  # Integer division

        # Step 3: Sample equally from each label
        sampled_indices = []
        for label, indices in label_counts.items():
            if len(indices) <= samples_per_label:
                sampled_indices.extend(indices)
            else:
                sampled_indices.extend(
                    random.sample(indices, samples_per_label))

        rest = n_obs - len(sampled_indices)
        if rest > 0:
            rest_indices = random.sample(all_indices, rest)
            sampled_indices.extend(rest_indices)
        # Shuffle the sampled indices
        random.shuffle(sampled_indices)

        # Select the sampled subset from the dataset
        sampled_dataset = dataset.select(sampled_indices)

        return sampled_dataset

    def subsample(self, dataset, n_obs=None, indices=None):
        """
         Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        mylogs.bp("filter")
        if self.equal_labels and n_obs < 1000:
            ds = self.sample_equally_from_labels(dataset, n_obs)
            return ds
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[self.start_row:n_obs]
        ds = dataset.select(indices)
        return ds

    def get_data_path(self, split):
        path = self.data_path
        if not path.startswith("/"):
            path = op.join(mylogs.home, self.data_path)
        if split in self.split_folder:
            ds_name = self.split_folder[split]
        elif self.ds_name is not None:
            ds_name = self.ds_name
        else:
            ds_name = self.name
        path = op.join(path, ds_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        self.split = split
        split_prefix = ""
        if split in self.split_prefix:
            split_prefix = self.split_prefix[split]
        file_path = op.join(path, split_prefix + split + self.df_format)
        return file_path

    def get_fname(self, split):
        split_prefix = "" 
        if split in self.split_prefix:
            split_prefix = self.split_prefix[split]
        return split_prefix + split

    def get_folder_path(self):
        folder = self.data_path
        ds_name = self.ds_name if self.ds_name else self.name
        if not folder.startswith("/"):
            folder = op.join(mylogs.home, self.data_path, ds_name)
        else:
            folder = op.join(folder, ds_name)
        return folder 

    def get_stored_file(self, split, n_obs):
        extension = ".csv"
        fname = self.get_fname(split)
        directory = self.get_folder_path()
        Path(directory).mkdir(parents=True, exist_ok=True)
        infile = None
        if self.equal_labels and split != "test":
            fname += "_eq"
        obs_str = str(n_obs) if n_obs is not None and n_obs > 0 else "all"
        if split == "train":
            if obs_str != "all":
                outfile = os.path.join(directory,
                                       fname + "_" + str(self.seed) + "_" + obs_str + extension)
            else:
                outfile = os.path.join(
                    directory, fname + "_" + obs_str + extension)
        else:
            outfile = os.path.join(
                directory, fname + "_" + obs_str + extension)

        if Path(outfile).is_file() and self.use_cache_file is True:
            infile = outfile

        self.files[split] = outfile
        return directory, infile, outfile

    def save_dataset(self, dataset, output_filename, directory = "", save_ds=True, save_df=True):
        if isinstance(dataset, pd.DataFrame):
            # Save Pandas DataFrame to CSV
            dataset.to_csv(output_filename, index=False)
            print(f"Dataset saved as CSV: {output_filename}")
        elif isinstance(dataset, Dataset) and self.use_df:
            df = dataset.to_pandas()
            # Detect columns that need serialization
            for column in df.columns:
                # Check if the column contains at least one dictionary
                if all(isinstance(val, dict) for val in df[column]):
                    # Serialize the entire column
                    df[column] = df[column].apply(serialize_column) 
            df.to_csv(output_filename, index=False)
        elif isinstance(dataset, Dataset):
            if save_df is True and not Path(output_filename).is_file():
                df = dataset.to_pandas()
                # Detect columns that need serialization
                for column in df.columns:
                    # Check if the column contains at least one dictionary
                    if any(isinstance(val, dict) for val in df[column]):
                        # Serialize the entire column
                        df[column] = df[column].apply(serialize_column)
                df.to_csv(output_filename, index=False)
            if not directory:
                directory = self.get_folder_path()
                directory = op.join(directory, self.split)
            if save_ds:
                dataset.save_to_disk(directory)
        else:
            raise ValueError("Unsupported dataset type. Cannot save.")

    def load_dataset(self, split, n_obs= None):
        if self.use_df:
            df = self.read_df(split)
            assert len(df) > 0, "data frame is empty for " + \
                       split + " of " + self.name + " " + path
            # df = self.filter(df, split)
            ds = Dataset.from_pandas(df)
            self.df = df
            return ds
        else:
            return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, prefix):
        mylogs.bp("map")
        return dataset.map(functools.partial(self.preprocessor, prefix=prefix),
                           remove_columns=dataset.column_names,
                           load_from_cache_file=False)

    def post_map_filter(self, example):
        return True 

    def pre_map_filter(self, example):
        return True 

    def get_records_num(self, split, n_obs):
        return n_obs

    def postproc_df(self, df):
        # df = df[df.prefix == self.name]
        return df

    def preproc_df(self, df, split):
        # df = df[df.prefix == self.name]
        return df

    def read_df(self, split):
        if split != "train" or self.do_split:
            self.do_shuffle = False
        path = self.get_data_path(split)
        if self.df_format == ".tsv":
            df = pd.read_table(path)
        else:
            df = pd.read_csv(path)
        return df

    def get(self, split, prefix="", n_obs=None, split_validation_test=False, lang=None, file_name=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        self.split = split
        mylogs.bp("get")
        assert type(prefix) != list, prefix 
        print("getting samples ... " + prefix)
        directory, file_name, outfile = self.get_stored_file(split, n_obs)
        split_prefix = "" # self.name + "_"
        if split in self.split_prefix:
            split_prefix = self.split_prefix[split]
        split_file = op.join(directory, split_prefix + split + ".csv")
        print("split_file:", split_file)
        save_ds = False
        do_subsample = True
        do_filter = True
        do_map = True

        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)
            if file_name is not None:
                # dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                df = df.dropna(how='all')
                #df.label = df.label.astype(int)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = self.load_dataset(split=mapped_split)
                indices = self.get_split_indices(
                    split, dataset, validation_size=len(dataset)//2)
                dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            if lang is not None:
                dataset = self.load_dataset(split="train", lang_code=lang)
            if file_name is not None:
                # dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                df = df.dropna(how='all')
                #df.label = df.label.astype(int)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = self.load_dataset(split="train")
                indices = self.get_split_indices(
                    split, dataset, validation_size=1000)
                dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)

            mylogs.bp("get")
            if file_name is not None:  # and split == "test":
                mylogs.minfo("------------- LOADING FROM DataFrame Stored FILE:" + \
                             file_name + " ----------")
                # dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                #df.label = df.label.astype(int)
                df = df.dropna(how='all')
                self.df = df
                dataset = Dataset.from_pandas(df)
                do_subsample = False
                do_filter = False
                do_map = True
                save_ds = False
            elif self.use_cache_file and not self.load_df and split_file is not None and Path(split_file).is_file(): 
                mylogs.minfo("------------- LOADING FROM Saved DataFrame Split FILE:" + \
                             split_file + " ----------")
                # dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(split_file)
                #df.label = df.label.astype(int)
                df = df.dropna(how='all')
                for column in df.columns:
                    # Check if the column contains at least one JSON string
                    if all(isinstance(val, str) and val.startswith('{') for val in df[column]):
                        # Deserialize the entire column
                        df[column] = df[column].apply(deserialize_column)

                dataset = Dataset.from_pandas(df)
            else:
                mylogs.minfo("------------- LOADING Dataset :" + \
                             self.name + " ----------")
                if self.use_df and self.load_df:
                    mylogs.bp("saveds")
                    dataset = self.load_dataset(split=mapped_split)
                    save_ds = True 
                else:
                    directory = self.get_folder_path()
                    directory = op.join(directory, split)
                    # directory = outfile.replace(self.df_format,"") 
                    if Path(directory).exists() and self.use_cache_file:
                        try:
                            dataset = load_from_disk(directory)
                        except FileNotFoundError:
                            dataset = self.load_dataset(split=mapped_split)
                            save_ds = True
                    else:
                        dataset = self.load_dataset(split=mapped_split)
                        save_ds = True

        self.save_dataset(dataset, split_file, directory, save_ds=save_ds)
        self.records_num[split] = len(dataset)
        if do_subsample and n_obs is not None and not self.post_subsample:
            dataset = self.subsample(dataset, n_obs)
        if do_filter is True:
            do_subsample = True
            dataset = dataset.filter(functools.partial(self.pre_map_filter),
                    load_from_cache_file=False)
        dataset = self.map_dataset(dataset, prefix)
        if do_filter is True:
            do_subsample = True
            dataset = dataset.filter(functools.partial(self.post_map_filter),
                    load_from_cache_file=False)
        if do_subsample and n_obs is not None and self.post_subsample:
            dataset = self.subsample(dataset, n_obs)
        if self.use_processed_df and not Path(outfile).is_file():
            self.save_dataset(dataset, outfile, save_ds=False, save_df=True) 
        return dataset

    # my post proc
    def post_process(self, preds, labels):
        _preds, _labels = preds, labels
        if self.labels_map and self.mapping in self.labels_map:
            d = self.labels_map[self.mapping]
            _preds, _labels = [], []
            keys = list(d.keys())
            values = list(d.values())
            for pp in preds:
                if pp in values:
                    _preds.append(keys[values.index(pp)])
                else:
                    _preds.append("-1")
            for ll in labels:
                if ll in values:
                    _labels.append(keys[values.index(ll)])
                else:
                    _labels.append(ll)
        return _preds, _labels

    def get_verbalizer_choice(self, label):
        return random.choice(self.verbalizer[label])

    def fill_verbalizer(self, data, target):
        mylogs.bp("verb")
        label = data["target"].strip()
        source = data["source"].strip()
        vtarget = ""
        vcounter = source + self.rel_vnat 
        target_list = []
        pos = self.target_pos
        random_choice = "none"
        mask_placeholder = ""
        if "[MASK]" in vcounter:
            mask_placeholder = " [MASK] "
            if pos != 100:
                while "[MASK]" in vcounter:
                    vcounter = vcounter.replace("[MASK]","",1)
                    if "[MASK]" in vcounter or pos == 0 or "[M-MASK]" in vcounter:
                        random_choice = self.get_verbalizer_choice(label) 
                        vtarget = " [MASK] " +  random_choice
                        target_list.append(vtarget)
        elif pos != 100:
            while "{ans}" in vcounter:
                random_choice = self.get_verbalizer_choice(label) 
                vtarget = random_choice
                target_list.append(vtarget)
                vcounter = vcounter.replace("{ans}","",1)

        if pos == 10:
            t = mask_placeholder + target 
            target_list.append(t) 
        elif pos > 0:
            t = mask_placeholder + target 
            target_list.insert(pos -1, t) 
        if target_list:
            target = " ".join(target_list)
        return target, random_choice

    def before_scoring(self, df):
        if not "vnat" in self.template:
            return df
        valid_labels = self.labels_map["map"].values() 
        def preserve_choices(text):
            pattern = r'choice\d+'
            matches = re.findall(pattern, text)
            preserved_text = ' '.join(matches)
			
            return preserved_text.lower().strip() 

        def clean_text_and_extract_removed(text):
            words = text.split()
            cleaned_words = [word for word in words if word in valid_labels]
            removed_words = [word for word in words if word not in valid_labels]
            cleaned_text = ' '.join(cleaned_words)
            removed_text = ' '.join(removed_words)
            return cleaned_text, removed_text
        def remove_choice(text):
            # Regular expression to match 'Choice' followed by any number of digits
            pattern = re.compile(r'choice\d*')
            # Substitute the matched patterns with an empty string
            cleaned_text = pattern.sub('', text)
            # Remove any extra whitespace that may result from the removal
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            return cleaned_text.lower().strip()

        def match_text1(row):
            pred_choice = preserve_choices(row["pred_text1"]) 
            cleaned_vpred = remove_choice(str(row['vpred']))
            cleaned_vtarget = remove_choice(str(row['vtarget']))
            match_choice = pred_choice == row['target_text'].strip() 
            match_text = cleaned_vpred.strip() != '' and cleaned_vpred in cleaned_vtarget
            match_text = match_text and len(cleaned_vpred) < len(cleaned_vtarget) + 10
            other_choice_text = not match_text and cleaned_vpred in row["input_text"]
            match_choice = match_choice and not other_choice_text
            return row['target_text'] if (match_choice or match_text) else 'mistake'

        def match_text(row):
            cleaned_vpred = remove_choice(str(row['vpred']))
            cleaned_vtarget = remove_choice(str(row['vtarget']))
            cond = cleaned_vtarget in cleaned_vpred 
            cond = cond and len(cleaned_vpred) < len(cleaned_vtarget) + 10

            return row['target_text'] if cond else 'mistake'

		# Apply the cleaning function to each row in 'pred_text1'
        if True: #self.target_pos == 0:
            df["vpred"] = df["pred_text1"]
            #df["pred_text1"] = df["pred_text1"].apply(lambda x: "positive" if x in self.verbalizer["positive"] else "negative")
            if "v3" in self.template or "tr2" in self.template:
                df["pred_text1"] = df.apply(match_text1, axis=1) 
            else:
                df["pred_text1"] = df.apply(match_text, axis=1) 
        else:
            df[['pred_text1', 'vpred']] = df.apply(lambda row:  clean_text_and_extract_removed(row['pred_text1']), axis=1, result_type='expand')

        # Filter out rows that do not have 'positive' or 'negative' after cleaning
        # filtered_df = df[df["pred_text1"].isin(valid_labels)]

        return df

    # my template functions
    def fill_prompt(self, template, name, place_holder, plen= 0, num_holder="_i"):
        _pholder = place_holder
        place_holder = place_holder.replace("task", self.get_id())
        place_holder = place_holder.replace("[", "<")
        place_holder = place_holder.replace("]", ">")
        while _pholder in template:
            if num_holder in _pholder:
                prompt = ""
                start = 0
                if num_holder == "_i":
                    start = self.pcounter
                for i in range(start, start + plen):
                    token = place_holder
                    if num_holder != "_1":
                        token = token.replace(num_holder, "_" + str(i))
                    else:
                        token = token.replace(num_holder, "")
                    prompt += " " + token
                    self.pcounter += 1
            else:
                prompt = place_holder
            prompt = prompt.strip()
            for token in prompt.split():
                if not name in self.prompt_set:
                    self.prompt_set[name] = []
                if not token in self.prompt_set[name]:
                    self.prompt_set[name].append(token)
            template = template.replace(_pholder, prompt, 1)
        return template

    def get_prompt_length(self, pnum= 0, is_target = False):
        mylogs.bp("plen")
        if is_target:
            tlength = self.prompt_config["target_length"]
            if tlength is None:
                return 0
            if type(tlength) == list:
                return tlength[pnum] if pnum < len(tlength) else tlength[-1]
            else:
                return tlength
        plength = self.prompt_config["length"]
        if plength is None:
            return 0
        if type(plength) == list:
            return plength[pnum] if pnum < len(plength) else plength[-1]
        else:
            return plength

    def fill_prompt_regex(self, template, regex):
        m = re.search(regex, template)
        pnum = 0
        self.pcounter = 0
        while m:
            if len(m.groups()) == 2:
                name = m.groups()[0]
                emb = m.groups()[1]
                plen = 1
                if emb.isdigit():
                    plen = int(emb)
                num_holder = "_" + str(plen)
                if emb == "i":
                    plen = self.get_prompt_length(pnum)
                    num_holder = "_i"
                elif emb == "j":
                    plen = self.get_prompt_length(pnum)
                    num_holder = "_j"
                elif emb == "k":
                    plen = self.get_prompt_length(pnum, is_target=True)
                    num_holder = "_k"
                place_holder = "[" + name + "_" + emb + "]"
                if "task" in name:
                    tid = self.get_id()
                    name = name.replace("task", tid)
                template = self.fill_prompt(template, name, place_holder, plen=plen,
                                            num_holder=num_holder)
                m = re.search(regex, template)
                pnum += 1
        return template

    def insert_prompts(self, template):
        mylogs.bp("fill_prompt")
        template = self.fill_prompt_regex(template, "\[([@a-zA-Z-]+)_(\d+)\]")
        template = self.fill_prompt_regex(
            template, "\[([@a-zA-Z\d-]+)_([a-zA-Z\?\d]+)\]")
        return template

    def get_prompts(self):
        data = {"task": self.get_id()}
        self.fill_template(data, {})
        return self.prompt_set

    def temp_mixed(self, template_type = None):
        template = self.template
        if template_type is None: 
            return template
        if template_type == "Filling":
            template += "-unsup"
        if template_type == "Mapping":
            template += "-sup"
        return template
            
    def get_template_format(self):
        src = "(prefix) (prompt) (nat_prefix) {source} (prefix) (prompt) (nat) (prompt) (mask)"
        target = "(mask) (prefix) (nat) {target}"  # {end}"
        return src, target

    def get_template(self, template_type = None):
        src, target = self.get_template_format()
        template = self.template
        mylogs.bp("template")
        if hasattr(self, template):
            template_func = getattr(self, template)
            self.template = template_func(template_type)
        parts = self.template.split("-")
        add_prefix = self.task_args.setdefault("add_prefix", False)
        if not "px" in parts and add_prefix:
            parts.insert(0, "px")
        pcom = 0  # number of shared prompts among all tasks
        for part in parts:
            if part == "mask":
                src = src.replace("(mask)", "[MASK] (mask)")
                target = target.replace("(mask)", "[MASK] (mask)")
            elif part == "unsup":
                src = src.replace("(mask)", "[MASK]")
                target = target.replace("(mask)", "[MASK]")
            elif part == "unsupnat":
                target = target.replace("(mask)", "[MASK]")
            elif part == "sup":
                src = src.replace("(mask)", "")
                target = target.replace("(mask)", "")
                self.qpos = "start"
            elif part == "pcom":
                src = src.replace("(prompt)", "[com_i] (prompt) ", 1)
                pcom += 1
            elif part == "pmask":
                src = src.replace("(prompt)", "[tar-task_k] [MASK] (prompt) ", 1)
            elif part == "ptar":
                src = src.replace("(prompt)", "[tar-task_k] (prompt) ", 1)
            elif part == "p0" or part == "0":
                src = src.replace("(prompt)", "", 1)
            elif part == "px0" or part == "0":
                src = src.replace("(prefix)", "", 1)
            elif part == "px":
                src = src.replace("(prefix)", "{prefix}", 1)
            elif part == "pt":
                src = src.replace("(prompt)", "[task_i] (prompt) ", 1)
            elif part == "pnat":
                src = src.replace("(prompt)", "{prompt_from_nat} (prompt) ", 1)
            elif part == "pn":
                src = src.replace("(prompt)", "{prompt_n} (prompt) ", 1)
            elif part == "pnt":
                src = src.replace("(prompt)", "{prompt_nt} (prompt) ", 1)
            elif part == "pnr":
                src = src.replace("(prompt)", "{prompt_nr} (prompt) ", 1)
            elif part == "psh":
                src = src.replace("(prompt)", "{prompt_shared_tokens} (prompt) ", 1)
            elif part == "psht":
                src = src.replace("(prompt)", "{prompt_task_eq_shared} (prompt) ", 1)
            elif part == "pshr":
                src = src.replace("(prompt)", "{prompt_shared_random} (prompt) ", 1)
            elif part == "nat_prefix":
                src = src.replace("(nat_prefix)", "{rel_nat}", 1)
            elif part == "nat_input" or part == "nat":
                src = src.replace("(nat)", "{rel_nat}", 1)
            elif part.startswith("omit"):
                omit = ""
                if "_" in part:
                    _,omit = part.split("_")
                self.omit = omit 
            elif part.startswith("qpos"):
                pos = "end"
                if "_" in part:
                    _,pos = part.split("_")
                self.qpos = pos
            elif part.startswith("vnat"):
                pos = 10
                if "_" in part:
                    _,pos = part.split("_")
                self.target_pos = int(pos)
                src = src.replace("(nat)", "{rel_vnat}", 1)
            elif part == "input_shared_words":
                src = src.replace("(prefix)", "{rel_shared_words}:", 1)
            elif part == "nat_target":
                target = target.replace("(nat)", "{rel_nat}", 1)
            elif part == "target_shared_words":
                target = target.replace("(prefix)", "{rel_shared_words}:", 1)
            elif part.startswith("v"): 
                self.rel_nat_key = part
            elif part.startswith("tr"): 
                pass # for different tirals
            else:
                raise ValueError("Invalid part in template:" + part)

        # remove unused place holders
        src = re.sub(r'\(.*?\)', '',src)
        src = re.sub(' +', ' ', src)
        target = re.sub(r'\(.*?\)', '',target)
        target = re.sub(' +', ' ', target)

        return src, target, pcom

    def extend_data(self, inp_data, pcom=0):
        data = {}
        mylogs.bp("task")
        if "task" in inp_data:
            task = inp_data["task"]
            task = self.name
            data["rel_tok"] = REL_TO_TOKEN[task] if task in REL_TO_TOKEN else self.rel_tok
            data["rel_word"] = REL_TO_WORD[task] if task in REL_TO_WORD else self.rel_word
            if self.rel_vnats and self.rel_nat_key:
                self.rel_vnat = self.rel_vnats[self.rel_nat_key]  
                data["rel_vnat"] = self.rel_vnat 
            elif self.rel_vnat and self.rel_vnat != self.name:
                data["rel_vnat"] = self.rel_vnat  
            if self.rel_nats and self.rel_nat_key:
                data["rel_nat"] = self.rel_nats[self.rel_nat_key]  
            if self.rel_nat and self.rel_nat != self.name:
                data["rel_nat"] = self.rel_nat  
            elif task in REL_TO_PHRASE: 
                data["rel_nat"] = REL_TO_PHRASE[task] 
            rel_from_nat = REL_TO_PHRASE[task] if task in REL_TO_PHRASE else task
            rel_from_nat = rel_from_nat.split()
            num_prompts = self.task_args.setdefault("num_prompts", 1)
            task_comb = self.task_args.setdefault("task_comb", "none")
            tid = self.task_args["id"]
            prompt_n = []
            if task_comb == "none":
                prompt_n = [
                    "[p" + str(tid) + str(i) + "_i]" for i in range(num_prompts)]
            elif task_comb == "comb":
                prompt_n = [
                    "[p" + str(ii) + "0_i]" for ii in range(1, tid + 1)]
                prompt_n.extend(["[p" + str(tid) + "_i]"])

            data["prompt_n"] = " ".join(prompt_n)
            shuffle(prompt_n)
            data["prompt_nr"] = " ".join(prompt_n)

            l = self.get_prompt_length(0)*(len(prompt_n) - pcom)
            prompt_nt = "[task" + "_" + str(l) + "]"
            data["prompt_nt"] = prompt_nt

            prompt_from_nat = ["[task_" + w + "]" for w in rel_from_nat]
            prompt_from_nat_cycle = []
            for i in range(self.get_prompt_length(0)):
                j = i % len(rel_from_nat)
                tok = "[task" + "_" + rel_from_nat[j] + "?" + str(i) + "]"
                prompt_from_nat_cycle.append(tok)
            if self.prompt_config["fixed_length"]:
                data["prompt_from_nat"] = " ".join(prompt_from_nat_cycle)
            else:
                data["prompt_from_nat"] = " ".join(prompt_from_nat)

            if task in REL_TO_SHARED_TOKENS:
                rel_with_shared_tokens = REL_TO_SHARED_TOKENS[task]
            else:
                rel_with_shared_tokens = task
            rel_with_shared_tokens = rel_with_shared_tokens.split()
            data["rel_shared_words"] = " ".join(rel_with_shared_tokens)
            # prompt shr creates same prompts for shared tokens of tasks,
            # the length of prompts
            # is specified with i
            prompt_shared_tokens = [
                "[" + w + "_i]" for w in rel_with_shared_tokens]
            data["prompt_shared_tokens"] = " ".join(prompt_shared_tokens)
            # prompt is the same as prompt sh but the tokens are shuffled
            shuffle(rel_with_shared_tokens)
            prompt_shared_random = [
                "[" + w + "_j]" for w in rel_with_shared_tokens]
            data["prompt_shared_random"] = " ".join(prompt_shared_random)
            # psht is for comparision. it uses task specific prompts with the length
            # of shared prompts concatenated to each other,
            # however prompts for each tasks are distnict
            # it also substract the length of common or shared prompts among all tasks
            l = self.get_prompt_length(0)*(len(rel_with_shared_tokens) - pcom)
            prompt_task_eq_shared = "[task" + "_" + str(l) + "]"
            data["prompt_task_eq_shared"] = prompt_task_eq_shared

        return data

    def replace_mask(self, text):
        # Replace [MASK] with <extra_id_i>
        mask_counter = 0
        mask_placeholder = "[M-MASK]"
        if mask_placeholder in text:
            replacement = f"<extra_id_0>"
            text = text.replace(mask_placeholder, replacement)
            mask_counter = 1
        mask_placeholder = "[MASK]"
        while mask_placeholder in text:
            replacement = f"<extra_id_{mask_counter}>"
            text = text.replace(mask_placeholder, replacement, 1)
            mask_counter += 1
        return text

    def fill_template(self, data, ex_fields):
        mylogs.bp("fill")
        template_type = None
        if "group" in data:
            template_type = data["group"]
        src, tgt, pcom = self.get_template(template_type)

        
        mask = "<extra_id_0>"
        ex_data = self.extend_data(data, pcom=pcom)
        data = {**data, **ex_data}
        if "rel_nat" in ex_data and "{source}" in ex_data["rel_nat"] and "{rel_nat}" in src:
            src = src.replace("{source}.","")
            src = src.replace("{source}","")
            src = src.replace("{rel_nat}", ex_data["rel_nat"])
        if "target" in data:
            if "rel_vnat" in ex_data and "vnat" in self.template and "{rel_vnat}" in src:
                src = src.replace("{rel_vnat}", ex_data["rel_vnat"])
                tgt,vtgt = self.fill_verbalizer(data, tgt)
                ex_fields["vtarget"] = vtgt

        # data["mask"] = mask
        data["end"] = "</s>"
        data["prefix"] = self.name + ":"
        data = defdict(data)
        # fill the templates with data

        # Replace masks in src and tgt
        # src = self.replace_mask(src)
        # src_texts = src_texts.replace("[MASK]", mask)
        ex_fields["query_template"] = src 
        ex_fields["resp_template"] = src 
        src_texts = src.format_map(data)
        tgt_texts = tgt.format_map(data)
        src_texts = self.replace_mask(src_texts)
        tgt_texts = self.replace_mask(tgt_texts)

        src_texts = self.insert_prompts(src_texts)
        ex_fields["query"] = src_texts
        ex_fields["resp"] = tgt_texts 
        return src_texts, tgt_texts, ex_fields

    def get_label_list(self):
        labels_list = []
        if self.labels_map and self.mapping:
            for label in self.labels_list:
                labels_list.append(
                    "<" + self.labels_map[self.mapping][label] + ">")
        return labels_list

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       prefix: str = None,
                       extra_fields={}):
        if prefix and not self.prefix:
            self.prefix = prefix
        if self.prefix:
            prefix = self.prefix
        if not prefix:
            prefix = self.name
        src_prefix = prefix
        src_prefix += ":"
        mylogs.bp("format")
        mylogs.bp(self.split + "frm")
        if (self.map_labels and self.mapping in self.labels_map
                and self.labels_map[self.mapping]):
            labels_list = []
            for label in self.labels_list:
                labels_list.append(self.labels_map[self.mapping][label])

            tt = []
            for label in targets:
                assert label in self.labels_map[self.mapping], self.name + ":" + label \
                    + ":" + str(self.labels_map)
                # tt.append("<" + self.labels_map[label] + ">")
                ans = self.labels_map[self.mapping][label]
                tt.append(ans.strip())
            targets = tt
        else:
            labels_list = self.labels_list

        try:
            orig_src = ' '.join(sources)
        except:
            sources = [t if t is not None else 'none' for t in sources]
            orig_src = ' '.join(sources)
        src = ' '.join(sources)
        tgt = ' '.join(targets)
        src = src.strip()
        tgt = tgt.strip()

        prompt_len = self.get_prompt_length()
        max_input_len = 511 - len(tgt) - prompt_len
        if self.multi_choice:
            max_input_len -= 9  # for options tag
            max_input_len -= sum([len(l) + 1 for l in labels_list])

        if self.multi_choice:
            src = src + " options:" + ",".join(labels_list)

        src = src[:max_input_len]
        group = None if not "group" in extra_fields else extra_fields["group"]

        data = {'source': src,
                'target': tgt,
                "group" : group,
                "prefix" : src_prefix,
                'task': self.get_id(),
                ** extra_fields}
        ex_fields = {}
        ex_fields["event"] = orig_src
        ex_fields["tail"] = tgt
        ex_fields["sel"] = False
        ex_fields["split"] = self.split
        src_text, tgt_text, ex_fields = self.fill_template(data, ex_fields)
        src_text = src_text.strip()
        tgt_text = tgt_text.strip() + " </s>"
        ex_fields["target_text"] = tgt_text
        if not "examples" in self.counter:
            self.counter["examples"] = 1
        if self.counter["examples"] < 5:
            mylogs.vlog.info(
                f"=========== Extra Fields | split={self.split} =========")
            mylogs.vlog.info("%s", ex_fields)
            self.counter["examples"] += 1
        mylogs.bp("format")
        extra_fields = {**extra_fields, **ex_fields}
        return {'source': src_text,
                'target': tgt_text,
                'task': self.name,
                'extra_fields': extra_fields}


class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, prefix):
        answer = pad_punctuation(example['answers']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question[:100],
                  "context:", context[:350]]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, prefix)


class DROP(AbstractTask):
    name = "drop"
    metric = [metrics.squad]

    def load_dataset(self, split):
        if split == "train":
            return datasets.load_dataset("json", field="history_690",
                                         data_files=op.join(
                        mylogs.home, "drop/drop_dataset/drop_dataset_train.json"))
        else:
            return datasets.load_dataset("json", field="history_690",
                                         data_files=op.join(
                        mylogs.home, "drop/drop_dataset/drop_dataset_dev.json"))

    def preprocessor(self, example, prefix):
        answer = pad_punctuation(example['answers_spans']['spans'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['passage'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, prefix)


class QA(AbstractTask):
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    rel_vnats = {
            "v0": "[MASK] is correct",
            "v1": "The answer is [MASK], [MASK] is correct",
            "v3": "[MASK], the correct choice is [MASK]",
            "v33": "[MASK] [MASK]",
            "v4": "[MASK]",
            "v3so": "[MASK], so the correct choice is [MASK]",
            "v2": "[MASK], so [MASK] is correct",
            "vs1": "{ans}, the correct choice is ",
            "vs2": "{ans}",
        }
    rel_nat = "The correct choice is "
    qpos = "end"
    omit = ""
    len_thresh = None
    def temp_len(self, template_type):
        example = self.cur_example
        question =example['question_stem'] if "question_stem" in example else example["question"]
        choices = example["choices"]
        choices = choices['text']
        average_length = sum(len(choice.split(" ")) for choice in choices) / len(choices)
        if not "?" in question:
           template = "vnat_0-vs2"
        elif average_length < 4:
           template = "vnat_0-v4"
        else:
           template = "vnat_0-vs2"
        return template

    def post_map_filter(self, example):
        anslen = len(example['extra_fields']["answer"])
        return self.len_thresh is None or anslen < self.len_thresh

    def seq2seq_format(self, src_texts, tgt_texts, prefix):
        if self.qpos == "end":
            src_texts.append(", " + self.question)
        else:
            src_texts.insert(0, self.question)
        answer = self.get_verbalizer_choice()
        mylogs.bp("qpos")
        extra_fields = {"answer": answer}
        return super().seq2seq_format(src_texts, tgt_texts, prefix, 
                extra_fields=extra_fields)

class OpenBookQA(QA):
    name = "openbook-qa"
    labels_list = ["0", "1", "2", "3"]
    labels_map = {"map": {"0":"choice1", "1":"choice2", "2":"choice3", 
        "3":"choice4"}}
    def load_dataset(self, split):
        return load_dataset("allenai/openbookqa", "additional", split=split)


    def preprocessor(self, example, prefix):
        self.cur_example = example
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3"}
        choices = example["choices"]
        labels = choices["label"]
        self.question = "question:" + example['question_stem'].strip(".")
        if self.chpos == "end":
            src_texts = ["", choices["text"][0], " choice1,", 
                         "", choices["text"][1], " choice2,",
                         "", choices["text"][2], " choice3,", 
                         "", choices["text"][3], " choice4"]
        else:
            src_texts = ["choice1:", choices["text"][0], ", ", 
                         "choice2:", choices["text"][1], ", ",
                         "choice3:", choices["text"][2], ", ", 
                         "choice4", choices["text"][3], ""]
        if self.omit != "fact1":
            src_texts.append(", fact:" + example["fact1"])
        tgt_texts = [label2id[example["answerKey"]]]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)

    def get_verbalizer_choice(self, label=""):
        label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        ansIndex = label2id[self.cur_example["answerKey"]]
        ans = self.cur_example["choices"]["text"][ansIndex] 
        return ans

class PIQA(QA):
    name = "piqa"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {"map": {"0":"choice1", "1":"choice2", "0.0":"choice1", "1.0":"choice2"}}

    def get_verbalizer_choice(self, label=""):
        example = self.cur_example
        opt = int(example["label"]) 
        opt = str(opt)
        options = {"0":"sol1","1":"sol2"}
        ans = options[opt]
        ans = example[ans]
        return ans

    def load_dataset(self, split):
        return datasets.load_dataset('piqa', split=split)
        path = op.join(mylogs.home, "piqa", "final", split + ".csv")
        # return datasets.load_dataset('csv', data_files=path)
        df = pd.read_csv(path)
        #df.label = df.label.astype(int)
        ds = Dataset.from_pandas(df)
        return ds

    def preprocessor(self, example, prefix):
        self.cur_example = example
        self.question = "question:" + example['goal'] 
        src_texts = [
                "choice1:", example["sol1"], 
                "choice2:", example["sol2"]
                ]
        tgt_texts = [str(example["label"])]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)

class CommonsenseQA2(QA):
    name = "commonsense-qa-2"
    labels_list = ["yes", "no"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "test": "validation",
                           "validation": "validation"}
    rel_nat = "is it correct?"

    def load_dataset(self, split):
        path = "/home/ahmad/datasets/commonsense-qa-2/"
        data_files = {
                    "train": path + "train.json",
                    "validation": path + "dev.json",
                    "test": path + "dev.json"
                    }

        # Load the dataset from the jsonl files
        return load_dataset('json', data_files=data_files, split=split)
        # return load_dataset("tasksource/commonsense_qa_2.0")

    def preprocessor(self, example, prefix):
        self.question = "question:" + example["question"]
        src_texts = [question]
        tgt_texts = [str(example["answer"])]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)



class CommonsenseQA(QA):
    name = "commonsense-qa"
    use_df = True
    labels_list = ["0", "1", "2", "3", "4"]
    labels_map = {"map": {"0":"choice1", "1":"choice2", "2":"choice3", 
        "3":"choice4","4":"choice5"}}
    rel_vnats___temp = {
            "v0": "[MASK] is correct",
            "v1": "The answer is [MASK], so [MASK] is correct",
            "v2": "[MASK], so [MASK] is correct",
        }
    split_to_data_split = {"train": "train",
                           "test": "validation",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('commonsense_qa', split=split)

    def get_verbalizer_choice(self, label=""):
        label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        ansIndex = label2id[self.cur_example["answerKey"]]
        ans = self.cur_example["choices"]["text"][ansIndex] 
        return ans

    def preprocessor(self, example, prefix):
        self.cur_example = example
        choices = example["choices"]
        
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
        self.question = "question:" + example['question'];
        if self.chpos == "end":
            src_texts = ["", choices["text"][0], " choice1", 
                     "", choices["text"][1], " choice2,", 
                     "", choices["text"][2], " choice3,",  
                     "", choices["text"][3], " choice4,",  
                     "", choices["text"][4], " choice5"]
        else:
            src_texts = ["choice1:", choices["text"][0], ", ", 
                     "choice2:", choices["text"][1], ", ", 
                     "choice3:", choices["text"][2], ", ",  
                     "choice4:", choices["text"][3], ", ",  
                     "choice5:", choices["text"][4], ""]
        tgt_texts = [label2id[example["answerKey"]]]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)

class MaskedCommonsenseQA(CommonsenseQA):
    name = "masked-csqa"
    ds_name = "commonsense-qa"
    split_prefix = {"train": "masked_", "test":"masked_"}
    use_df = True
    rel_vnats = {
            "v0": "[MASK] is correct",
            "v1": "The answer is [MASK], [MASK] is correct",
            "v3": "the correct choice is [MASK]",
            "v4": " ",
            "v3so": "[MASK], so the correct choice is [MASK]",
            "v2": "[MASK], so [MASK] is correct",
            "vs1": "{ans}, the correct choice is ",
            "vs2": "{ans}",
            "vvs2": "{ans}",
        }

    def preprocessor(self, example, prefix):
        self.cur_example = example
        choices = example["choices"]
        
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
        question = example['new_question']
        if "vs" in self.template:
            question = question.replace("[MASK]","what") 
        self.question = "question:" + question 
        src_texts = ["choice1:", choices["text"][0], 
                     "choice2:", choices["text"][1],
                     "choice3:", choices["text"][2], 
                     "choice4:", choices["text"][3], 
                     "choice5:", choices["text"][4]]
        tgt_texts = [label2id[example["answerKey"]]]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)


class SocialIQA(QA):
    name = "social-i-qa"
    labels_list = ["0", "1", "2"]
    labels_map = {
            "map": {"0":"choice0", "1":"choice1", "2": "choice2"},
            # "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
        }
    metric = [metrics.accuracy]
    rel_nat = "Correct answer is"
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "test": "validation",
                           "validation": "validation"}

    def get_verbalizer_choice(self, label=""):
        example = self.cur_example
        opt = int(example["label"]) - 1
        opt = str(opt)
        options = {"0":"answerA","1":"answerB", "2":"answerC"}
        ans = options[opt]
        ans = example[ans]
        return ans

    def load_dataset(self, split):
        # return load_dataset('social_i_qa.py', split=split)
        return load_dataset(mylogs.home + '/datasets/social-i-qa/social_i_qa.py', split=split)

    def preprocessor(self, example, prefix):
        self.cur_example = example
        self.question = "question:" + example['question']
        src_texts = [
                "context:", example["context"], 
                "|| choice0:", example["answerA"], 
                "|| choice1:", example["answerB"], 
                "|| choice2:", example["answerC"]
                ]
        # opt = str(int(example["label"] - 1))
        opt = int(example["label"]) - 1
        opt = str(opt)
        options = {"0":"A","1":"B", "2":"C"}
        ans = options[opt]
        # tgt_texts = [example[ans]]
        tgt_texts = [opt]
        return super().seq2seq_format(src_texts, tgt_texts, prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map": {"0":"entailment", "1":"neutral"},
            # "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
        }
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split)
        data_files = {"train": "train-00000-of-00001.parquet","test":"test-00000-of-00001.parquet"}
        return datasets.load_dataset("parquet", data_dir="/home/ahmad/datasets/scitail", data_files=data_files, split=split)

        return datasets.load_from_disk("/home/ahmad/datasets/scitail")

    def preprocessor(self, example, prefix):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["premise:", example['sentence1'],
                     "hypothesis:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    # labels_map = {"map":{"0":"unequal","1":"duplicate"}
    labels_map = {
            "map": {"0":"not_equivalent","1":"equivalent"},
            "map1": {"0":"not_duplicate","1":"duplicate"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
        }
    # labels_map = {"map":{"0":"F","1":"G"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MRPC1(MRPC):
    name = "mrpc1"
    split_folder = {"train": "mrpc", "test":"mrpc"}
    labels_map = {
            "map": {"0":"not_equivalent","1":"equivalent"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
        }


class MRPC2(MRPC):
    name = "mrpc2"
    split_folder = {"train": "mrpc", "test":"mrpc"}
    labels_map = {
            "map": {"0":"not_equivalent","1":"equivalent"},
            "map2": {"0":"not_duplicate","1":"duplicate"}
        }


class MRPC3(MRPC):
    name = "mrpc3"
    split_folder = {"train": "mrpc", "test":"mrpc"}
    labels_map = {
            "map": {"0":"not_equivalent","1":"equivalent"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
        }


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    # labels_map = {"map":{"0": "inadmissible", "1":"acceptable"}
    labels_map = {"map": {"0": "unacceptable", "1":"acceptable"}}
    # labels_map = {"map":{"0": "A", "1":"B"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class Sentiment(AbstractTask):
    labels_map = {"map": {"0":"negative", "1":"positive"}}
    use_df = False
    verbalizer2 = {
        "positive": ["good","great"],
        "negative": ["bad", "poor"]
    }
    verbalizer = {
    "positive": ["good", "awesome", "excellent", "fantastic", 
        "great", "amazing", "wonderful", "superb", "outstanding", "impressive", "spectacular"],
    "negative": ["bad", "terrible", "awful", "horrible", 
        "poor", "dreadful", "lousy", "unpleasant", "subpar", "abysmal", "atrocious"],
    "neutral": ["okay", "fine", "average", "mediocre", 
        "acceptable", "standard", "fair", "decent", "satisfactory", "adequate", "uninspiring"]
    }

    rel_vnats = {
            "v1": "It sounds [MASK]",
            "vs1": "{ans}, the opinion is ",
            "v3": "[MASK], the opinion is [MASK]",
            "v2": "It sounds [MASK], the opinion is [MASK]",
            "vs2": "{ans}",
            "vvs2": "{ans}",
        }
    rel_vnat = "My opinion is "
    target_pos = -1 
    rel_nat = "My opinion is "

class IMDB(Sentiment):
    name = "imdb"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "train",
                           "test": "test"}
    labels_map = {"map": {"0":"negative", "1":"positive"}}
    verbalizer2 = {
        "positive": ["good","great"],
        "negative": ["bad", "poor"]
    }
    rel_vnat = "It sounds [MASK], so it's "
    rel_nat = "My opinion is "

    def load_dataset(self, split):
        return datasets.load_dataset('imdb', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)
    

class TweetEval(Sentiment):
    name = "tweet-eval"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map": {"0":"negative", "1":"neutral", "2":"positive"},
        }
    # rel_nat = "The sentiment is"
    rel_vnat = "It sounds [MASK], I feel "
    rel_nat = "It sounds "

    def load_dataset(self, split):
        return datasets.load_dataset('tweet_eval', 'sentiment',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SST2(Sentiment):
    name = "sst2"
    use_gen_map = True
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    verbalizer3 = {
        "positive": ["good","great"],
        "negative": ["bad", "awful"]
    }
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map": {"0":"negative", "1":"positive"},
        }
    # labels_map = {"map":{"0":"bad", "1":"good"}
    # labels_map = {"map":{"0":"L", "1":"M"}
    #rel_nat = "As a result, they feel"
    rel_vnat = "It sounds [MASK]"
    target_pos = -1 
    rel_nat = "It sounds "

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class YelpPolarity(Sentiment):
    name = "yelp-polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}
    # rel_nat = " he feels "
    rel_vnat = "I think they are [MASK]"
    rel_nat = "My opinion is "
    labels_map = {
            "map": {"0":"negative", "1":"positive"},
        }

    def load_dataset(self, split):
        print(split)
        return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class Amazon_Polarity(AbstractTask):
    name = "amazon-polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp-polarity', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class STSB(AbstractTask):
    name = "stsb"
    map_labels = False
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class Atomic(AbstractTask):
    name = "atomic"
    map_labels = False
    metric = [metrics.rouge]
    metric_names = ["rouge"]
    generation = True
    do_shuffle = True
    post_subsample = False
    use_df = True
    load_df = True
    df_format= ".tsv" 
    samples_per_head_per_split = {"train":1, "test":3}
    rels = []
    split_folder = {"train": "atomic", "test":"atomic"}
    start_row = 0

    def __init__(self, config, task_args, task="", tokenizer=None):
        super().__init__(config, task_args, task, tokenizer)
        train_sph = task_args.get("samples_per_head", 1)
        self.samples_per_head_per_split["train"] = train_sph
        if not self.rels:
            if not task_args.rels:
                self.rels = [self.name]
            else:
                self.rels = task_args.rels
        if type(self.rels) != list:
            self.rels = [self.rels]

    def get_records_num(self, split, n_obs):
        return n_obs*self.samples_per_head_per_split["train"]

    def read_df(self, split):
        if split != "train" or self.do_split:
            self.do_shuffle = False
        path = self.get_data_path(split)
        if self.df_format == ".tsv":
            df = pd.read_table(path)
        else:
            df = pd.read_csv(path)
        if "all" in self.rels:
            all_rels = df["prefix"].unique()
            self.rels += list(all_rels)
            self.rels.remove("all")
        if self.do_split or (split == "test" and len(df) < 300):
            path = self.get_data_path("train")
            if self.df_format == ".tsv":
                df = pd.read_table(path)
            else:
                df = pd.read_csv(path)
            if split == "test":
                df = df.tail(300)
            else:
                df = df.head(len(df) - 300)
        return df

    def load_dataset(self, split):
        # df = self.filter(df, split)
        df = self.read_df(split)
        df = self.preproc_df(df, split)
        assert len(df) > 0, "data frame is empty for " + \
                   split + " of " + self.name + " " + path

        # df = self.filter(df, split)
        ds = Dataset.from_pandas(df)
        self.df = df
        return ds

    def temp_flex(self, template_type):
        if self.name in ["ObjectUse", "AtLocation", "MadeUpOf" ,"HasProperty",
                "CapableOf","Desires","NotDesires","xAttr","xReact","oReact","isFilledBy"]:
            return "unsup-nat"
        else:
            return "sup-nat"

    def check_n_obs2(self, n_obs, total_size):
        if n_obs < 0:
            return total_size
        df = self.df
        lst = df['input_text'].value_counts()[:n_obs].index
        out = df[df['input_text'].isin(lst)]
        #m = pd.Series(range(0, n_obs), index=lst)
        #out = df[df['input_text'].isin(lst)].sort_values('input_text', key=lambda x: m[x])
        n_obs = len(out)
        return n_obs

    def subsample(self, dataset, n_obs=None, indices=None):
        mylogs.bp("filter")
        rows = []
        counter = {}
        df = self.df
        samples_per_head = self.samples_per_head_per_split[self.split]
        assert self.start_row < len(df), "start row is more than lenght of dataframe"
        for idx, row in df.iterrows():
            if idx < self.start_row:
                continue
            if row.prefix != self.name:
                continue
            if not row.input_text in counter:
                counter[row.input_text] = 0
            counter[row.input_text] += 1
            if counter[row.input_text] > samples_per_head:
                continue
            rows.append(row.to_dict())
            if len(counter) > n_obs:
                break
        df = pd.DataFrame(data=rows)
        self.df = self.postproc_df(df)
        ds = Dataset.from_pandas(self.df)
        return ds

    def postproc_df(self, df):
        df["rel_nat"] = self.rel_nat
        if self.prefix:
            df["orig_prefix"] = df["prefix"] 
            df["prefix"] = self.prefix
        return df

    def preproc_df(self, df, split):
        mylogs.bp("filter")
        df["freqs"] = df.groupby(['input_text'])[
                                 'input_text'].transform('count')
        # df['px'] = df[['input_text', 'prefix']].groupby(['input_text'])['prefix'].transform(lambda x: ','.join(x))
        df['px_count'] = df[['input_text', 'prefix']].groupby(['input_text'])['prefix'].transform('nunique')
        print("len df:", len(df))
        print("len new df:", len(df))
        sort_by = ["px_count", "freqs","input_text", "prefix"] 
        if "sel" in df:
            sort_by = ["sel", "freqs", "input_text", "prefix"] 
        df = df.sort_values(by=sort_by, ascending=False)
        i = 0
        for idx, row in df.iterrows():
            group = "" if not "group" in row else row["group"]
            text = "{}:{} | {} | {} ".format(
                row.prefix, group, row.input_text, row.target_text)
            mylogs.success(text, log=False)
            i += 1
            if i > 100:
                break
        return df

    def filter(self, df, split):
        cond = ""
        mylogs.bp("filter")
        df = df[~df["target_text"].str.contains('none', na=False) 
                & (df["target_text"].str.len() >= 3)]
        if not "all" in self.rels:
            for val in self.rels:
                cond += f"| (df['prefix'] == '{val}') "
            cond = cond.strip("|")
            if cond:
                df = df[eval(cond)]
        return df

    # ppppppppppppppp
    def preprocessor(self, example, prefix):
        mylogs.bp("task_prep")
        src_texts = [str(example["input_text"])]
        tgt_texts = [str(example["target_text"])]
        extra_fields = {}
        if "group" in example:
            extra_fields["group"] = example["group"]
        extra_fields["event"] = example["input_text"]
        extra_fields["rel"] = example["prefix"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)


class xIntent(Atomic):
    name = "xIntent"
    rel_nat = "they intend"

class isAfter(Atomic):
    name = "isAfter"
    rel_nat = "Something that happens after {source} is"


class isBefore(Atomic):
    name = "isBefore"
    rel_nat = "Something that happens before {source} is"

class HinderedBy(Atomic):
    name = "HinderedBy"
    rel_nat = "is hindered by"

class AtomicRel(Atomic):
    name = "atomic-rels"
    train_groups = {}
    test_groups = {}
    use_rel_nat = False

    def __init__(self, config, task_args, task="", tokenizer=None):
        super().__init__(config, task_args)
        rels = task_args.rels
        if rels is None:
            rels = []
        if type(rels) != list:
            rels = [rels]
        if self.rels:
            rels.extend(self.rels)
        rels = list(set(rels))
        for g,l in self.train_groups.items():
            rels.extend(l)
        for g,l in self.test_groups.items():
            rels.extend(l)
        rels = list(set(rels))
        self.rels = rels

    def get_id(self):
        return "-".join(self.rels)

    def get_fname(self, split):
        fname = split
        if self.rels:
            rels = sorted(self.rels)
            fname = split + "_" + "-".join(rels)
        return fname.strip("_")

    def preproc_df(self, df, split):
        print("len df:", len(df))
        #df = df.groupby(["prefix"]).head(samples_per_rel)
        df = super().preproc_df(df, split)
        print("len new df:", len(df))
        return df 

    def check_n_obs(self, n_obs, total_size):
        return total_size

    def subsample(self, dataset, n_obs=None, indices=None):
        mylogs.bp("subsample")
        rows = []
        counter = {}
        pcounter = {}
        df = self.df
        if self.split == "train":
            groups = self.train_groups
        else:
            groups = self.test_groups
        rels = []
        for g,l in groups.items():
            rels.extend(l)
        if not rels:
            rels = self.rels
        assert len(rels) > 0, "rels is empty"
        assert self.start_row < len(df), "start row is more than lenght of dataframe"
        samples_per_head = self.samples_per_head_per_split[self.split]
        pbar = tqdm(total=n_obs * len(rels), position=0, leave=True) #,dynamic_ncols=True)
        for idx, row in df.iterrows():
            # print(len(rows), row.prefix, pcounter)
            if len(rows) >= n_obs * len(rels):
                break
            if row.prefix not in rels:
                continue
            w = rels.count(row.prefix)
            if not row.prefix in pcounter:
                pcounter[row.prefix] = 0
            if pcounter[row.prefix] < self.start_row:
                pcounter[row.prefix] += 1
                print("skipping", idx)
                continue
            if not row.input_text in counter:
                counter[row.input_text] = 0
            counter[row.input_text] += 1
            if counter[row.input_text] > samples_per_head:
                continue
            pcounter[row.prefix] += 1
            if pcounter[row.prefix] > self.start_row + n_obs * w: 
                continue
            rows.append(row.to_dict())
            pbar.update(1)
        self.df = pd.DataFrame(data=rows)
        ds = Dataset.from_pandas(self.df)
        return ds

    def get_records_num(self, split, n_obs):
        if "all" in self.rels:
            self.read_df("train")
        return n_obs*len(self.rels)*self.samples_per_head_per_split["train"]

    def preprocessor(self, example, prefix):
        group = relation = example["prefix"].strip()
        rel_nat = ""
        if self.use_rel_nat:
            assert relation in REL_TO_PHRASE, "Relation " + relation + " has no natural phrase"
            rel_nat = REL_TO_PHRASE[relation] 
        src_texts = ["head:", str(example["input_text"]), rel_nat,
                     "tail:", str(example["target_text"])]
        if self.split == "train":
            groups = self.train_groups
        else:
            groups = self.test_groups
        if groups:
            for g, members in groups.items():
                if group in members:
                    group = g
                    break

        tgt_texts = [group]
        extra_fields = {}
        extra_fields["event"] = example["input_text"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)

    def after_scoring(self, df, golds, preds):
        mylogs.bp("after_pred")
        n_obs = len(golds)
        file_name = self.files["test"]
        if file_name:
            df = pd.read_csv(file_name)
            for index, (pred, gold) in enumerate(zip(preds, golds)):
                df.at[index, 'group'] = pred

        outfile = file_name            
        df = df[["input_text","prefix","target_text","group"]]
        df.to_csv(outfile, index=False)
     
class FreeCS(Atomic):
    name = "free-cs"
    df_format= ".csv" 
    split_folder = {"train": "free-rels", "test":"free-rels"}
    #split_folder = {"train": "sent", "test":"sent"}
    #split_prefix = {"train": "sup_", "test":"sup_"}
    split_prefix = {"train": "8000_rand_", "test":""}
    #split_prefix = {"train": "opsent_6500_", "test":""}
    def preproc_df(self, df, split):
        return df

    def subsample(self, dataset, n_obs=None, indices=None):
        df = self.df
        df = df.head(n_obs)
        ds = Dataset.from_pandas(df)
        return ds


class FreeRel(AtomicRel):
    name = "free-rels"
    rels = ["free-cs"]
    df_format = ".csv"
    split_folder = {"train": "omcs", "test":"omcs"}
    split_prefix = {"train": "omcs", "test":"16000_"}
    def preproc_df(self, df, split):
        return df

    def subsample(self, dataset, n_obs=None, indices=None):
        df = self.df
        df = df.head(n_obs)
        ds = Dataset.from_pandas(df)
        return ds

class TaskClassifier(AtomicRel):
    name = "task-clf"
    train_groups = {
           "Filling": ["Desires","CapableOf","xReact", "xAttr", 
                   "Causes","AtLocation", "HasProperty", "ObjectUse", "MadeUpOf"],
           "Mapping": ["xIntent","xWant", "xEffect","xNeed", 
               "isAfter", "isBefore", "oWant", "HasSubEvent"]
           }
    test_groups = {
           "Filling": ["NotDesires", "oReact", "Causes", 
               "HasProperty","MadeUpOf","AtLocation", "isFilledBy"],
           "Mapping": ["xReason","oWant", "HinderedBy", "oEffect","isBefore"]
           }

class TaskClassifier2(TaskClassifier):
    name = "task-clf2"
    use_rel_nat = True
    start_row= 500

class Splitter(AtomicRel):
    name = "splitter"
    def preprocessor(self, example, prefix):
        group = relation = example["prefix"].strip()
        rel_nat = ""
        assert relation in REL_TO_PHRASE, "Relation " + relation + " has no natural phrase"
        rel_nat = REL_TO_PHRASE[relation] 
        src_texts = [str(example["input_text"]), rel_nat,
                     str(example["target_text"])]
        if example["target_text"] is None:
            example["target_text"] = "none"

        pos = str(len(example["input_text"])) + ":"
        tgt_texts = [example["target_text"]]
        extra_fields = {}
        extra_fields["event"] = example["input_text"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)

class SplitDS(AbstractTask):
    metric = [metrics.rouge]
    use_df = True
    metric_names = ["rouge"]
    name = None
    load_df = False
    do_shuffle = False

    def after_scoring(self, df, golds, preds):
        mylogs.bp("after_pred")
        rows = []
        rows2 = []
        for pred, gold in zip(preds, golds):
            pred = pred.strip()
            data = {}
            data["input_text"] = gold.replace(pred,"")
            data["target_text"] = pred
            data["prefix"] = "free-cs"
            if pred in gold:
                rows.append(data)
            else:
                rows2.append(data)

        df = pd.DataFrame(data=rows)
        n_obs = len(df)
        directory, file_name, outfile = self.get_stored_file("train", n_obs)
        outfile = outfile.replace(".tsv",".csv")
        df.to_csv(outfile, index=False)

        df = pd.DataFrame(data=rows2)
        n_obs = len(df)
        directory, file_name, outfile = self.get_stored_file("junc", n_obs)
        outfile = outfile.replace(".tsv",".csv")
        df.to_csv(outfile, index=False)

class SplitOMCS(SplitDS):
    name = "omcs"
    def preprocessor(self, example, prefix):
        src_texts = [str(example["text"])]
        tgt_texts = src_texts
        extra_fields = {}
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)

class SplitOpSent(SplitDS):
    name = "opsent"
    split_prefix = {"train": "opsent", "test":"opsent_"}
    def preprocessor(self, example, prefix):
        src_texts = [str(example["text"])]
        tgt_texts = src_texts
        extra_fields = {}
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)

class SplitSent(SplitDS):
    name = "sent"
    split_prefix = {"train": "omcs", "test":"sup_"}
    def preprocessor(self, example, prefix):
        text = str(example["text"])
        src_texts = [text]
        tgt_texts = src_texts
        extra_fields = {}
        return self.seq2seq_format(src_texts, tgt_texts,
                                   prefix, extra_fields=extra_fields)
class Causes(Atomic):
    name = "Causes"
    rel_nat = "Sometimes {source} causes"


class xReason(Atomic):
    name = "xReason"
    rel_nat = "The reason for PersonX doing this is"


class Desires(Atomic):
    do_split = True
    name = "Desires"


class xAttr(Atomic):
    name = "xAttr"

class comet(Atomic):
    name = "comet"

class xNeed(Atomic):
    name = "xNeed"
    rel_nat = "But before {source}, PersonX needed"


class xReact(Atomic):
    name = "xReact"


class oReact(Atomic):
    name = "oReact"


class AtLocation(Atomic):
    name = "AtLocation"
    rel_nat = "you are likely to find {source} in"
    # rel_nat = "is located at"

class ObjectUse(Atomic):
    name = "ObjectUse"
    rel_nat = "is used for"


class Desires(Atomic):
    name = "Desires"
    rel_nat = "wants"

class NotDesires(Atomic):
    name = "NotDesires"
    rel_nat = "does not want"

class MadeUpOf(Atomic):
    name = "MadeUpOf"
    rel_nat = "is made up of"

class CapableOf(Atomic):
    name = "CapableOf"
    rel_nat = "can"

class HasSubEvent(Atomic):
    name = "HasSubEvent"
    rel_nat = "Something you might do while {source} is"

class HasProperty(Atomic):
    name = "HasProperty"
    prefix = "HasProperty"
    rel_nat = " is "


class isFilledBy(Atomic):
    name = "isFilledBy"
    rel_nat = "can be filled by"


class xWant(Atomic):
    name = "xWant"
    rel_nat = "After {source}, PersonX would want"

class oWant(Atomic):
    name = "oWant"
    rel_nat = "as a result of {source}, others would want"



class xEffect(Atomic):
    name = "xEffect"
    rel_nat = "As a result of {source}, PersonX will"


class oEffect(Atomic):
    name = "oEffect"
    rel_nat = "as a result of {source}, others will"


class CommonGen(AbstractTask):
    name = "common-gen"
    metric = [metrics.rouge]
    metric_names = ["rouge"]
    generation = True

    def load_dataset(self, split):
        return load_dataset(mylogs.home + "/datasets/common_gen.py", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["concepts:"] + example["concepts"]
        tgt_texts = [example['target']]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class QQP(AbstractTask):
    name = "qqp"
    use_gen_map = True
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    labels_map = {
            "map1": {"0":"not_equivalent","1":"equivalent"},
            "map": {"0":"not_duplicate","1":"duplicate"},
            "map2": {"0":"not_equal","1":"duplicate"},
            "map3": {"0":"different","1":"duplicate"},
        }
    # labels_map = {"map":{"0":"unequal","1":"duplicate"}
    # labels_map = {"map":{"0":"different","1":"identical"}
    # labels_map = {"map":{"0":"F","1":"G"}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['question1'],
                     "sentence2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    # labels_map = {"map":{"0":"en", "1":"neutral", "2": "contradicts"}
    labels_map = {
            "map": {"0":"entailment", "1":"neutral", "2": "contradiction"},
            # "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
        }
    # labels_map = {"map":{"0":"0", "1":"1", "2": "2"}
    # labels_map = {"map":{"0":"C", "1":"D", "2": "E"}
    rel_nat = "The logical relation between premise and hypothesis is "

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['premise'],
                     "sentence2:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class ParsNLI(AbstractTask):
    name = "parsnli"
    labels_list = ["c", "e", "n"] # "xx"]
    use_df = True
    split_prefix = {"train":"out_"}
    labels_map = {
            "map2": {
                "e":"علت", 
                "n":"خنثی", 
                "c": "تضاد",
               # "xx": "نامشخص",
                },
            # "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
        }
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    # labels_map = {"map":{"e":"en", "n":"neutral", "c": "contradiction"}
    def pre_map_filter(self, example):
        label = example["label"]
        label = label.strip()
        return label in self.labels_list

    def load_dataset(self, split):
        return datasets.load_dataset("persiannlp/parsinlu_entailment", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['sent1'],
                     "hypothesis:", example["sent2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class ParsSent(AbstractTask):
    name = "pars-sent"
    labels_list = ["c", "e", "n"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    # labels_map = {"map":{"e":"en", "n":"neutral", "c": "contradiction"}

    def load_dataset(self, split):
        return datasets.load_dataset("persiannlp/parsinlu_sentiment", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['sent1'],
                     "hypothesis:", example["sent2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map": {"0":"not_equivalent","1":"equivalent"}
        }

    def load_dataset(self, split):
        return datasets.load_dataset("paws", "labeled_final", split=split)
        return datasets.load_dataset(mylogs.home + '/paws/paws.py',
                                     'labeled_final', split=split)
        path = op.join(mylogs.home, "paws", "final", split + ".tsv") 
        df = pd.read_table(path)
        ds = Dataset.from_pandas(df)
        return ds

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2", "-1"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map": {"-1":"neutral", "0":"entailment", "1":"neutral", "2": "contradiction"}
        }

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        label = str(example['label'])
        tgt_texts = [label]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MultiNLI(AbstractTask):
    name = "multinli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map": {"0":"entailment", "1":"neutral", "2": "contradiction"}
        }

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split)
        data_files = {"train": "train-00000-of-00001.parquet"}
        # ,"test":"validation-00000-of-00001.parquet"}
        return datasets.load_dataset("parquet", data_dir="/home/ahmad/datasets/multinli", data_files=data_files, split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    use_gen_map = True
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    #rel_nat = "Can the question be answered by the passage?"
    rel_nat = "The logical relation between sentence and question is "
    labels_map = {"map": {"0":"entailment", "1":"not_entailment"}}
    # labels_map = {"map":{"0":"entails", "1":"irrelated"}
    # labels_map = {"map":{"0":"yes", "1":"no"}
    # labels_map = {"map":{"0":"C", "1":"D"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['question'][:100],
                     "sentence2:", example["sentence"][:350]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class QNLI1(QNLI):
    name = "qnli1"
    split_folder = {"train": "qnli", "test":"qnli"}
    labels_map = {
            "map": {"0":"entailment","1":"not_entailment"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
            "map4": {"0":"equivalent","1":"not_equivalent"}
        }


class QNLI2(QNLI):
    name = "qnli2"
    split_folder = {"train": "qnli", "test":"qnli"}
    labels_map = {
            "map": {"0":"entailment","1":"not_entailment"},
            "map2": {"0":"not_duplicate","1":"duplicate"},
            "map4": {"0":"equivalent","1":"not_equivalent"}
        }


class RTE(AbstractTask):
    name = "rte"
    use_gen_map = False
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map": {"0":"entailment", "1":"not_entailment"},
            # "map2":{"0":"not_duplicate", "1":"duplicate"}
        } # entailment nont_entailment
    # labels_map = {"map":{"0":"C", "1":"D"} # entailment nont_entailment

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class RTE1(RTE):
    name = "rte1"
    split_folder = {"train": "rte", "test":"rte"}
    labels_map = {
            "map": {"0":"entailment","1":"not_entailment"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
        }


class RTE2(RTE):
    name = "rte2"
    split_folder = {"train": "rte", "test":"rte"}
    labels_map = {
            "map": {"0":"entailment","1":"not_entailment"},
        #      "map2":{"0":"not_equal","1":"duplicate"}
        }


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {"map": {"0":"not_entailment", "1":"entailment"}}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    labels_map = {"map": {"0":"no", "1":"yes"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'boolq', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["passage:", example["passage"],
                     "question:", example["question"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map": {"0":"entailment", "1":"not_entailment"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'rte', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]
    labels_map = {"map": {"0":"entailment", "2":"neutral", "1": "contradiction"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'cb', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map": {"0":"Choice1", "1":"Choice2"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'copa', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    map_labels = True
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers]
               # metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1", "accuracy"]
    labels_map = {"map": {"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'multirc', split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        if text is None:
            mylogs.minfo(">>>>>>>>>>>>>>>>>>>>>>>>>>>> None text")
            return 'none'
        else:
            text = re.sub('<br>', ' ', text)
            text = re.sub('<(/)?b>', '', text)
            return text

    def post_process(self, preds, labels):
        return preds, labels

    def preprocessor(self, example, prefix):
        group = example['idx']
        if type(group) == str:
            group = group.replace("'", "\"")
            group = json.loads(group)
        group = group['question']
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix, extra_fields={"group": group})


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map": {"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'wic', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(
    proxies={
            "http": "http://fodev.org:8118",
            "https": "http://fodev.org:8118"
        }
    )


class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map": {"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue,
                                     'wsc.fixed', split=split)
        # , download_config=download_config)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, prefix):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'record', split=split)

    def preprocessor(self, batch, prefix):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            answers = ex["answers"] if num_answers > 0 else ["<unk>"]
            for ans in answers:
                fmt = self.seq2seq_format([inputs], [ans], prefix)
                new_batch["source"].extend([fmt["source"]])
                new_batch["target"].extend([fmt["target"]])
                new_batch["task"].extend([self.name])
                #exf = {**fmt["extra_fields"], **{"answers": ex["answers"]}}
                exf = {"answers": ex["answers"]}
                new_batch["extra_fields"].extend([exf])
        return new_batch

    def map_dataset(self, dataset, prefix):
        return dataset.map(functools.partial(self.preprocessor, prefix=prefix),
                           batched=True, remove_columns=dataset.column_names)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map": {"0":"Choice1", "1":"Choice2"},
            # "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
        }

    def load_dataset(self, split):
        return datasets.load_dataset('/home/ahmad/datasets/winogrande/winogrande.py', "winogrande_xl", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example["sentence"],
                     "Choice1:", example["option1"],
                     "Choice2:", example["option2"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


TASK_MAPPING = OrderedDict(
    [
        ('atomic', Atomic),
        ('isAfter', isAfter),
        ('isBefore', isBefore),
        ('xIntent', xIntent),
        ('xReason', xReason),
        ('Desires', Desires),
        ('Causes', Causes),
        ('xAttr', xAttr),
        ('xNeed', xNeed),
        ('xReact', xReact),
        ('oReact', oReact),
        ('AtLocation', AtLocation),
        ('ObjectUse', ObjectUse),
        ('Desires', Desires),
        ('CapableOf', CapableOf),
        ('HasProperty', HasProperty),
        ('isFilledBy', isFilledBy),
        ('xWant', xWant),
        ('oWant', oWant),
        ('xEffect', xEffect),
        ('oEffect', oEffect),
        ('atomic-rels', AtomicRel),
        ('free-cs', FreeCS),
        ('free-rels', FreeRel),
        ('task-clf', TaskClassifier),
        ('task-clf2', TaskClassifier2),
        ('splitter', Splitter),
        ('omcs', SplitOMCS),
        ('opsent', SplitOpSent),
        ('sent', SplitSent),
        ('squad', Squad),
        ('mrpc', MRPC),
        ('mrpc1', MRPC1),
        ('mrpc2', MRPC2),
        ('mrpc3', MRPC3),
        ('cola', COLA),
        ('sst2', SST2),
        ('tweet-eval', TweetEval),
        ('imdb', IMDB),
        ('qnli', QNLI),
        ('qnli1', QNLI1),
        ('qnli2', QNLI2),
        ('rte', RTE),
        ('rte1', RTE1),
        ('rte2', RTE2),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('parsnli', ParsNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multinli', MultiNLI),
        ('snli', SNLI),
        ('piqa', PIQA),
        ('openbook-qa', OpenBookQA),
        ('obqa', OpenBookQA),
        ('drop', DROP),
        ('newsqa', Squad),
        ('searchqa', Squad),
        ('triviaqa', Squad),
        ('nq', Squad),
        ('hotpotqa', Squad),
        ("social-i-qa", SocialIQA),
        ("siqa", SocialIQA),
        ("commonsense-qa", CommonsenseQA),
        ("csqa", CommonsenseQA),
        ("masked-csqa", MaskedCommonsenseQA),
        ("commonsense-qa-2", CommonsenseQA2),
        ("common-gen", CommonGen),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp-polarity', YelpPolarity),
        ('amazon-polarity', Amazon_Polarity),
        ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, task_args=None, tokenizer=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, task_args, task, tokenizer)
        else:
            try:
                return globals()[task](config, task_args, task, tokenizer)
            except:
                raise ValueError(
                "Unrecognized task {} for AutoTask Model.\n" +
                "Task name should be one of {}.".format(task,
                                            ", ".join(c for c in TASK_MAPPING.keys())
                )
        )
