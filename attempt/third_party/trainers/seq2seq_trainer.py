from packaging import version
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union


from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer
from .trainer import BaseTrainer
from transformers.file_utils import is_datasets_available

# my import
from torch.utils.data import DataLoader
import datasets
import attempt.mylogs as mylogs
import wandb
import numpy as np
import os
import glob
import shutil
from pathlib import Path

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class Seq2SeqTrainer(Seq2SeqTrainer, BaseTrainer):
    def __init__(self, train_dataset_sizes=None, shared=False, multiple_metrics=None, adapter_config=None, shuffle=False, save_checkpoint=False, *args, **kwargs):
        #kwargs.pop("data_info")
        #kwargs.pop("multi_task_compute_metrics")
        #kwargs.pop("evaluation_metrics")
        super().__init__(*args, **kwargs)
        self.adapter_config = adapter_config
        self.multiple_metrics = multiple_metrics
        self.train_dataset_sizes = train_dataset_sizes
        self.shared = shared
        self.shuffle = shuffle
        self.save_checkpoint = save_checkpoint
        self.best_prompt_checkpoint = None
        self.gen_conf = None

    def get_train_dataloader(self):
        if self.shuffle:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            return super().get_train_dataloader()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.save_checkpoint:
            super()._save_checkpoint(model, trial, metrics)
        else:
            # Determine the new best metric / best model checkpoint
            checkpoint_folder = f"checkpoint-{self.state.global_step}_prompt_only"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]
                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.best_prompt_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    checkpoints = glob.glob(os.path.join(
                        self.args.output_dir, "checkpoint-*"))
                    for checkpoint_dir in checkpoints:
                        try:
                            shutil.rmtree(checkpoint_dir)
                        except OSError as e:
                            print("Error: %s : %s" % (checkpoint_dir, e.strerror))

                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    self.state.best_metric = metric_value
                    #self.state.best_model_checkpoint = output_dir
                    self.best_prompt_checkpoint = output_dir
                    wandb.run.summary[f"best_{metric_to_check}"] = metric_value 
                    wandb.run.summary["best_step"] = self.state.global_step 
                    wandb.run.summary["best_epoch"] = self.state.epoch 
                    wandb.run.summary["best_checkpoint"] = checkpoint_folder 
                    
                    print("========== Best Model detected =======")
                    print(output_dir)
                    model.store_encoders(output_dir = output_dir, 
                            save_source_prompts=True)
                    print("======================================")
                else:
                    print("======================================")
                    print("Skipping checkpoint")
                    print("======================================")

    def evaluate(
        self,
        eval_dataset: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._num_beams = num_beams,
        print("=================== Evaluation ==================")
        print("Experiment: ", mylogs.args("exp_number"), "/", mylogs.args("total_exp"))
        print("Tags: ", mylogs.get_tag(as_str=True))
        print("Conf: ", mylogs.args("conf"))
        print("Model: ", mylogs.args("model_name_or_path"))
        print("Train samples: ", mylogs.args("max_train_samples"))
        print("Batch size: ", mylogs.args("per_device_train_batch_size"))
        print("Tasks: ", mylogs.args("task_name"), " minus ", mylogs.args("exclude_tasks"))
        print("Save in: ", mylogs.args("save_path"))
        print("=================================================")
        if eval_dataset is None and self.eval_dataset is None:
            if self.args.do_eval:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")
            else:
                return None
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        self.log_metrics("eval", metrics)
        logger = mylogs.tlog
        logger.info(f"***** metrics *****")
        wandb.log(metrics)
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

        return metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        model.encoder.training = False
        model.encoder.gen_conf = self.gen_conf
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # gen_inputs = inputs.clone().detach()

        gen_kwargs = {
            "max_length":200,
            "min_length":2,
            "do_sample":True, 
            "top_p":0.9, 
            "top_k":10,
            "num_beams":5,
            "temperature": 1.0,
            "num_return_sequences":1, 
            "repetition_penalty":5.5,
            "task": inputs["task"] if "task" in inputs else "all",
        #    "bad_words_ids": bad_words_ids
        }
        gen_kwargs2 = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "task": inputs["task"] if "task" in inputs else "all",
            "repetition_penalty": self.gen_conf["rep_penalty"] if "rep_penalty" in self.gen_conf else None
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if not self.args.prediction_loss_only:
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"])

        loss = None
        if self.args.prediction_loss_only:
            with torch.no_grad():
                if False: # self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if has_labels:
                    if self.label_smoother is not None:
                        loss = self.label_smoother(
                            outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(
                            outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        gen_conf: Optional[dict] = None, # my parameter
        task: Optional[str] = None # my parameter
    ):
        self.gen_conf = gen_conf
        self._max_length = max_length
        self._num_beams = num_beams
        self.task = task
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
