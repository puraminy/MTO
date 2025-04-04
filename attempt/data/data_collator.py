import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq
import torch

class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # First let the parent collator handle standard fields
        batch = super().__call__(features)
        
        # Remove extra fields (excluding the standard ones)
        breakpoint()
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'task_id']}
        
        return batch


