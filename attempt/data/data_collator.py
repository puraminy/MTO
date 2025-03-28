import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq


class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # First let the parent collator handle standard fields
        batch = super().__call__(features)
        
        # Preserve additional fields (like task_ids)
        breakpoint()
        extra_fields = {k: [feature[k] for feature in features] 
                       for k in features[0].keys() 
                       if k not in ['input_ids', 'attention_mask', 'labels']}
        
        # Convert lists to tensors
        for k, v in extra_fields.items():
            if isinstance(v[0], torch.Tensor):
                batch[k] = torch.stack(v)
            elif isinstance(v[0], (int, float)):
                batch[k] = torch.tensor(v)
            else:
                batch[k] = v  # Keep as list for non-tensor fields
                
        return batch
