from typing import Any, List, Optional, TypeVar
import numpy as np
from datasets import concatenate_datasets

def my_interleave_datasets(
        datasets: List["Dataset"],
        probabilities: Optional[List[float]] = None,
        batch_size: Optional[int] = 1,
        oversampling: Optional[bool] = True,
        seed: Optional[int] = None,
        info: Optional[Any] = None,
        split: Optional[Any] = None,
        **kwargs,
        ):
    if not all([dset.features.type == datasets[0].features.type for dset in datasets]):
        raise ValueError("Features must match for all datasets")

    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = concatenate_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    def iter_indices():
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if probabilities is None:
            start = 0 
            while True:
                yield start % len(datasets) 
                start += 1
        else:
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))

    current_index = [0] * len(datasets)
    is_exhausted = np.full(len(lengths), False)
    indices = []
    bool_strategy_func = np.all if oversampling else np.any
    for source_idx in iter_indices():
        if bool_strategy_func(is_exhausted):
            break
        # let's add the example at the current index of the `source_idx`-th dataset
        ds_index = current_index[source_idx] 
        ds_offset = offsets[source_idx]
        ds_end = ds_offset + lengths[source_idx]
        range_start = ds_offset + ds_index
        range_end = min(range_start + batch_size, ds_end)
        ds_sel_range = list(range(range_start, range_end))
        current_index[source_idx] += batch_size 
        if current_index[source_idx] >= lengths[source_idx]:
            current_index[source_idx] = 0
            is_exhausted[source_idx] = True
        if len(ds_sel_range) < batch_size:
            continue
        indices.extend(ds_sel_range)
    return concatenated_datasets.select(indices, **kwargs)

if __name__ == "__main__":
    from datasets import Dataset
    import debugpy
    port = 1234
    debugpy.listen(('0.0.0.0', int(port)))
    print("Waiting for client at run...port:", port)
    #debugpy.wait_for_client()  # blocks execution until client is attached
    d1 = Dataset.from_dict({"a": [0, 1, 2, 3]})
    d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
    d3 = Dataset.from_dict({"a": [20, 21, 22, 23, 24, 25]})
    for bs in [2,3]:
        for over_sampling in [True, False]:
            print("Batch size:", bs)
            print("Over sampling:", over_sampling)
            dataset = my_interleave_datasets([d1, d2, d3], 
                    batch_size=bs, oversampling=over_sampling, probabilities=None)
            print(dataset["a"])
            print("------------------")
