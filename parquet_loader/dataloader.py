import logging
from typing import List, Callable, Any, Iterable

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

from .dataset import DistParquetDataset


logger = logging.getLogger(__name__)


def squeezed_wrapper(fn):
    # squeeze the output of the wrapped function
    def squeeze_output(outputs):
        if isinstance(outputs, torch.Tensor):
            return torch.squeeze(outputs, 0)
        elif isinstance(outputs, (list, tuple)):
            return [squeeze_output(output) for output in outputs]
        elif isinstance(outputs, dict):
            return {key: squeeze_output(value) for key, value in outputs.items()}
        else:
            return outputs
    
    def wrapper(*args, **kwargs):
        outputs = fn(*args, **kwargs)
        return squeeze_output(outputs)
    
    return wrapper



class DistParquetDataLoader(DataLoader):
    def __init__(
        self, 
        dataset: DistParquetDataset, 
        batch_size: int | None = 1, 
        shuffle: bool | None = None, 
        sampler: Sampler | Iterable | None = None, 
        batch_sampler: Sampler[List] | Iterable[List] | None = None, 
        num_workers: int = 0, 
        collate_fn: Callable[[List], Any] | None = None, 
        pin_memory: bool = False, 
        drop_last: bool = False, 
        timeout: float = 0, 
        worker_init_fn: Callable[[int], None] | None = None, 
        multiprocessing_context=None, 
        generator=None, 
        *, 
        prefetch_factor: int | None = None, 
        persistent_workers: bool = False, 
        pin_memory_device: str = ""
    ):  
        dataset.set_shuffle(shuffle)
        dataset.set_drop_last(drop_last)
        self._num_batches = dataset.get_num_batches(batch_size)
        
        # reset arguments
        shuffle = False
        batch_size = 1
        drop_last = False
        if sampler is not None:
            logger.warning("`sampler` option is not supported in DistParquetDataLoader")
            sampler = None
        if collate_fn is None:
            collate_fn = default_collate
        collate_fn = squeezed_wrapper(collate_fn)

        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers, 
            collate_fn=collate_fn, 
            pin_memory=pin_memory, 
            drop_last=drop_last, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn, 
            multiprocessing_context=multiprocessing_context, 
            generator=generator, 
            prefetch_factor=prefetch_factor, 
            persistent_workers=persistent_workers, 
            pin_memory_device=pin_memory_device
        )

    
    def __len__(self) -> int:
        return self._num_batches
    
