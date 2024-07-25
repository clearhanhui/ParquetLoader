import logging
from typing import List, Callable, Any, Iterable

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _SingleProcessDataLoaderIter as TorchSingleProcessDataLoaderIter,
    _MultiProcessingDataLoaderIter as TorchMultiProcessingDataLoaderIter,
)

from .dataset import ParquetDataset


logger = logging.getLogger(__name__)


def squeeze_first_dim(data):
    if isinstance(data, torch.Tensor):
        return torch.squeeze(data, 0)
    elif isinstance(data, (list, tuple)):
        return [squeeze_first_dim(output) for output in data]
    elif isinstance(data, dict):
        return {key: squeeze_first_dim(value) for key, value in data.items()}
    else:
        return data

class _SingleProcessDataLoaderIter(TorchSingleProcessDataLoaderIter):
    def _next_data(self):
        data = super()._next_data()
        return squeeze_first_dim(data)
    
class _MultiProcessingDataLoaderIter(TorchMultiProcessingDataLoaderIter):
    def _next_data(self):
        data = super()._next_data()
        return squeeze_first_dim(data)



class ParquetDataLoader(DataLoader):
    def __init__(
        self, 
        dataset: ParquetDataset, 
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
            logger.warning("`sampler` option is not supported in ParquetDataLoader")
            sampler = None

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
    
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)