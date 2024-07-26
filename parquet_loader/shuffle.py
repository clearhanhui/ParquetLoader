import copy
from typing import List, Dict, Tuple

import numpy as np
import pyarrow as pa

from .utils import ParquetMetadata, RowGroupInterval, associate_to_workers


class Shuffler:
    """Base shuffle the data."""
    def __init__(self, seed=0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def associate_to_workers(
        self,
        metas: List[ParquetMetadata],
        world_size: int = 1,
        num_workers: int = 1,
        current_rank: int = 0,
        current_worker_rank: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> Tuple[Dict[int, List[RowGroupInterval]], int]:  
        raise NotImplementedError

    def shuffle(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class NoShuffler(Shuffler):
    """No shuffle."""

    def associate_to_workers(
        self,
        metas: List[ParquetMetadata],
        world_size: int = 1,
        num_workers: int = 1,
        current_rank: int = 0,
        current_worker_rank: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> Tuple[Dict[int, List[RowGroupInterval]], int]:  
        
        return associate_to_workers(
            metas=metas, 
            world_size=world_size, 
            num_workers=num_workers,
            current_rank=current_rank,
            current_worker_rank=current_worker_rank,
            drop_last=drop_last,
            batch_size=batch_size
        )

    def shuffle(self, data: pa.Table) -> pa.Table:
        return data


class FullShuffler(Shuffler):
    """Full shuffle."""
    def associate_to_workers(
        self,
        metas: List[ParquetMetadata],
        world_size: int = 1,
        num_workers: int = 1,
        current_rank: int = 0,
        current_worker_rank: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> Tuple[Dict[int, List[RowGroupInterval]], int]:  
        
        # shuffle files
        metas = copy.deepcopy(metas)
        self.rng.shuffle(metas)

        intervals, num_rows = associate_to_workers(
            metas=metas, 
            world_size=world_size, 
            num_workers=num_workers,
            current_rank=current_rank,
            current_worker_rank=current_worker_rank,
            drop_last=drop_last,
            batch_size=batch_size
        )

        # shuffle row groups
        for fi, itvs in intervals.items():
            self.rng.shuffle(itvs)

        return intervals, num_rows
    

    def shuffle(self, data: pa.Table) -> pa.Table:
        indices = self.rng.permutation(len(data))
        return data.take(indices)

