import os
from typing import Any, List, Optional

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pqds
import torch.distributed as dist
from torch.utils.data import IterableDataset

from .utils import (
    ParquetMetadata, 
    detect_distributed_env, 
    detect_worker_env
)
from .shuffle import NoShuffler, FullShuffler


class DistParquetDataset(IterableDataset):
    def __init__(
        self, 
        path: str,
        column_names: Optional[List[str]] = None,
    ):
        ds = pqds.dataset(path)
        self.metas = [
            ParquetMetadata(
                name=f.path,
                num_rows=f.metadata.num_rows,
                num_row_groups=f.metadata.num_row_groups,
                num_rows_per_row_group=[
                    f.metadata.row_group(i).num_rows 
                    for i in range(f.metadata.num_row_groups)
                ]
            )
            for f in ds.get_fragments()
        ]
        self.column_names = column_names or ds.schema.names
        del ds
        self.world_size, self.global_rank, self.num_nodes = detect_distributed_env()


    def __len__(self):
        if not hasattr(self, 'num_rows'):
            self.num_rows = sum([m.num_rows for m in self.metas])
        return self.num_rows
    
    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle = shuffle
        self.shuffler = FullShuffler() if shuffle else NoShuffler()
    
    def set_drop_last(self, drop_last: bool) -> None:
        self.drop_last = drop_last
    

    def get_num_batches(self, batch_size=1):
        assert hasattr(self, 'drop_last'), 'call `set_drop_last` before call `get_num_batches`'
        self.batch_size = batch_size
        if self.drop_last:
            return len(self) // batch_size 
        else: 
            return (len(self) + batch_size - 1) // batch_size


    def __iter__(self):
        assert hasattr(self, 'batch_size'), 'call `get_num_batches` before call `__iter__`'
        assert hasattr(self, 'drop_last'), 'call `set_drop_last` before call `__iter__`'
        assert hasattr(self, 'shuffler'), 'call `set_shuffle` before call `__iter__`'
        self.num_workers, self.worker_rank = detect_worker_env()
        if not hasattr(self, 'intervals'):
            self.intervals, self.num_rows = self.shuffler.associate_row_groups_to_workers(
                    metas=self.metas, 
                    world_size=self.world_size, 
                    num_workers=self.num_workers,
                    current_rank=self.global_rank,
                    current_worker_rank=self.worker_rank,
                    batch_size=self.batch_size
                )
        return self.iter_batch()


    def __getitem__(self, index: int) -> pd.DataFrame:
        global_index = 0
        for itv in self.intervals:
            global_index += (itv.local_row_end - itv.local_row_start)
            if global_index > index:
                f = pq.ParquetFile(self.metas[itv.file_index])
                table = f.read_row_group(itv.row_group_index)
                f.close()
                return table.slice(index - (global_index -(itv.local_row_end - itv.local_row_start)),1).to_pydict()


    def iter_batch(self):
        num_rows_need = self.batch_size
        tables = []
        table_left = None
        for fi, itvs in self.intervals.items():
            pf = pq.ParquetFile(self.metas[fi].name)
            for itv in itvs:
                table = pf.read_row_group(itv.row_group_index).select(self.column_names)
                offset = itv.local_row_end - itv.local_row_start
                tables.append(table.slice(itv.local_row_start, offset))
                num_rows_need -= offset
                if num_rows_need > 0:
                    continue
                else:
                    while num_rows_need < 0:
                        table_left = pa.concat_tables(tables)
                        batch_data = table_left.slice(0, self.batch_size)\
                                    .to_pandas(split_blocks=True, self_destruct=True).to_numpy()
                        yield batch_data
                        tables = [table_left.slice(self.batch_size)] # reset tables
                        num_rows_need += self.batch_size
            pf.close()

        # last batch, it may not be full batch
        table_left = pa.concat_tables(tables)
        batch_data = table_left.slice(0, self.batch_size)\
                    .to_pandas(split_blocks=True, self_destruct=True).to_numpy()
        yield batch_data

