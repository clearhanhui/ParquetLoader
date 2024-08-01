import logging
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from torch.utils.data import IterableDataset

from .utils import (
    ParquetMetadata, 
    detect_distributed_env, 
    detect_worker_env
)
from .shuffle import NoShuffler, FullShuffler
from .reader import SyncParquetReader, AsyncParquetReader


logger = logging.getLogger(__name__)


class ParquetDataset(IterableDataset):
    def __init__(
        self, 
        path: str,
        columns: Optional[List[str]] = None,
        async_read: bool = False,
        max_preload: int = 1
    ):  
        self.path = path
        self.columns = columns
        self.async_read = async_read
        self.max_preload = max_preload
        self.reader = AsyncParquetReader(self.columns, self.max_preload) \
                      if async_read and max_preload > 0 else \
                      SyncParquetReader(self.columns, self.max_preload)
        self.world_size, self.global_rank, self.num_nodes = detect_distributed_env()
        self.batch_size = 1


    def __len__(self):
        if not hasattr(self, 'num_rows'):
            self._associate_to_workers()
        return self.num_rows
    
    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle = shuffle
        self.shuffler = FullShuffler() if shuffle else NoShuffler()
    
    def set_drop_last(self, drop_last: bool) -> None:
        self.drop_last = drop_last

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def get_num_batches(self, batch_size=None):
        assert hasattr(self, 'drop_last'), 'call `set_drop_last` before call `get_num_batches`'
        batch_size = batch_size or self.batch_size
        if self.drop_last:
            return len(self) // batch_size 
        else: 
            return (len(self) + batch_size - 1) // batch_size


    def _try_fetch_metadata(self):
        if hasattr(self, 'metas'):
            return 

        _ds = ds.dataset(self.path)
        self.metas = [
            ParquetMetadata(
                file_path=f.path,
                num_rows=f.metadata.num_rows,
                num_row_groups=f.metadata.num_row_groups,
                num_rows_per_row_group=[
                    f.metadata.row_group(i).num_rows 
                    for i in range(f.metadata.num_row_groups)
                ]
            )
            for f in _ds.get_fragments()
        ]
        self.columns = self.columns or _ds.schema.names
        del _ds
    

    def _associate_to_workers(self):
        self._try_fetch_metadata()
        self.num_workers, self.worker_rank = detect_worker_env()
        self.intervals, self.num_rows = self.shuffler.associate_to_workers(
            metas=self.metas, 
            world_size=self.world_size, 
            num_workers=self.num_workers,
            current_rank=self.global_rank,
            current_worker_rank=self.worker_rank,
            batch_size=self.batch_size
        )


    def __iter__(self):
        assert hasattr(self, 'batch_size'), 'call `get_num_batches` before call `__iter__`'
        assert hasattr(self, 'drop_last'), 'call `set_drop_last` before call `__iter__`'
        assert hasattr(self, 'shuffler'), 'call `set_shuffle` before call `__iter__`'
        self._associate_to_workers()
        return self.iter_batch()


    def __getitem__(self, index: int) -> pd.DataFrame:
        logger.warning("call `__getitem__` is inefficient, only for test usage.")
        self._try_fetch_metadata()
        global_index = 0
        for itvs in self.intervals:
            for itv in itvs:
                offset = itv.local_row_end - itv.local_row_start
                global_index += offset
                if global_index > index:
                    f = pq.ParquetFile(self.metas[itv.file_index])
                    table = f.read_row_group(itv.row_group_index)
                    f.close()
                    return table.slice(index - (global_index - offset), 1)\
                        .to_pandas(split_blocks=True, self_destruct=True).to_numpy()


    def iter_batch(self):
        self.reader.setup(self.metas, self.intervals)
        num_rows_need = self.batch_size
        tables = []
        table_left = None

        for table in self.reader.table_iterator:
            tables.append(table)
            num_rows_need -= table.shape[0]
            if num_rows_need > 0:
                continue
            else:
                table_left = pa.concat_tables(tables)
                table_left = self.shuffler.shuffle(table_left)
                while num_rows_need <= 0:
                    batch_data = table_left.slice(0, self.batch_size)\
                                 .to_pandas(split_blocks=True, self_destruct=True).to_numpy()
                    yield batch_data
                    table_left = table_left.slice(self.batch_size)
                    num_rows_need += self.batch_size
                tables = [table_left] # reset tables

        # last batch, it may not be full batch
        if len(tables) > 0:
            table_left = pa.concat_tables(tables)
            if table_left.shape[0] == 0:
                return
            batch_data = table_left.slice(0, self.batch_size)\
                        .to_pandas(split_blocks=True, self_destruct=True).to_numpy()
            yield batch_data

