import logging
from queue import Empty, Queue
from threading import Thread
from typing import List, Dict
from contextlib import contextmanager

import pyarrow.parquet as pq

from .utils import RowGroupInterval, ParquetMetadata


logger = logging.getLogger(__name__)


class Reader:
    def __init__(self, columns, max_preload: int = 1):
        self.columns = columns
        self.max_preload = max_preload
    
    def setup(self, metas: List[ParquetMetadata],  intervals: List[List[RowGroupInterval]]):
        self.metas = metas
        self.intervals = intervals

    @contextmanager
    def _open_parquet_file(self, file_path):
        pf = pq.ParquetFile(file_path)
        try:
            yield pf
        finally:
            pf.close()

    @property
    def table_iterator(self):
        raise NotImplementedError
    


class SyncParquetReader(Reader):
    """Load data synchronously"""

    @property
    def table_iterator(self):
        for itvs in self.intervals:
            with self._open_parquet_file(self.metas[itvs[0].file_index].file_path) as pf:
                for itv in itvs:
                    offset = itv.local_row_end - itv.local_row_start
                    yield pf.read_row_group(itv.row_group_index, self.columns)\
                            .slice(itv.local_row_start, offset)


class AsyncParquetReader(Reader):
    """Load data asynchronously"""

    _END_TOKEN = "_END"
    _DEFAULT_TIMEOUT = 1

    def _preload(
        self, 
        metas: List[ParquetMetadata], 
        intervals: List[List[RowGroupInterval]],
        queue: Queue
    ):
        try:
            for itvs in intervals:
                with self._open_parquet_file(metas[itvs[0].file_index].file_path) as pf:
                    for itv in itvs:
                        offset = itv.local_row_end - itv.local_row_start
                        table = pf.read_row_group(itv.row_group_index, self.columns)\
                                  .slice(itv.local_row_start, offset)
                        queue.put(table, block=True)
        except Exception as e:
            queue.put(e)
        finally:
            queue.put(self._END_TOKEN)

    @property
    def table_iterator(self):
        queue = Queue(self.max_preload)
        preload_thread = Thread(
            target=self._preload, 
            args=(self.metas, self.intervals, queue), 
            daemon=True
        )
        preload_thread.start()

        try:
            while True:
                try:
                    item = queue.get(timeout=self._DEFAULT_TIMEOUT)
                    if item is self._END_TOKEN:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
                except Empty:
                    continue
        finally:
            preload_thread.join()