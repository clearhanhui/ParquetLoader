"""
This script is used to benchmark the performance of async load and sync load.
To simulate the real-world scenario, we add a delay to the data loading and training process.
"""

import time
from pyinstrument.profiler import Profiler as time_profiler
from parquet_loader import ParquetDataset, ParquetDataLoader
from parquet_loader.reader import AsyncParquetReader, SyncParquetReader

path = 'synthetic_data'
delay_in_seconds = 0.01


class AsyncParquetReaderWithDelays(AsyncParquetReader):
        def _preload(self, metas, intervals, queue):
            try:
                for fi, itvs in intervals.items():
                    with self._open_parquet_file(metas[fi].file_path) as pf:
                        for itv in itvs:
                            offset = itv.local_row_end - itv.local_row_start
                            table = pf.read_row_group(itv.row_group_index).slice(itv.local_row_start, offset)
                            time.sleep(delay_in_seconds)
                            queue.put(table, block=True)
            except Exception as e:
                queue.put(e)
            finally:
                queue.put(self._END_TOKEN)


class SyncParquetReaderWithDelays(SyncParquetReader):
    @property
    def table_iterator(self):
        for fi, itvs in self.intervals.items():
            with self._open_parquet_file(self.metas[fi].file_path) as pf:
                for itv in itvs:
                    offset = itv.local_row_end - itv.local_row_start
                    time.sleep(delay_in_seconds)
                    yield pf.read_row_group(itv.row_group_index).slice(itv.local_row_start, offset)


class ParquetDatasetWithDelays(ParquetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = AsyncParquetReaderWithDelays(self.max_preload) \
                      if self.async_read and self.max_preload > 0 else \
                      SyncParquetReaderWithDelays()


def sync_load():
    prof = time_profiler()
    prof.start()
    dataset = ParquetDatasetWithDelays(path)
    dataloader = ParquetDataLoader(dataset, batch_size=66, shuffle=False)
    for i, batch in enumerate(dataloader):
        time.sleep(delay_in_seconds)
        # print(i, batch.shape)
        pass
    prof.stop()
    print(f'sync load: {prof.output_text(unicode=True, color=True)}')


def async_load():
    prof = time_profiler()
    prof.start()
    dataset = ParquetDatasetWithDelays(path, async_read=True)
    dataloader = ParquetDataLoader(dataset, batch_size=66, shuffle=False)
    for i, batch in enumerate(dataloader):
        time.sleep(delay_in_seconds)
        # print(i, batch.shape)
        pass
    prof.stop()
    print(f'async load: {prof.output_text(unicode=True, color=True)}')


if __name__ == '__main__':
    sync_load()
    async_load()


#######################
### Result on my PC ###
#######################
"""
sync load: 
  _     ._   __/__   _ _  _  _ _/_   Recorded: 03:49:10  Samples:  11961
 /_//_/// /_\ / //_// / //_'/ //     Duration: 39.205    CPU time: 15.765
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benchmarks/benchmark_asyncload.py

39.204 sync_load  benchmark_asyncload.py:50
├─ 22.411 _SingleProcessDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
│     [22 frames hidden]  torch, parquet_loader, pyarrow, pandas
│        21.974 ParquetDatasetWith01sDelay.iter_batch  parquet_loader/dataset.py:107
│        ├─ 13.900 SyncParquetReaderWith01sDelay.table_iterator  benchmark_asyncload.py:32
│        │  ├─ 10.947 sleep  <built-in>
│        │  └─ 2.883 ParquetFile.read_row_group  pyarrow/parquet/core.py:423
└─ 16.692 sleep  <built-in>


async load: 
  _     ._   __/__   _ _  _  _ _/_   Recorded: 03:49:52  Samples:  10615
 /_//_/// /_\ / //_// / //_'/ //     Duration: 25.855    CPU time: 15.253
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benchmarks/benchmark_asyncload.py

25.854 async_load  benchmark_asyncload.py:63
├─ 16.503 sleep  <built-in>
└─ 9.269 _SingleProcessDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
      [31 frames hidden]  torch, parquet_loader, pyarrow, pandas
"""