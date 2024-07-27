"""
This script benchmarks the streaming load and full load of a parquet dataset.
"""

import torch
import pyarrow.dataset as ds
from torch.utils.data import TensorDataset, DataLoader
from parquet_loader import ParquetDataset, ParquetDataLoader
from pyinstrument.profiler import Profiler as time_profiler
from memory_profiler import profile as mem_profile

## config
path = '../synthetic_data'
num_workers = 4
batch_size = 66

# streaming load
@mem_profile
def streaming_load():
    prof = time_profiler()
    prof.start()
    dataset = ParquetDataset(path)
    dataloader = ParquetDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        # print(i, batch.shape)
        pass
    prof.stop()
    print(f'streaming load: {prof.output_text(unicode=True, color=True)}')

## full load
@mem_profile
def full_load():
    prof = time_profiler()
    prof.start()
    data = torch.from_numpy(ds.dataset(path).to_table().to_pandas().to_numpy())
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        # print(i, batch.shape)
        pass
    prof.stop()
    print(f'full load: {prof.output_text(unicode=True, color=True)}')


if __name__ == '__main__':
    streaming_load()
    full_load()




#######################
### Result on my PC ###
#######################
"""
streaming load: 
  _     ._   __/__   _ _  _  _ _/_   Recorded: 22:54:37  Samples:  3309
 /_//_/// /_\ / //_// / //_'/ //     Duration: 7.290     CPU time: 2.587
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benckmark.py

7.290 streaming_load  benckmark.py:17
├─ 6.797 _MultiProcessingDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
│     [2 frames hidden]  torch
│        6.625 _MultiProcessingDataLoaderIter._next_data  parquet_loader/dataloader.py:34
│        ├─ 6.507 _MultiProcessingDataLoaderIter._next_data  torch/utils/data/dataloader.py:1298
│        │     [31 frames hidden]  torch, multiprocessing, <built-in>, s...
│        │        4.395 read  <built-in>
│        └─ 0.110 squeeze_first_dim  parquet_loader/dataloader.py:18
│           └─ 0.103 _VariableFunctionsClass.squeeze  <built-in>
├─ 0.365 [self]  benckmark.py
└─ 0.095 ParquetDataset.__init__  parquet_loader/dataset.py:22
   └─ 0.075 <listcomp>  parquet_loader/dataset.py:28


Filename: /home/hanhui/codes/ParquetLoader/benckmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    17    480.8 MiB    480.8 MiB           1   @mem_profile
    18                                         def streaming_load():
    19    480.8 MiB      0.0 MiB           1       prof = time_profiler()
    20    480.8 MiB      0.0 MiB           1       prof.start()
    21    561.0 MiB     80.2 MiB           1       dataset = ParquetDataset(path)
    22    561.0 MiB      0.0 MiB           1       dataloader = ParquetDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    23    553.4 MiB   -146.1 MiB        1517       for i, batch in enumerate(dataloader):
    24                                                 # print(i, batch.shape)
    25    553.3 MiB   -138.5 MiB        1516           pass
    26    553.4 MiB      0.0 MiB           1       prof.stop()
    27    573.5 MiB     20.0 MiB           1       print(f'streaming load: {prof.output_text(unicode=True, color=True)}')


full load: 
  _     ._   __/__   _ _  _  _ _/_   Recorded: 22:54:47  Samples:  2048
 /_//_/// /_\ / //_// / //_'/ //     Duration: 3.042     CPU time: 6.524
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benckmark.py

3.041 full_load  benckmark.py:30
├─ 1.670 _MultiProcessingDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
│     [52 frames hidden]  torch, multiprocessing, <built-in>, s...
├─ 1.277 [self]  benckmark.py
├─ 0.055 DataLoader.__iter__  torch/utils/data/dataloader.py:425
│     [3 frames hidden]  torch
└─ 0.032 table_to_dataframe  pyarrow/pandas_compat.py:760


Filename: /home/hanhui/codes/ParquetLoader/benckmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    30    573.5 MiB    573.5 MiB           1   @mem_profile
    31                                         def full_load():
    32    573.5 MiB      0.0 MiB           1       prof = time_profiler()
    33    573.5 MiB      0.0 MiB           1       prof.start()
    34   1180.9 MiB    607.4 MiB           1       data = torch.from_numpy(ds.dataset(path).to_table().to_pandas().to_numpy())
    35   1180.9 MiB      0.0 MiB           1       dataset = TensorDataset(data)
    36   1180.9 MiB      0.0 MiB           1       dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    37   1183.3 MiB -137055.7 MiB        1517       for i, batch in enumerate(dataloader):
    38                                                 # print(i, batch.shape)
    39   1183.3 MiB -136873.6 MiB        1516           pass
    40    988.7 MiB   -194.6 MiB           1       prof.stop()
    41    988.8 MiB      0.1 MiB           1       print(f'full load: {prof.output_text(unicode=True, color=True)}')
"""