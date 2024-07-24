import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from torch.utils.data import TensorDataset, DataLoader
from parquet_loader import DistParquetDataset, DistParquetDataLoader
from pyinstrument.profiler import Profiler as time_profiler
from memory_profiler import profile as mem_profile

## config
path = 'synthetic_data'
num_workers = 4
batch_size = 66
prof = time_profiler()

# streaming load
@mem_profile
def streaming_load():
    prof.start()
    dataset = DistParquetDataset(path)
    dataloader = DistParquetDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        print(i, batch.shape)
        pass
    prof.stop()
    print(f'streaming load: {prof.output_text(unicode=True, color=True)}')

## full load
@mem_profile
def full_load():
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
  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:51:22  Samples:  3161
 /_//_/// /_\ / //_// / //_'/ //     Duration: 7.046     CPU time: 2.483
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benckmark.py

7.045 streaming_load  benckmark.py:18
├─ 6.433 _MultiProcessingDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
│     [32 frames hidden]  torch, multiprocessing, <built-in>, s...
│        4.169 read  <built-in>
├─ 0.437 [self]  benckmark.py
└─ 0.122 DistParquetDataset.__init__  parquet_loader/dataset.py:22
   └─ 0.102 <listcomp>  parquet_loader/dataset.py:28


Filename: /home/hanhui/codes/ParquetLoader/benckmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    18    484.6 MiB    484.6 MiB           1   @mem_profile
    19                                         def streaming_load():
    20    484.6 MiB      0.0 MiB           1       prof.start()
    21    564.8 MiB     80.2 MiB           1       dataset = DistParquetDataset(path)
    22    564.8 MiB      0.0 MiB           1       dataloader = DistParquetDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    23    549.9 MiB   -172.1 MiB        1517       for i, batch in enumerate(dataloader):
    24                                                 # print(i, batch[0].shape)
    25    549.8 MiB   -157.3 MiB        1516           pass
    26    549.9 MiB      0.0 MiB           1       prof.stop()
    27    568.0 MiB     18.0 MiB           1       print(f'streaming load: {prof.output_text(unicode=True, color=True)}')


full load: 
  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:51:22  Samples:  5239
 /_//_/// /_\ / //_// / //_'/ //     Duration: 10.177    CPU time: 9.080
/   _/                      v4.6.2

Program: /home/hanhui/codes/ParquetLoader/benckmark.py

10.175 f  memory_profiler.py:759
├─ 7.045 streaming_load  benckmark.py:18
│  ├─ 6.433 _MultiProcessingDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
│  │     [28 frames hidden]  torch, multiprocessing, <built-in>
│  │        4.169 read  <built-in>
│  ├─ 0.437 [self]  benckmark.py
│  └─ 0.122 DistParquetDataset.__init__  parquet_loader/dataset.py:22
│     └─ 0.102 <listcomp>  parquet_loader/dataset.py:28
└─ 3.131 full_load  benckmark.py:30
   ├─ 1.766 _MultiProcessingDataLoaderIter.__next__  torch/utils/data/dataloader.py:625
   │     [22 frames hidden]  torch, multiprocessing, <built-in>
   └─ 1.267 [self]  benckmark.py


Filename: /home/hanhui/codes/ParquetLoader/benckmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    30    568.0 MiB    568.0 MiB           1   @mem_profile
    31                                         def full_load():
    32    568.0 MiB      0.0 MiB           1       prof.start()
    33   1163.9 MiB    595.9 MiB           1       data = torch.from_numpy(ds.dataset(path).to_table().to_pandas().to_numpy())
    34   1163.9 MiB      0.0 MiB           1       dataset = TensorDataset(data)
    35   1163.9 MiB      0.0 MiB           1       dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    36   1166.2 MiB -159278.6 MiB        1517       for i, batch in enumerate(dataloader):
    37                                                 # print(i, batch[0].shape)
    38   1166.2 MiB -159091.5 MiB        1516           pass
    39    972.9 MiB   -193.3 MiB           1       prof.stop()
    40    989.4 MiB     16.5 MiB           1       print(f'full load: {prof.output_text(unicode=True, color=True)}')
"""