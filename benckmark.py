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
num_workers = 2
batch_size = 66
prof = time_profiler()

# streaming load
@mem_profile
def load_streaming():
    prof.start()
    dataset = DistParquetDataset(path)
    dataloader = DistParquetDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        # print(i, batch[0].shape)
        pass
    prof.stop()
    print(f'full load: {prof.output_text(unicode=True, color=True)}')

## full load
@mem_profile
def load_full():
    prof.start()
    data = torch.from_numpy(ds.dataset(path).to_table().to_pandas().to_numpy())
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        # print(i, batch[0].shape)
        pass
    prof.stop()
    print(f'full load: {prof.output_text(unicode=True, color=True)}')

load_streaming()
load_full()


