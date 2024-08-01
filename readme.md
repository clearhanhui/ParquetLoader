# ParquetLoader


This project is inspired by [litdata](https://github.com/Lightning-AI/litdata).
It implements a PyTorch dataset and dataloader that support streaming and distributed loading of [Parquet](https://parquet.apache.org/docs/) datasets.

Key features:

* Streaming loading of large Parquet datasets, e.g., Hive tables stored in Parquet format.
* Near-zero redundancy loading across ranks & workers during distributed training.
* Asynchronous preloading to overlap training and loading for better efficiency.

Limitations:

* Less efficient than full memory loading for small datasets.
* Degrades to full loading (or worse) for datasets with only one or a few Parquet files/row groups.
* Row group size affects efficiency; it's recommended to set it to 1-1000 times the batch size.

## Installation

Install from source

``` shell 
git clone https://github.com/clearhanhui/ParquetLoader.git
cd ParquetLoader
pip install .
```


## Usage

``` python 
from parquet_loader import ParquetDataset, ParquetDataLoader
dataset = ParquetDataset('/path/to/parquet/dataset')
dataloader = ParquetDataLoader(dataset)
```

See examples in [tests](./tests).

## Benchmark

* fullly loading vs streaming loading

  |                   | Time(s) | Memory(MB) |
  | ----------------- | ------- | ---------- |
  | fullly loading    | 3.041   | 153        |
  | streaming loading | 7.290   | 610        |


* synchronous loading vs asynchronous loading

  |                      | Time(s) |
  | -------------------- | ------- |
  | synchronous loading  | 39.204  |
  | asynchronous loading | 25.854  |

See full results in [benckmarks](./benchmarks).