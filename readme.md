# ParquetLoader


This project is inspired by [litdata](https://github.com/Lightning-AI/litdata).
It implements a torch dataset and dataloader that support streaming and distributed loading parquet dataset.

It highlights in:

* Streamingly load large parquet dataset, eg. Hive tables stored with parquet format. 
* Almost zero-redundancy loading across ranks & workers when distributed training.
* (TODO) Preload asynchronously for better efficiency.

It also has limitations:

* For small dataset its efficiency is lower than fully loading to memory.
* For dataset only contains 1 or few parquet file(s) and row group(s), it will degenrate to full loading, and even worse.
* Size of row group will affect the efficiency, it's suggested to set it to 1-1000 times of batch size.

## Installation

Install from source

``` shell 
git clone https://github.com/clearhanhui/ParquetLoader.git
cd ParquetLoader
pip install .
```


## Usage

```python 
from parquet_loader import ParquetDataset, ParquetDataLoader
dataset = ParquetDataset('/path/to/parquet/dataset')
dataloader = ParquetDataLoader(dataset)
```

## Benchmark

||||
|-|-|-| 
||Time(s)|Memory(MB)|
|Torch|3.041|153|
|ParquetLoader|7.290|610|


For more, see `test.py` and `benchmark.py`.