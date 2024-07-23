# Parquet Dataloader


This project is inspired by [litdata](https://github.com/Lightning-AI/litdata).
It implements a PyTorch Parquet dataset and dataloader that support streaming and distributed loading.

It highlights in:

* Streamingly load large parquet dataset, eg. Hive tables stored with parquet format. 

* Almost zero-redundancy loading across ranks & workers when distributed training.

* (TODO) Preload asynchronously for better efficiency.