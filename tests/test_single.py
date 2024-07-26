import torch.distributed as dist
import torch.multiprocessing as mp
from parquet_loader import ParquetDataset, ParquetDataLoader
parquet_path = 'synthetic_data'

if __name__ == '__main__':
    dataset = ParquetDataset(parquet_path, async_read=True)
    dataloader = ParquetDataLoader(dataset, batch_size=666, shuffle=False)
    for i, batch in enumerate(dataloader):
        # print(f"{i}, {batch.shape}")
        pass
